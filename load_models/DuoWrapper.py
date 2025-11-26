# DuoWrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from load_models import TempScaleWrapper

def softmax_kl(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")

class DuoWrapper(nn.Module):
    def __init__(self, model_large: nn.Module, model_small: nn.Module, mode="unweighted", temp_large=1.0, temp_small=1.0):
        super().__init__()
        assert mode in ["unweighted", "uqonly", "asymmetric"]
        self.mode = mode
        
        # Remove TempScaleWrapper if already applied for fresh calibration
        if isinstance(model_large, TempScaleWrapper):
            print("⚠️ Unwrapping model_large for fresh temperature calibration.")
            model_large = model_large.model
        if isinstance(model_small, TempScaleWrapper):
            print("⚠️ Unwrapping model_small for fresh temperature calibration.")
            model_small = model_small.model

        self.model_large = model_large
        self.model_small = model_small
        
        # Store temperatures
        self.temp_large = temp_large
        self.temp_small = temp_small

    def forward(self, x):
        """Return logits for loss computation or evaluation."""
        logit_l = self.model_large(x["teacher_fl_inputs"])
        logit_s = self.model_small(x["teacher_fs_inputs"])
        
        if self.mode == "unweighted":
            # Simple average of logits
            return (logit_l + logit_s) / 2
        elif self.mode == "uqonly":
            # Use only FL predictions
            return logit_l
        elif self.mode == "asymmetric":
            # Apply separate temperatures then average
            scaled_logit_l = logit_l / self.temp_large
            scaled_logit_s = logit_s / self.temp_small
            return (scaled_logit_l + scaled_logit_s) / 2

    def predict_with_uncertainty(self, x):
        """Return detailed predictions and uncertainty scores."""
        logit_l = self.model_large(x)
        logit_s = self.model_small(x)
        
        if self.mode == "unweighted":
            # Simple average
            avg_logit = (logit_l + logit_s) / 2
            probs_avg = F.softmax(avg_logit, dim=-1)
            preds = probs_avg.argmax(dim=-1)
            cert_avg = probs_avg.max(dim=-1).values
            
            result = {
                "logit": avg_logit,
                "probs": probs_avg,
                "preds": preds,
                "uncertainty(softmax_response)": 1 - cert_avg,
            }
            
        elif self.mode == "uqonly":
            # FL predictions, but asymmetric uncertainty
            probs_l = F.softmax(logit_l, dim=-1)
            probs_s = F.softmax(logit_s, dim=-1)
            
            # Use FL for predictions
            preds = logit_l.argmax(dim=-1)
            
            # Asymmetric uncertainty: disagreement between models
            cert_l = probs_l.max(dim=-1).values
            cert_s = probs_s.max(dim=-1).values
            
            # Uncertainty based on disagreement and individual confidences
            kl_div = F.kl_div(F.log_softmax(logit_l, dim=-1), 
                             F.softmax(logit_s, dim=-1), 
                             reduction='none').sum(dim=-1)
            uncertainty = 1 - cert_l + 0.1 * kl_div  # Combine individual and disagreement
            
            result = {
                "logit": logit_l,
                "probs": probs_l,
                "preds": preds,
                "uncertainty(softmax_response)": uncertainty,
            }
            
        elif self.mode == "asymmetric":
            # Apply temperatures and average
            scaled_logit_l = logit_l / self.temp_large
            scaled_logit_s = logit_s / self.temp_small
            avg_logit = (scaled_logit_l + scaled_logit_s) / 2
            
            probs_avg = F.softmax(avg_logit, dim=-1)
            preds = probs_avg.argmax(dim=-1)
            cert_avg = probs_avg.max(dim=-1).values
            
            # Asymmetric uncertainty considering temperature scaling
            probs_l = F.softmax(scaled_logit_l, dim=-1)
            probs_s = F.softmax(scaled_logit_s, dim=-1)
            
            kl_div = F.kl_div(F.log_softmax(scaled_logit_l, dim=-1), 
                             probs_s, reduction='none').sum(dim=-1)
            uncertainty = 1 - cert_avg + 0.1 * kl_div
            
            result = {
                "logit": avg_logit,
                "probs": probs_avg,
                "preds": preds,
                "uncertainty(softmax_response)": uncertainty,
            }

        return result

    def find_joint_temperatures(self, val_loader, device):
        """
        Find optimal temperatures for both models jointly using validation data
        """
        print("🎯 Finding joint temperatures...")
        
        # Collect all logits and labels
        all_logits_l = []
        all_logits_s = []
        all_labels = []
        
        self.model_large.eval()
        self.model_small.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    # Multi-transform batch
                    inputs = batch['teacher_inputs'].to(device)
                    labels = batch['labels'].to(device)
                else:
                    # Standard batch
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                
                logits_l = self.model_large(inputs)
                logits_s = self.model_small(inputs)
                
                all_logits_l.append(logits_l)
                all_logits_s.append(logits_s)
                all_labels.append(labels)
        
        # Concatenate all batches
        logits_l = torch.cat(all_logits_l, dim=0)
        logits_s = torch.cat(all_logits_s, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Grid search for initial temperatures
        best_nll = float("inf")
        best_Tl, best_Ts = 1.0, 1.0
        
        print("Starting grid search...")
        for Tl in torch.arange(0.1, 5.1, 0.2):
            for Ts in torch.arange(0.1, 5.1, 0.2):
                # Average temperature-scaled logits
                logits_avg = (logits_l / Tl + logits_s / Ts) / 2
                nll = F.cross_entropy(logits_avg, labels).item()
                if nll < best_nll:
                    best_nll = nll
                    best_Tl, best_Ts = Tl.item(), Ts.item()
        
        print(f"Grid search best: Tl={best_Tl:.2f}, Ts={best_Ts:.2f}, NLL={best_nll:.4f}")
        
        # Fine-tune with gradient descent
        Tl = torch.tensor([best_Tl], requires_grad=True, device=device)
        Ts = torch.tensor([best_Ts], requires_grad=True, device=device)
        optimizer = torch.optim.LBFGS([Tl, Ts], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            logits_avg = (logits_l / Tl + logits_s / Ts) / 2
            loss = F.cross_entropy(logits_avg, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Update temperatures
        self.temp_large = max(0.1, Tl.item())  # Ensure positive temperatures
        self.temp_small = max(0.1, Ts.item())
        
        # Final NLL calculation
        final_logits = (logits_l / self.temp_large + logits_s / self.temp_small) / 2
        final_nll = F.cross_entropy(final_logits, labels).item()
        
        print(f"Final temperatures: Tl={self.temp_large:.4f}, Ts={self.temp_small:.4f}")
        print(f"Final NLL: {final_nll:.4f}")
        print("✅ Joint temperature calibration complete.")
        
        return self.temp_large, self.temp_small