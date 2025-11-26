import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.uncertainty_metrics import compute_ece

def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calibrates temperature using grid search + LBFGS refinement."""
    # Fasr grid search for temperature
    original_probs = F.softmax(logits, dim=-1)
    original_nll = F.cross_entropy(logits, labels).item()
    original_ece = compute_ece(original_probs, labels)
    print(f"NLL before temperature scaling = {original_nll:.4f}")
    print(f"ECE before temperature scaling = {original_ece:.4f}")
    best_nll = float("inf")
    best_T = 1.0
    for T in torch.arange(0.05, 5.05, 0.05):
        T = T.item()
        loss = F.cross_entropy(logits / T, labels).item()
        if loss < best_nll:
            best_nll = loss
            best_T = T
    print(f"Grid search best T = {best_T:.3f}, NLL = {best_nll:.4f}")
    
    # Refine with LBFGS
    print(f"Use LBFGS to find a fine-grained temperature")
    temp_tensor = torch.tensor([best_T], requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([temp_tensor], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temp_tensor, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    
    T_refined = temp_tensor.detach().item()

    final_nll = F.cross_entropy(logits / T_refined, labels).item()
    scaled_ece = compute_ece(F.softmax(logits / T_refined, dim=-1), labels)
    print(f"Refined T = {T_refined:.4f}")
    print(f"NLL after temperature scaling = {final_nll:.4f}")
    print(f"ECE after temperature scaling = {scaled_ece:.4f}")
    
    return T_refined