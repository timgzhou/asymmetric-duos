import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calibrate_temperature

class TempScaleWrapper(nn.Module):
    def __init__(self, model: nn.Module, init_temp: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, x):
        logits = self.model(x)
        logits = logits["logit"] if isinstance(logits, dict) else logits
        return logits / self.temperature.to(logits.device)

    def calibrate_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Wrapper that calls the global calibrate_temperature() helper."""
        T_refined = calibrate_temperature(logits, labels)
        self.temperature.data.copy_(torch.tensor(T_refined, device=self.temperature.device))
        return T_refined
    
    def predict_with_uncertainty(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        certainty = probs.max(dim=-1).values
        return {
            "logit": logits,
            "probs": probs,
            "preds": preds,
            "uncertainty(softmax_response)": 1 - certainty
    }