import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score

def compute_metrics(probs, preds, labels, uncertainties, num_classes):
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()

    certainties = 1 - uncertainties
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    brier = brier_score_loss(
        y_true=np.eye(num_classes)[labels].reshape(-1),
        y_proba=probs.reshape(-1)
    )
    nll = log_loss(labels, probs, labels=list(range(num_classes)))
    ece = compute_ece(torch.tensor(probs), torch.tensor(labels))

    cp_auroc = roc_auc_score((preds == labels).astype(int), certainties)
    aurc, eaurc, sac_dict = compute_risk_coverage_metrics(labels, preds, uncertainties)

    metrics = {
        "Acc": acc,
        "F1": f1,
        "Brier": brier,
        "NLL": nll,
        "ECE": ece,
        "CP_AUROC": cp_auroc,
        "AURC": aurc,
        "E-AURC": eaurc,
    }
    metrics.update({f"SAC@{int(k*100)}": v for k, v in sac_dict.items()})
    return metrics

def compute_ece(probs, labels, n_bins=15):
    """Computes Expected Calibration Error (ECE)."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_accuracy = accuracies[mask].float().mean()
            bin_confidence = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_confidence - bin_accuracy)
    return ece.item()

def compute_risk_coverage_metrics(labels, preds, uncertainties, sac_thresholds=np.arange(0.90, 0.991, 0.01)):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    uncertainties = np.asarray(uncertainties)

    correctness = (preds == labels).astype(float)
    sorted_indices = np.argsort(uncertainties)  # sort by increasing uncertainty (most confident first)
    sorted_correctness = correctness[sorted_indices]

    # Risk = cumulative error rate as coverage increases
    risks = np.cumsum(1 - sorted_correctness) / (np.arange(1, len(correctness)+1))
    aurc = risks.mean()

    # Optimal AURC = always predict correct first (sorted_correctness = all 1s, then all 0s)
    ideal_correctness = np.sort(correctness)[::-1]  # all correct first
    optimal_risks = np.cumsum(1 - ideal_correctness) / (np.arange(1, len(correctness)+1))
    aurc_optimal = optimal_risks.mean()
    eaurc = aurc - aurc_optimal

    # SAC = max coverage while maintaining risk ≤ (1 - acc_target)
    sac = {}
    for t in sac_thresholds:
        allowed_risk = 1 - t
        valid = risks <= allowed_risk
        if not np.any(valid):
            sac[t] = 0.0
        else:
            max_idx = np.max(np.where(valid))
            sac[t] = (max_idx + 1) / len(correctness)

    return aurc, eaurc, sac


def evaluate_model(model, dataloader, device, distill=False):
    model.eval()
    all_preds, all_probs, all_logits, all_labels, all_uncertainties = [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if distill:
                x = batch["student_inputs"]
                y = batch["labels"]
            else:
                x, y = batch

            x, y = x.to(device), y.to(device)
            output = model.predict_with_uncertainty(x)

            all_preds.append(output["preds"].cpu())
            all_probs.append(output["probs"].cpu())
            all_logits.append(output["logit"].cpu())
            all_uncertainties.append(output["uncertainty(softmax_response)"].cpu())
            all_labels.append(y.cpu())

    return {
        "preds": torch.cat(all_preds),
        "probs": torch.cat(all_probs),
        "logits": torch.cat(all_logits),
        "uncertainties": torch.cat(all_uncertainties),
        "labels": torch.cat(all_labels),
    }
