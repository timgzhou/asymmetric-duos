import torch
import os
import sys
import pandas as pd
import argparse
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    original_nll = F.cross_entropy(logits, labels).item()
    print(f"NLL before temperature scaling = {original_nll:.4f}")
    best_nll = float("inf")
    best_T = 1.0
    for T in torch.arange(0.05, 5.05, 0.05):
        T = T.item()
        loss = F.cross_entropy(logits / T, labels).item()
        if loss < best_nll:
            best_nll = loss
            best_T = T
    print(f"Grid search best T = {best_T:.3f}, NLL = {best_nll:.4f}")

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
    print(f"Refined T = {T_refined:.4f}")
    print(f"NLL after temperature scaling = {final_nll:.4f}")
    return T_refined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    print("\n\n=============CONFIGS===========\n")
    print(args)
    print("\n=============CONFIGS===========\n\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # iterate through available predictions
    prediction_paths = [f for f in os.listdir(args.prediction_dir_path) if os.path.isfile(os.path.join(args.prediction_dir_path, f))]


    all_results = []

    for prediction_csv_path in prediction_paths:
        full_name = prediction_csv_path.split(".csv")[0]

        logits_path = os.path.join(args.prediction_dir_path, prediction_csv_path)
        logits = pd.read_csv(logits_path)
        logits = torch.tensor(logits.values, dtype=torch.float32, device=device)

        target_path = f"y-prediction/{args.dataset_name}/val/point_prediction.csv"
        target = pd.read_csv(target_path)["target"]
        target = torch.tensor(target.values, dtype=torch.long, device=device)

        opt_temp = calibrate_temperature(logits, target)

        all_results.append({
            "full_name": full_name,
            "temperature": opt_temp
        })

    # Save the temperatures
    save_dir = f"checkpoints/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "temperature_single_model.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved calibrated temperatures to: {save_path}")

if __name__ == "__main__":
    main()
