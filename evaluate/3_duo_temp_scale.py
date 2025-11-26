import torch
import os
import sys
import pandas as pd
import argparse
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
def jointly_calibrate_temperature(logits_l, logits_s, labels):
    print("=====Joint temperature calibration in progress...=====")
    best_nll = float("inf")
    best_Tl, best_Ts = 1.0, 1.0

    for Tl in torch.arange(0.05, 5.05, 0.2):
        for Ts in torch.arange(0.05, 5.05, 0.2):
            logits_avg = (logits_l / Tl + logits_s / Ts) / 2
            nll = F.cross_entropy(logits_avg, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_Tl, best_Ts = Tl.item(), Ts.item()

    print(f"Grid best Tl={best_Tl:.2f}, Ts={best_Ts:.2f}, NLL={best_nll:.4f}")

    Tl = torch.tensor([best_Tl], requires_grad=True, device=logits_l.device)
    Ts = torch.tensor([best_Ts], requires_grad=True, device=logits_s.device)
    optimizer = torch.optim.LBFGS([Tl, Ts], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        logits_avg = (logits_l / Tl + logits_s / Ts) / 2
        loss = F.cross_entropy(logits_avg, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_Tl, final_Ts = Tl.item(), Ts.item()
    print(f"Refined Tl={final_Tl:.4f}, Ts={final_Ts:.4f}")
    print(f"Final NLL = {F.cross_entropy((logits_l / Tl + logits_s / Ts)/2, labels).item():.4f}")

    print("=====Joint calibration complete and models wrapped.=====")
    return final_Tl,final_Ts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--duo_csv_path",type=str, required=True)
    args = parser.parse_args()
    print("\n\n=============CONFIGS===========\n")
    print(args)
    print("\n=============CONFIGS===========\n\n")
    
    duo_df = pd.read_csv(args.duo_csv_path)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split="val"
    all_results = []
    for index, row in duo_df.iterrows():
        print(row['model_large'], row['model_small'])
        curr_model_small = row['model_small']
        curr_model_large = row['model_large']
        small_pred_dir = os.path.join(args.prediction_dir_path,f"{curr_model_small}.csv")
        large_pred_dir = os.path.join(args.prediction_dir_path,f"{curr_model_large}.csv")
        if not os.path.isfile(large_pred_dir):
            print(f"skipping duo with {curr_model_large=} since prediction file doesn't exist.")
            continue
        elif not os.path.isfile(small_pred_dir):
            print(f"skipping duo with {curr_model_small=} since prediction file doesn't exist.")
            continue
        pred_small = pd.read_csv(small_pred_dir)
        logits_small = torch.tensor(pred_small.values, dtype=torch.float32, device=device)
        pred_large = pd.read_csv(large_pred_dir)
        logits_large = torch.tensor(pred_large.values, dtype=torch.float32, device=device)
        target_path = os.path.join(f"y-prediction/{args.dataset_name}/val","point_prediction.csv")
        target = pd.read_csv(target_path)["target"]
        target = torch.tensor(target.values, dtype=torch.long, device=device)
        opt_temp_l,opt_temp_s = jointly_calibrate_temperature(logits_large, logits_small, target)
        all_results.append({
            "model_large": curr_model_large,
            "model_small": curr_model_small,
            "temperature_large": opt_temp_l,
            "temperature_small": opt_temp_s,
        })

    save_dir = f"checkpoints/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "temperature_duo.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"{index+1}/{len(duo_df)}Saved calibrated duo temperatures to: {save_path}")

if __name__ == "__main__":
    main()
