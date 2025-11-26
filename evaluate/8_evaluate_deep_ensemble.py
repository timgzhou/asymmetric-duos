import os
import sys
import argparse
import torch
import pandas as pd
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uncertainty_metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--backbone_csv_path", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--save_dir", default="evaluation/eval_res")
    args = parser.parse_args()
    print("\n\n=============CONFIGS===========\n")
    print(args)
    print("\n=============CONFIGS===========\n\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    label_path = os.path.join(args.prediction_dir, "..", "point_prediction.csv")
    labels = pd.read_csv(label_path)["target"]
    labels = torch.tensor(labels.values, dtype=torch.long, device=device)

    backbone_df = pd.read_csv(args.backbone_csv_path)
    
    os.makedirs(args.save_dir, exist_ok=True)
    eval_path = os.path.join(args.save_dir, f"evaluation_deep_ensemble_{args.dataset_name}.csv")

    def append_to_csv(row_dicts):
        df_out = pd.DataFrame(row_dicts)
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        file_exists = os.path.exists(eval_path)
        file_has_data = os.path.getsize(eval_path) > 0 if file_exists else False
        df_out.to_csv(eval_path, mode='a', index=False, header=not file_has_data)


    def extract_m(file):
        name = file.replace(".csv", "")
        *_, m_str = name.split("_")
        return int(m_str) if m_str.isdigit() else float("inf")  # put non-matched files at the end

    for file in sorted(os.listdir(args.prediction_dir), key=extract_m):
        print(f"working on {file=}")
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(args.prediction_dir, file)
        model_name = file.replace(".csv", "")

        # Parse model and member count
        if "_" not in model_name:
            print(f"⚠️ Unexpected file name format: {file}, skipping.")
            continue

        # Assume last underscore group is the member count
        *model_parts, m_str = model_name.split("_")
        if not m_str.isdigit():
            print(f"⚠️ Could not extract member count from {model_name}, skipping.")
            continue
        base_model = "_".join(model_parts)
        print(f"extracted {base_model=}")

        model_info = backbone_df[backbone_df["model_name"] == base_model]
        if model_info.empty:
            print(f"⚠️ No backbone info for {base_model}, skipping.")
            continue
        model_info = model_info.iloc[0]

        logits = pd.read_csv(file_path, header=None).values
        logits = torch.tensor(logits, dtype=torch.float32, device=device)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        num_classes = probs.shape[1]

        uncert_sr = 1 - probs.max(dim=1).values
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
        
        row_dicts = []
        for uncertainty_type, uncertainties in [("softmax_response", uncert_sr), ("entropy", entropy)]:
            metrics = compute_metrics(probs, preds, labels, uncertainties, num_classes)
            metrics.update({
                "model": model_name,
                "wrapper": "LogitAverage",
                "uncertainty_type": uncertainty_type,
                "gflops": model_info["GFLOPS"],
                "params": model_info["Params"],
                "split": "test"
            })
            row_dicts.append(metrics)
            
        append_to_csv(row_dicts)
        print(f"{base_model} ensemble m={m_str} got {metrics=}")


if __name__ == "__main__":
    main()
