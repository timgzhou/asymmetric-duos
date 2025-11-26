import torch
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_csv", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, default="checkpoints/imagenet")
    args = parser.parse_args()
    backbone = pd.read_csv(args.backbone_csv)
    total_num_models = backbone.shape[0]
    os.makedirs(args.model_save_dir,exist_ok=True)
    for index, row in backbone.iterrows():
        model_name = row['model_name']
        print(model_name, row['source'], row["tv_weights"])
        model, _ = get_model_with_head(
            model_name=model_name,
            num_classes=None,
            source=row['source'],
            tv_weights=row["tv_weights"],
            keep_imagenet_head=True
        )
        save_path = os.path.join(args.model_save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"({index+1}/{total_num_models}) Imagenet model saved to {save_path}")

if __name__ == "__main__":
    main()
