import torch
import os
import sys
import pandas as pd
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataset_class, get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_path", type=str, required=True)
    parser.add_argument("--backbone_csv", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ood_dataset_dir", type=str, default=None)
    parser.add_argument("--mode", default="single model", choices=["single model", "soup"],
                    help="Mode for loading models: 'single model', 'soup'.")
    parser.add_argument("--feature_extractor_mode", action='store_true')
    args = parser.parse_args()
    print("\n\n=============CONFIGS===========\n")
    print(args)
    print("\n=============CONFIGS===========\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_df = pd.read_csv(args.backbone_csv)
    print("Backbones csv:")
    print(backbone_df.head())
    
    # 1. Load dataset
    dataset_cls = get_dataset_class(args.dataset_name.lower())
    dataset = dataset_cls(args.dataset_dir)
    num_classes = dataset.num_classes
    keep_imagenet_head = (args.dataset_name.lower()=="imagenet")
    print(f"{keep_imagenet_head=}")
    
    # iterate through available models
    model_paths = [f for f in os.listdir(args.model_dir_path) if os.path.isfile(os.path.join(args.model_dir_path, f))]

    total_model = len(model_paths)
    print(f"{model_paths = }")
    for i,model_path in enumerate(model_paths):
        full_name = model_path.split(".pth")[0]
        # Extract base model_name
        if "greedy" in full_name:
            model_name = "_".join(full_name.split("_")[:-1])  # remove last _suffix
            print(f"soup detected for {full_name=}, extracted {model_name=}")
        elif args.mode == "single model":
            model_name = full_name
        else:
            raise ValueError(f"Unsupported mode {args.mode}")
        try:
            source = backbone_df[backbone_df["model_name"] == model_name]["source"].values[0]
        except IndexError:
            print(f"backbone df has no {model_name} row")
            continue

        model, transforms = get_model_with_head(
            model_name=model_name,
            num_classes=num_classes,
            source=source,
            freeze=False,
            keep_imagenet_head=keep_imagenet_head
        )
        ckpt = torch.load(os.path.join(args.model_dir_path, model_path), map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model.to(device)

        # 3. Get DataLoader
        dataloaders = get_dataloaders(
            args.dataset_name, 
            args.dataset_dir, 
            batch_size=32, 
            num_workers=4, 
            transforms=transforms, 
            ood_root_dir=args.ood_dataset_dir)

        for val_set in ["val","test","ood_test"]:
            if val_set not in dataloaders or not dataloaders[val_set]:
                print(f"{args.dataset_name} has no {val_set} set")
                continue
            if args.feature_extractor_mode:
                logits_save_dir = f"y-prediction/{args.dataset_name}/{val_set}/features"
            else: logits_save_dir = f"y-prediction/{args.dataset_name}/{val_set}/raw"
            os.makedirs(logits_save_dir,exist_ok=True)
            prediction_save_path = os.path.join(logits_save_dir,f"{full_name}.csv")
            if os.path.exists(prediction_save_path):
                print(f"{prediction_save_path} already exists, skipping..")
                continue
            else:
                logits, labels = [], []
                with torch.no_grad():
                    for x, y in dataloaders[val_set]:
                        x, y = x.to(device), y.to(device)
                        if args.feature_extractor_mode:
                            out=model.forward_features_pooled(x)
                            if out.dim() > 2:
                                out = out.view(out.size(0), -1)
                        else: out = model(x)  # unscaled forward
                        logits.append(out["logit"] if isinstance(out, dict) else out)
                        labels.append(y)
                logits = torch.cat(logits)
                labels = torch.cat(labels)

                logits_df=pd.DataFrame(logits.cpu().numpy())
                logits_df.to_csv(prediction_save_path,index=False)
                print(f"logits_df has dimension {logits_df.shape}, saved to {prediction_save_path}")
                
                if not args.feature_extractor_mode:
                    point_prediction_save_dir = f"y-prediction/{args.dataset_name}/{val_set}"
                    os.makedirs(point_prediction_save_dir,exist_ok=True)
                    point_prediction_save_path = os.path.join(point_prediction_save_dir,f"point_prediction.csv")
                    point_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    if os.path.exists(point_prediction_save_path):
                        df = pd.read_csv(point_prediction_save_path)
                        if "target" not in df.columns:
                            df.insert(0, "target", labels.cpu().numpy())
                        df[full_name] = point_preds
                    else:
                        df = pd.DataFrame({
                            "target": labels.cpu().numpy(),
                            full_name: point_preds
                        })
                    df.to_csv(point_prediction_save_path, index=False)
        print(f"{i}/{total_model} complete, model name = {model_name}")
                
            
if __name__ == "__main__":
    main()
