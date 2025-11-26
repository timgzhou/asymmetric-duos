import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
import argparse
from laplace import Laplace
from utils.uncertainty_metrics import compute_metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataset_class, get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--hessian_structure", type=str, default="kron")
    parser.add_argument("--subset_of_weights", type=str, default="last_layer")
    args = parser.parse_args()
    print("\n\n=============CONFIGS===========\n")
    print(args)
    print("\n=============CONFIGS===========\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load dataset
    dataset_cls = get_dataset_class(args.dataset_name.lower())
    dataset = dataset_cls(args.dataset_dir)
    num_classes = dataset.num_classes
    keep_imagenet_head = (args.dataset_name.lower()=="imagenet")


    model, transforms = get_model_with_head(
        model_name=args.model_name,
        num_classes=num_classes,
        source="torchvision",
        freeze=False,
        keep_imagenet_head=keep_imagenet_head
    )
    ckpt = torch.load(args.trained_model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)

    dataloaders = get_dataloaders(
        args.dataset_name, 
        args.dataset_dir, 
        batch_size=32, 
        num_workers=4, 
        transforms=transforms)

    la = Laplace(model, "classification",
            subset_of_weights=args.subset_of_weights,
            hessian_structure=args.hessian_structure)
    la.fit(dataloaders["train"])
    la.optimize_prior_precision(
        method="gridsearch",
        pred_type="glm",
        link_approx="probit",
        val_loader=dataloaders["val"]
    )

    for val_set in ["val","test","ood_test"]:
        if val_set not in dataloaders or not dataloaders[val_set]:
            print(f"{args.dataset_name} has no {val_set} set")
            continue
        else:
            all_preds = []
            with torch.no_grad():
                for x, y in dataloaders[val_set]:
                    x, y = x.to(device), y.to(device)
                    pred = la(x, pred_type="glm", link_approx="probit")
                    all_preds.append(pred.cpu())
            logits = torch.cat(all_preds)
            prediction_save_dir = f"y-prediction/{args.dataset_name}/{val_set}/raw"
            os.makedirs(prediction_save_dir,exist_ok=True)
            prediction_save_path = os.path.join(prediction_save_dir,f"{args.model_name}_{args.hessian_structure}_{args.subset_of_weights}_LA.csv")
            pd.DataFrame(logits.cpu().numpy()).to_csv(prediction_save_path,index=False)
            print(f"{val_set=} predictions saved to {prediction_save_path}")
            
            if val_set=="test":
                probs=pd.read_csv(prediction_save_path)
                num_classes=probs.shape[1]
                probs=torch.tensor(probs.values,dtype=torch.float32,device=device)
                uncert_sr = 1 - probs.max(dim=1).values
                label_path=f"y-prediction/{args.dataset_name}/{val_set}/point_prediction.csv"
                labels = pd.read_csv(label_path)["target"]
                labels = torch.tensor(labels.values, dtype=torch.long, device=device)
                preds = probs.argmax(dim=1)
                metrics=compute_metrics(probs, preds, labels, uncert_sr, num_classes)
                print("\n<<<<<<<<<<RESULT>>>>>>>>>>>\n=============CONFIGS===========\n")
                print(args)
                print(metrics)
                print("\n<<<<<<<<<<RESULT>>>>>>>>>>>\n")

if __name__ == "__main__":
    main()
