import os
import argparse
import torch
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataloaders
from load_data.datasets import IWildCamDataset, Caltech256Dataset

def load_model_state(model_name, num_classes, source, path):
    model, _ = get_model_with_head(model_name, num_classes, source)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

def evaluate_val_metric(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def make_uniform_soup(paths, model_name, num_classes, source, val_loader, device):
    print("Creating Uniform Soup...")
    soup_state = None
    count = 0
    for path in paths:
        model = load_model_state(model_name, num_classes, source, path)
        state_dict = model.state_dict()
        if soup_state is None:
            soup_state = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in soup_state:
                soup_state[k] += state_dict[k].float()
        count += 1
    for k in soup_state:
        soup_state[k] /= count

    model_ref, _ = get_model_with_head(model_name, num_classes, source)
    model_ref.load_state_dict(soup_state)
    acc = evaluate_val_metric(model_ref.to(device), val_loader, device)
    print(f"Uniform Soup ready. Val accuracy: {acc:.4f} | Models used: {count}")
    return model_ref

def make_greedy_soup(paths, model_name, num_classes, source, val_loader, device):
    print("Creating Greedy Soup...")
    base_model = load_model_state(model_name, num_classes, source, paths[0])
    best_val = evaluate_val_metric(base_model.to(device), val_loader, device)
    print(f"best trial has {best_val=}")
    soup_state = {k: v.clone().float() for k, v in base_model.state_dict().items()}
    soup_count = 1

    for path in paths[1:]:
        candidate_model = load_model_state(model_name, num_classes, source, path)
        candidate_state = {k: v.to(device) for k, v in candidate_model.state_dict().items()}
        temp_state = {
            k: (soup_state[k] * soup_count + candidate_state[k].float()) / (soup_count + 1)
            for k in soup_state
        }

        model_ref, _ = get_model_with_head(model_name, num_classes, source)
        model_ref.load_state_dict(temp_state)
        new_val = evaluate_val_metric(model_ref.to(device), val_loader, device)

        if new_val > best_val:
            soup_state = temp_state
            soup_count += 1
            best_val = new_val
            print(f"Y- Added model from {path}, new val acc = {new_val:.4f}")
        else:
            print(f"N- Skipped model from {path}, val acc = {new_val:.4f}")

    model_ref, _ = get_model_with_head(model_name, num_classes, source)
    model_ref.load_state_dict(soup_state)
    print(f"Greedy Soup ready. Final val acc: {best_val:.4f} | Models used: {soup_count}")
    return model_ref

def save_logit_ensemble_predictions(top_paths, model_name, num_classes, source, test_loader, device, dataset_name):
    for i in range(1, len(top_paths) + 1):
        models = [
            load_model_state(model_name, num_classes, source, p).to(device).eval()
            for p in top_paths[:i]
        ]
        all_logits = []

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                logits = torch.stack([m(x) for m in models], dim=0).mean(dim=0)  # [B, C]
                all_logits.append(logits.cpu())  # keep as tensor
        all_logits = torch.cat(all_logits, dim=0).numpy()

        # Save logits as raw values (no header or index)
        save_dir = os.path.join("y-prediction", dataset_name, "test", "deep_ensemble")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_{i}.csv")
        pd.DataFrame(all_logits).to_csv(save_path, index=False, header=False)
        print(f"Saved ensemble logits with {i} models to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--source", default="torchvision")
    parser.add_argument("--data_dir", required=True)

    args = parser.parse_args()
    soup_dir = os.path.join("checkpoints", args.dataset_name, "soup")
    save_dir = os.path.join("checkpoints", args.dataset_name, "ff")
    os.makedirs(save_dir, exist_ok=True)
    trial_csv = os.path.join(soup_dir, f"{args.model_name}",f"{args.model_name}_trials.csv")
    
    df = pd.read_csv(trial_csv)
    df_sorted = df.sort_values(by="val_metric", ascending=False) # sort needed for greedy soup
    paths = df_sorted["checkpoint_path"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cls = IWildCamDataset if args.dataset_name.lower() == "iwildcam" else Caltech256Dataset
    num_classes = dataset_cls.num_classes
    _, transforms = get_model_with_head(args.model_name, num_classes, args.source)
    val_loader = get_dataloaders(args.dataset_name, args.data_dir, 64, 4, transforms)["val"]

    # Uniform Soup
    uniform_model = make_uniform_soup(paths, args.model_name, num_classes, args.source, val_loader, device).to(device)
    uniform_soup_save_path = os.path.join(save_dir,f"{args.model_name}_uniform.pth")
    torch.save(uniform_model.state_dict(), uniform_soup_save_path)
    print(f"Saved uniform soup model to {uniform_soup_save_path}")

    # Greedy Soup
    greedy_model = make_greedy_soup(paths, args.model_name, num_classes, args.source, val_loader, device).to(device)
    greedy_soup_save_path = os.path.join(save_dir,f"{args.model_name}_greedy.pth")
    torch.save(greedy_model.state_dict(), greedy_soup_save_path)
    print(f"Saved greedy soup model to {greedy_soup_save_path}")

if __name__ == "__main__":
    main()
