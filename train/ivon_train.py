import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from utils.uncertainty_metrics import compute_metrics
import wandb
import ivon
import pandas as pd
import copy

from load_models import get_model_with_head
from load_data import get_dataloaders
from utils import load_config

def predict_with_ivon(model, optimizer, config_ivon, val_dataloader, device, save_logits_dir=None):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            sampled_probs = []
            for i in range(config_ivon["test_samples"]):
                with optimizer.sampled_params():
                    sampled_logit = model(x)
                    sampled_probs.append(F.softmax(sampled_logit, dim=1))
            logits = torch.mean(torch.stack(sampled_probs), dim=0)
            _, prediction = logits.max(1)
            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    metric = f1_score(all_labels, all_preds, average="macro") if config_ivon["dataset_name"] == "iwildcam" else accuracy_score(all_labels, all_preds)
    if save_logits_dir:
        os.makedirs(os.path.dirname(save_logits_dir), exist_ok=True)
        pd.DataFrame(all_logits.numpy()).to_csv(save_logits_dir,index=False)
    return metric

def train_with_ivon(config_ivon, root_dir, device, model_state):
    from load_data.datasets import IWildCamDataset, Caltech256Dataset
    dataset_cls = IWildCamDataset if config_ivon["dataset_name"] == 'iwildcam' else Caltech256Dataset
    num_classes = dataset_cls.num_classes
    model, transforms = get_model_with_head(
        model_name=config_ivon['model_name'],
        num_classes=num_classes,
        source=config_ivon.get('source', 'torchvision'),
        tv_weights=config_ivon.get('tv_weights', 'DEFAULT'),
        freeze=False
    )
    model.to(device)
    wandb_logging=True

    wandb_settings = wandb.Settings(
        init_timeout=120, # sec
        _service_wait=300
    )
    try:
        wandb.login()
        wandb.init(project=f"{config_ivon['dataset_name']}-Camera-Ready", id=config_ivon['model_name']+"(IVON)"+config_ivon["learning_rate"], config=config_ivon, reinit=True, settings=wandb_settings)
    except:
        print("=======ERROR WandB=======\nConnecting to WandB failed skipping WandB.")
        wandb_logging=False
    model.load_state_dict(model_state)

    dataloaders = get_dataloaders(config_ivon["dataset_name"], root_dir, batch_size=config_ivon["batch_size"],
                                  num_workers=config_ivon["num_workers"], transforms=transforms)
    optimizer = ivon.IVON(model.parameters(), lr=config_ivon["learning_rate"], ess=(len(dataloaders["train"])*config_ivon["batch_size"]))
    criterion = nn.CrossEntropyLoss()
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=config_ivon["warmup_epochs"]),
            CosineAnnealingLR(optimizer, T_max=config_ivon["num_epochs"] - config_ivon["warmup_epochs"])
        ],
        milestones=[config_ivon["warmup_epochs"]]
    )
    best_val_metric = -float("inf")
    best_state_dict = None
    val_metric = predict_with_ivon(model,optimizer,config_ivon,dataloaders["val"],device,save_logits_dir)
    print(f"pre-training {val_metric=}")
    for epoch in range(config_ivon["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch in dataloaders["train"]:
            x, y = batch
            x, y = x.to(device), y.to(device)
            for _ in range(config_ivon["train_samples"]):
                with optimizer.sampled_params(train=True):
                    optimizer.zero_grad()
                    logits = model(x)
                    if isinstance(logits, dict):  # unwrap if model returns a dict
                        logits = logits["logit"]
                    loss = criterion(logits, y)
                    loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if scheduler: scheduler.step()
        if epoch<config_ivon["num_epochs"]-1:
            save_logits_dir=None
        else: save_logits_dir=f"y-prediction/{config_ivon['dataset_name']}/val/raw/{config_ivon['model_name']}_ivon.csv"
        val_metric = predict_with_ivon(model,optimizer,config_ivon,dataloaders["val"],device,save_logits_dir)
        print(f"{epoch = } Val EVAL: {val_metric=}")
        if wandb_logging:
            wandb.log({
                "val_metric": val_metric,
                "epoch": epoch,
                "train_loss": epoch_loss / len(dataloaders["train"]),
                "lr": scheduler.get_last_lr()[0],
                "final_batch_size": config_ivon["batch_size"]
            })
        if val_metric > best_val_metric:
            print(f"New best model found at epoch {epoch} with val_metric={val_metric:.4f} (previous {best_val_metric=})")
            best_val_metric = val_metric
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        print("No improvement during training — using last epoch state.")
        
    test_metric = predict_with_ivon(model,optimizer,config_ivon,dataloaders["test"],device,
                                    save_logits_dir=f"y-prediction/{config_ivon['dataset_name']}/test/raw/{config_ivon['model_name']}_ivon.csv")
    print(config_ivon)
    print(f"TEST EVAL: {test_metric=}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--source", type=str, default="torchvision")
    parser.add_argument("--tv_weights", type=str, default="DEFAULT")
    parser.add_argument("--train_samples",type=int, default=1)
    parser.add_argument("--test_samples",type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    
    print(f"\n\n======IVON Configs======\n{args=}\n==============\n\n")
    dataset_name = args.dataset_name
    training_cfg = load_config(os.path.join(args.config_dir, f"{dataset_name}.yaml"))
    root_dir = args.dataset_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp_checkpoint_dst = f"checkpoints/{dataset_name}/lp/{args.model_name}.pth"
    ff_checkpoint_dst = f"checkpoints/{dataset_name}/ff/{args.model_name}.pth"
    if (os.path.exists(ff_checkpoint_dst)):
        print("ff checkpoint exists, skipping.")
        model_state = torch.load(ff_checkpoint_dst)
    elif (os.path.exists(lp_checkpoint_dst)):
        print("lp checkpoint exists, skipping.")
        model_state = torch.load(lp_checkpoint_dst)
    else:
        raise RuntimeError(f"IVON (for Duos baseline purpose) needs to start with a post-lp checkpoint, but {lp_checkpoint_dst} does not exist.")
    #################### Finetune with IVON ####################
    config_ivon = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "learning_rate": args.learning_rate,
        "num_epochs": 36, # training_cfg["training"]["num_epochs_ff"],
        "num_workers": training_cfg["training"]["num_workers"],
        "batch_size": training_cfg["training"]["batch_size"],
        "grace_period": training_cfg["training"]["grace_period"],
        "warmup_epochs": training_cfg["training"]["warmup_epochs"],
        "source": args.source,
        "tv_weights": args.tv_weights,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples
    }
    print(config_ivon)
    trained_model = train_with_ivon(config_ivon, root_dir, device, model_state)
    
    ivon_checkpoint_dst = f"checkpoints/{args.dataset_name}/ivon/{args.model_name}.pth"
    os.makedirs(os.path.dirname(ivon_checkpoint_dst), exist_ok=True)
    torch.save(trained_model.state_dict(),ivon_checkpoint_dst)
    print(f"trained IVON model saved to {ivon_checkpoint_dst}")
    
    logits=pd.read_csv(f"y-prediction/{config_ivon['dataset_name']}/test/raw/{config_ivon['model_name']}_ivon.csv")
    num_classes=logits.shape[1]
    logits=torch.tensor(logits.values,dtype=torch.float32,device=device)
    probs= F.softmax(logits, dim=1)
    uncert_sr = 1 - probs.max(dim=1).values
    label_path=f"y-prediction/{config_ivon['dataset_name']}/test/point_prediction.csv"
    labels = pd.read_csv(label_path)["target"]
    labels = torch.tensor(labels.values, dtype=torch.long, device=device)
    preds = probs.argmax(dim=1)
    metrics=compute_metrics(probs, preds, labels, uncert_sr, num_classes)
    print(metrics)
        
if __name__ == "__main__":
    main()