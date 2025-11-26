import argparse
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import wandb
import shutil
import re
import csv
import pandas as pd


from load_models import get_model_with_head
from load_data import get_dataloaders
from utils import load_config

def train_model(config, root_dir, dataset_name, device, model_state=None, finetune=False):
    from load_data.datasets import IWildCamDataset, Caltech256Dataset
    dataset_cls = IWildCamDataset if dataset_name.lower() == 'iwildcam' else Caltech256Dataset
    num_classes = dataset_cls.num_classes
    model, transforms = get_model_with_head(
        model_name=config['model_name'],
        num_classes=num_classes,
        source=config.get('source', 'torchvision'),
        tv_weights=config.get('tv_weights', 'DEFAULT'),
        freeze=not finetune,
    )
    model.to(device)

    wandb_logging=False
    if finetune:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_HTTP_TIMEOUT"] = "120"
        wandb_settings = wandb.Settings(
            init_timeout=300,  # 5 minutes
            _service_wait=300
        )
        try:
            wandb.login()
            wandb.init(project=f"{dataset_name}-Camera-Ready", id=config['model_name'], config=config, reinit=True, settings=wandb_settings)
            wandb_logging=True
        except:
            os.environ["WANDB_MODE"] = "offline"
            print("Connecting to WandB failed skipping WandB.")
            wandb_logging=False
            # wandb.init(project=f"{dataset_name}-Camera-Ready", id=config['model_name'], config=config, reinit=True, mode="offline", settings=wandb_settings)
    if model_state:
        model.load_state_dict(model_state)


    dataloaders = get_dataloaders(dataset_name, root_dir, batch_size=config.get("batch_size", 16),
                                  num_workers=config["num_workers"], transforms=transforms)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    if finetune:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]),
                CosineAnnealingLR(optimizer, T_max=config["num_epochs"] - config["warmup_epochs"])
            ],
            milestones=[config["warmup_epochs"]]
        )
    else:
        scheduler = None

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch in dataloaders["train"]:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if isinstance(logits, dict):  # unwrap if model returns a dict
                logits = logits["logit"]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if scheduler: scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloaders["val"]:
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        metric = f1_score(all_labels, all_preds, average="macro") if dataset_name == "iwildcam" else accuracy_score(all_labels, all_preds)
        if wandb_logging:
            wandb.log({
                "val_metric": metric,
                "epoch": epoch,
                "train_loss": epoch_loss / len(dataloaders["train"]),
                "lr": scheduler.get_last_lr()[0],
                "final_batch_size": config["batch_size"]
            })
        if ((epoch % 3 == 0) or (epoch == config["num_epochs"] - 1)):
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = os.path.join(temp_dir, "model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                checkpoint = tune.Checkpoint.from_directory(temp_dir)
                tune.report(
                    {
                        "val_metric": metric,
                        "batch_size": config["batch_size"],
                        "training_iteration": epoch
                    },
                    checkpoint=checkpoint
                )
    if wandb_logging: wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--source", type=str, default="torchvision")
    parser.add_argument("--tv_weights", type=str, default="DEFAULT")
    parser.add_argument("--num_samples",type=int, default=12)
    parser.add_argument("--skip_lp_if_exist",type=bool,default=True)
    args = parser.parse_args()
    
    print(f"\n\n======hp_tune_configs======\n{args=}\n==============\n\n")

    training_cfg = load_config(os.path.join(args.config_dir, f"{args.dataset_name}.yaml"))
    root_dir = args.dataset_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_metric = "val_metric"

    reporter = CLIReporter(
        parameter_columns=['learning_rate', 'weight_decay'],
        metric_columns=[validation_metric, 'training_iteration'],
        max_report_frequency=900
    )
    
    #################### Linear Probing ####################
    optimal_csv_path = f"checkpoints/{args.dataset_name}/optimal_hyperparams_lp.csv"
    lp_checkpoint_dst = f"checkpoints/{args.dataset_name}/lp/{args.model_name}.pth"
    if (os.path.exists(lp_checkpoint_dst) and args.skip_lp_if_exist):
        print("lp checkpoint exists, skipping.")
    else:
        config_lp = {
            "model_name": args.model_name,
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "weight_decay": 0,
            "num_epochs": training_cfg["training"]["num_epochs_lp"],
            "num_workers": training_cfg["training"]["num_workers"],
            "batch_size": training_cfg["training"]["batch_size"],
            "source": args.source,
            "tv_weights": args.tv_weights,
        }
        result = tune.run(
            tune.with_parameters(
                train_model,
                root_dir=root_dir,
                dataset_name=args.dataset_name,
                device=device,
                finetune=False
            ),
            config=config_lp,
            num_samples=4,
            resources_per_trial={"cpu": 4, "gpu": 1},
            scheduler=ASHAScheduler(max_t=config_lp["num_epochs"],grace_period=config_lp["num_epochs"],metric=validation_metric, mode="max"),
            storage_path=os.path.abspath("./ray_results"),
            name=f"lp_search_{args.model_name}_{args.dataset_name}",
            progress_reporter=reporter,
            log_to_file=True,
            checkpoint_score_attr=validation_metric,
            keep_checkpoints_num=5,
            verbose=0
        )
        best_trial = result.get_best_trial(validation_metric, "max")
        print(f"Best LP metric: {best_trial.last_result[validation_metric]:.4f}")
        best_checkpoint = result.get_best_checkpoint(best_trial, metric=validation_metric, mode="max")
        
        os.makedirs(os.path.dirname(lp_checkpoint_dst), exist_ok=True)

        best_metric = best_trial.last_result[validation_metric]

        # Check if existing best is better
        overwrite = True
        if os.path.exists(optimal_csv_path):
            df_existing = pd.read_csv(optimal_csv_path)
            df_existing_model = df_existing[
                (df_existing["model_name"] == args.model_name)
            ]
            if not df_existing_model.empty:
                existing_metric = df_existing_model.iloc[0]["val_metric"]
                overwrite = best_metric > existing_metric
                if not overwrite:
                    print(f"Existing LP ({args.model_name}) has better metric ({existing_metric:.4f}), not overwriting.")

        if overwrite:
            with best_checkpoint.as_directory() as checkpoint_dir:
                model_path = os.path.join(checkpoint_dir, "model.pth")
                shutil.copy(model_path, lp_checkpoint_dst)
                print(f"Saved best LP model to: {lp_checkpoint_dst}")
            
            new_record = {
                "model_name": args.model_name,
                "learning_rate": best_trial.config["learning_rate"],
                "weight_decay": best_trial.config["weight_decay"],
                "batch_size": best_trial.last_result.get("batch_size", -1),
                "val_metric": best_metric,
                "checkpoint_path": lp_checkpoint_dst,
            }

            if os.path.exists(optimal_csv_path):
                # Drop existing record for this model_name
                df_existing = df_existing.drop(
                    df_existing[(df_existing["model_name"] == args.model_name)].index
                )
                optimal_df = pd.concat([df_existing, pd.DataFrame([new_record])], ignore_index=True)
            else:
                optimal_df = pd.DataFrame([new_record])

            optimal_df.to_csv(optimal_csv_path, index=False)
            print(f"Updated optimal LP hyperparams to: {optimal_csv_path}")


    #################### Finetune ####################
    best_state = torch.load(lp_checkpoint_dst)
    config_ft = {
        "model_name": args.model_name,
        "learning_rate": tune.loguniform(1e-6, 3e-4),
        "weight_decay": tune.loguniform(1e-8, 1e-5),
        "num_epochs": training_cfg["training"]["num_epochs_ff"],
        "num_workers": training_cfg["training"]["num_workers"],
        "batch_size": training_cfg["training"]["batch_size"],
        "grace_period": training_cfg["training"]["grace_period"],
        "warmup_epochs": training_cfg["training"]["warmup_epochs"],
        "source": args.source,
        "tv_weights": args.tv_weights,
    }
    ft_result = tune.run(
        tune.with_parameters(
            train_model,
            root_dir=root_dir,
            dataset_name=args.dataset_name,
            device=device,
            model_state=best_state,
            finetune=True
        ),
        config=config_ft,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": 4, "gpu": 1},
        scheduler=ASHAScheduler(max_t=config_ft["num_epochs"],grace_period=config_ft["grace_period"],metric=validation_metric, mode="max"),
        storage_path=os.path.abspath("./ray_results"),
        name=f"ff_search_{args.model_name}_{args.dataset_name}",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
        fail_fast=False,
        log_to_file=True,
        checkpoint_score_attr=validation_metric,
        keep_checkpoints_num=1,
        resume="AUTO"
    )
    
    # Filter successful trials
    successful_trials = [t for t in ft_result.trials if t.status == "TERMINATED" and t.checkpoint]
    failed_trials = [t for t in ft_result.trials if t not in successful_trials]

    print(f"{len(successful_trials)} successful trials")
    print(f"{len(failed_trials)} failed trials")

    if not successful_trials:
        print("No successful trials found. Skipping checkpoint and soup saving.")
        return

    ft_best_trial = max(successful_trials, key=lambda t: t.last_result.get(validation_metric, float("-inf")))
    print(f"Best FT metric: {ft_best_trial.last_result[validation_metric]:.4f}")

    ft_best_checkpoint = ft_result.get_best_checkpoint(ft_best_trial, metric=validation_metric, mode="max")
    optimal_csv_path = f"checkpoints/{args.dataset_name}/optimal_hyperparams_ff.csv"
    ft_checkpoint_dst = f"checkpoints/{args.dataset_name}/ff/{args.model_name}.pth"

    os.makedirs(os.path.dirname(ft_checkpoint_dst), exist_ok=True)

    best_metric = ft_best_trial.last_result[validation_metric]

    # Check if existing best is better
    overwrite = True
    if os.path.exists(optimal_csv_path):
        df_existing = pd.read_csv(optimal_csv_path)
        df_existing_model = df_existing[
            (df_existing["model_name"] == args.model_name)
        ]
        if not df_existing_model.empty:
            existing_metric = df_existing_model.iloc[0]["val_metric"]
            overwrite = best_metric > existing_metric
            if not overwrite:
                print(f"Existing FF ({args.model_name}) has better metric ({existing_metric:.4f}), not overwriting.")

    if overwrite:
        with ft_best_checkpoint.as_directory() as checkpoint_dir:
            model_path = os.path.join(checkpoint_dir, "model.pth")
            shutil.copy(model_path, ft_checkpoint_dst)
            print(f"Saved best FF model to: {ft_checkpoint_dst}")
        
        new_record = {
            "model_name": args.model_name,
            "learning_rate": ft_best_trial.config["learning_rate"],
            "weight_decay": ft_best_trial.config["weight_decay"],
            "batch_size": ft_best_trial.last_result.get("batch_size", -1),
            "val_metric": best_metric,
            "checkpoint_path": ft_checkpoint_dst
        }

        if os.path.exists(optimal_csv_path):
            # Drop existing record for this model_name
            df_existing = df_existing.drop(
                df_existing[(df_existing["model_name"] == args.model_name)].index
            )
            optimal_df = pd.concat([df_existing, pd.DataFrame([new_record])], ignore_index=True)
        else:
            optimal_df = pd.DataFrame([new_record])

        optimal_df.to_csv(optimal_csv_path, index=False)
        print(f"Updated optimal FF hyperparams to: {optimal_csv_path}")
        
    soup_dir = f"checkpoints/{args.dataset_name}/soup/{args.model_name}"
    os.makedirs(soup_dir, exist_ok=True)
    existing = {
        int(re.search(r"(\d+)\.pth$", f).group(1))
        for f in os.listdir(soup_dir) if f.endswith(".pth") and re.search(r"\d+\.pth$", f)
    }
    def get_next_index():
        i = 0
        while i in existing:
            i += 1
        existing.add(i)
        return i

    csv_path = os.path.join(soup_dir, f"{args.model_name}_trials.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "idx","trial_id", "learning_rate", "weight_decay",
            "batch_size", "val_metric", "checkpoint_path"
        ])
        if write_header:
            writer.writeheader()

        # Save each trial's best checkpoint
        for trial in ft_result.trials:
            if trial.status != "TERMINATED" or not trial.checkpoint:
                continue
            checkpoint = ft_result.get_best_checkpoint(trial, metric=validation_metric, mode="max")
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    model_path = os.path.join(checkpoint_dir, "model.pth")
                    idx = get_next_index()
                    soup_path = os.path.join(soup_dir, f"trial{idx}.pth")
                    shutil.copy(model_path, soup_path)
                    print(f"Saved trial checkpoint to: {soup_path}")

                    writer.writerow({
                        "model_name": args.model_name,
                        "idx": idx,
                        "trial_id": trial.trial_id,
                        "learning_rate": trial.config["learning_rate"],
                        "weight_decay": trial.config["weight_decay"],
                        "batch_size": trial.last_result.get("batch_size", -1),
                        "val_metric": trial.last_result[validation_metric],
                        "checkpoint_path": soup_path
                    })

if __name__ == "__main__":
    main()