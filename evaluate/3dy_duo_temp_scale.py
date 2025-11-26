import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uncertainty_metrics import compute_metrics

class DuoFeatureDataset(Dataset):
    def __init__(self, features_l, features_s, logits_l, logits_s, target_list):
        self.features_l = features_l
        self.features_s = features_s
        self.logits_l = logits_l
        self.logits_s = logits_s
        self.targets = target_list
        self.num_samples = len(features_l)
        assert len(features_l)==len(features_s), f"Features extracted from fL and fS have different lengths, {len(features_l)=} and {len(features_s)=}"
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            self.features_l[idx],
            self.features_s[idx],
            self.logits_l[idx],
            self.logits_s[idx],
            self.targets[idx],
        )

def dynamic_duo_forward(features_l, features_s, logits_l, logits_s, lp_l_temp, lp_s_temp, mode="concat"):
    if mode=="concat":
        features_cat=torch.cat((features_l, features_s), 1)
        temp_l = F.softplus(lp_l_temp(features_cat))
        temp_s = F.softplus(lp_s_temp(features_cat))
    else:
        temp_l = F.softplus(lp_l_temp(features_l))
        temp_s = F.softplus(lp_s_temp(features_s))
    scaled_logits_l = logits_l / temp_l
    scaled_logits_s = logits_s / temp_s
    return scaled_logits_l + scaled_logits_s
    
def jointly_calibrate_temperature(features_l, features_s, logits_l, logits_s, labels, num_epochs=32, device='cuda', optimizer="adam", mode="concat"):
    train_dataset = DuoFeatureDataset(features_l,features_s,logits_l, logits_s, labels)
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    in_dim_l, in_dim_s = features_l.shape[1], features_s.shape[1]
    print(f"Setting up dynamic duo temperature (linear) heads, {in_dim_l=} and {in_dim_s=}")
    
    lp_l_temp = nn.Linear(in_dim_l,1).to(device)
    lp_s_temp = nn.Linear(in_dim_s,1).to(device)
    
    if mode=="concat":
        lp_l_temp = nn.Linear(in_dim_l+in_dim_s,1).to(device)
        lp_s_temp = nn.Linear(in_dim_l+in_dim_s,1).to(device)
    
    nn.init.zeros_(lp_l_temp.weight)
    nn.init.constant_(lp_l_temp.bias, 1.0)
    nn.init.zeros_(lp_s_temp.weight)
    nn.init.constant_(lp_s_temp.bias, 1.0)

    loss_fn = nn.CrossEntropyLoss()
    if optimizer=="lbfgs":
        print("Using LBFGS optimizer")
        optimizer = torch.optim.LBFGS(
            list(lp_l_temp.parameters()) + list(lp_s_temp.parameters()),
            lr=0.5, max_iter=20, line_search_fn="strong_wolfe"
        )
        def closure():
            optimizer.zero_grad()
            loss = 0
            for b_features_l, b_features_s, b_logits_l, b_logits_s, b_labels in dataloader:
                b_features_l, b_features_s, b_logits_l, b_logits_s, b_labels = [
                    x.to(device) for x in (b_features_l, b_features_s, b_logits_l, b_logits_s, b_labels)
                ]
                dynamic_duo_logits = dynamic_duo_forward(b_features_l, b_features_s, b_logits_l, b_logits_s, lp_l_temp, lp_s_temp, mode)
                loss += loss_fn(dynamic_duo_logits, b_labels)
            loss.backward()
            return loss
        final_loss = optimizer.step(closure)
        if isinstance(final_loss, torch.Tensor):
            final_loss = final_loss.item()
        print(f"LBFGS optimization finished with final loss = {final_loss:.4f}")
        lp_l_temp.eval()
        lp_s_temp.eval()
        return lp_l_temp, lp_s_temp
    
    elif optimizer=="adam":
        optimizer = torch.optim.Adam(
            list(lp_l_temp.parameters()) + list(lp_s_temp.parameters())
        )
    else:
        raise NotImplementedError()
    
    print("=====Joint temperature calibration in progress...=====")
    best_loss = float('inf')
    best_state = None
    for i_epoch in range(num_epochs):
        lp_l_temp.train()
        lp_s_temp.train()
        running_loss=0.
        num_batches = 0
        for b_features_l, b_features_s, b_logits_l, b_logits_s, b_labels in dataloader:
            b_features_l = b_features_l.to(device)
            b_features_s = b_features_s.to(device)
            b_logits_l = b_logits_l.to(device)
            b_logits_s = b_logits_s.to(device)
            b_labels = b_labels.to(device)
            
            optimizer.zero_grad()
            
            dynamic_duo_logits = dynamic_duo_forward(b_features_l, b_features_s, b_logits_l, b_logits_s,lp_l_temp,lp_s_temp)
            loss = loss_fn(dynamic_duo_logits, b_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        avg_loss = running_loss / num_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                'lp_l_temp': lp_l_temp.state_dict(),
                'lp_s_temp': lp_s_temp.state_dict()
            }
        print(f"{i_epoch+1}/{num_epochs} epochs completed, avg_loss = {avg_loss:.4f}")
        
    if best_state is not None:
        lp_l_temp.load_state_dict(best_state['lp_l_temp'])
        lp_s_temp.load_state_dict(best_state['lp_s_temp'])
        print(f"Best loss: {best_loss:.4f}")
        
    print("\n=====Joint calibration complete=====\n")
    return lp_l_temp, lp_s_temp


def save_dynamic_duo_eval(
    logits, labels, row, dataset_name, split, optimizer, mode, wrapper="dynamic_duo", uncertainty_type="softmax_response"
):
    num_classes = logits.shape[1]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    uncertainties = 1-probs.max(dim=1).values

    metrics = compute_metrics(probs, preds, labels, uncertainties, num_classes)

    # Add metadata columns
    metrics.update({
        "model_large": row["model_large"],
        "model_small": row["model_small"],
        "wrapper": wrapper,
        "uncertainty_type": uncertainty_type,
        "gflops_large": row["gflops_large"],
        "gflops_small": row["gflops_small"],
        "gflops_balance": row["gflops_small"] / row["gflops_large"],
        "split": split,
        "mode": mode
    })

    # Save or append results
    save_dir = f"evaluation/eval_res"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"evaluation_dynamic_duo_{dataset_name}_{optimizer}_{mode}.csv")

    df_new = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Avoid duplicate rows for the same model pair + split
        mask = (
            (df_old["model_large"] == row["model_large"]) &
            (df_old["model_small"] == row["model_small"]) &
            (df_old["split"] == split)
        )
        df_old = df_old[~mask]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(csv_path, index=False)
    print(f"✅ Evaluation metrics saved to {csv_path}")
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--logits_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--duo_csv_path",type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=32)
    parser.add_argument("--optimizer",default="lbfgs")
    parser.add_argument("--mode",default="concat")
    args = parser.parse_args()
    print("\n\n=============CONFIGS===========\n")
    print(args)
    print("\n=============CONFIGS===========\n\n")
    
    duo_df = pd.read_csv(args.duo_csv_path)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index, row in duo_df.iterrows():
        curr_model_small = row['model_small']
        curr_model_large = row['model_large']
        lp_l_temp, lp_s_temp = None, None
        
        for val_set in ["val","test","ood_test"]:
            print(f"{index+1}/{len(duo_df)}", f"{curr_model_large=}", f"{curr_model_small=}")
            large_feat_path = os.path.join(args.features_dir,val_set,"features",f"{curr_model_large}.csv")
            small_feat_path = os.path.join(args.features_dir,val_set,"features",f"{curr_model_small}.csv")
            large_logits_path = os.path.join(args.logits_dir,val_set,"raw",f"{curr_model_large}.csv")
            small_logits_path = os.path.join(args.logits_dir,val_set,"raw",f"{curr_model_small}.csv")
            if not os.path.isfile(large_feat_path) or not os.path.isfile(small_feat_path) or not os.path.isfile(small_logits_path) or not os.path.isfile(large_logits_path):
                if not (args.dataset_name=="caltech256" and val_set=="ood_test"): print(f"skipping for {val_set} since feature file doesn't exist.")
                continue
            feature_large = torch.tensor(pd.read_csv(large_feat_path).values, dtype=torch.float32, device=device)
            feature_small = torch.tensor(pd.read_csv(small_feat_path).values, dtype=torch.float32, device=device)
            logits_large = torch.tensor(pd.read_csv(large_logits_path).values, dtype=torch.float32, device=device)
            logits_small = torch.tensor(pd.read_csv(small_logits_path).values, dtype=torch.float32, device=device)
            
            target_path = os.path.join(f"y-prediction/{args.dataset_name}/{val_set}","point_prediction.csv")
            target = torch.tensor(pd.read_csv(target_path)["target"].values, dtype=torch.long, device=device)
            
            if val_set == "val":
                lp_l_temp, lp_s_temp = jointly_calibrate_temperature(
                    feature_large, feature_small, logits_large, logits_small, 
                    target, num_epochs=args.num_epochs, device=device, optimizer=args.optimizer, mode=args.mode
                )
            elif val_set=="test":
                with torch.no_grad():
                    logits = dynamic_duo_forward(feature_large,feature_small,logits_large,logits_small,lp_l_temp,lp_s_temp,args.mode)
                prediction_save_path = os.path.join(f"y-prediction/{args.dataset_name}/{val_set}/dyn/{args.mode}",f"{curr_model_large}-{curr_model_small}.csv")
                os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)
                pd.DataFrame(logits.cpu().numpy()).to_csv(prediction_save_path,index=False)
                print(f"Dynamic Duo predicted logits saved to {prediction_save_path}")
                save_dynamic_duo_eval(
                    logits=logits,
                    labels=target,
                    row=row,
                    optimizer=args.optimizer,
                    mode=args.mode,
                    dataset_name=args.dataset_name,
                    split=val_set,
                )
        

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
