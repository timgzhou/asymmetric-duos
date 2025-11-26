import os
import argparse
import sys
import torch
import csv
import pandas as pd
from PIL import Image
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models import get_model_with_head
from load_models import TempScaleWrapper
from load_models import DuoWrapper

def get_largest_batch_size(model, transforms, device="cuda", initial_bs=512, min_bs=8):
    dummy_img = Image.new("RGB", (512, 512))
    dummy_tensor = transforms(dummy_img).unsqueeze(0).to(device)
    bs = initial_bs
    model = model.to(device).eval()
    while bs >= min_bs:
        try:
            x = dummy_tensor.repeat(bs, 1, 1, 1)
            with torch.no_grad():
                _ = model(x)
            del x
            torch.cuda.empty_cache()
            print(f"Max batch size for model: {bs}")
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch size {bs} OOMs, Halving..")
                bs //= 2
                torch.cuda.empty_cache()
            else:
                raise e
    raise RuntimeError("No batch size >=8 fits in memory.")


def benchmark_model(model, input_tensor, device, warmup=20, reps=100):
    input_tensor = input_tensor.to(device)
    model = model.to(device).eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
        
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    start = time.time()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start
    
    avg_latency = total_time / reps
    throughput = input_tensor.size(0) / avg_latency
    return avg_latency, throughput


def profile_single_models(single_csv, num_classes, device):
    df = pd.read_csv(single_csv)
    results = []
    total_count = df.shape[0]
    
    for i, row in df.iterrows():
        start_time = time.time()
        model_name = row["model_name"]
        source = row["source"]

        print(f"🔹 Benchmarking single model: {model_name} ({source})")

        for label, wrapper_fn in [
            ("base", lambda m: m)
        ]:
            print(f"🔧 Preparing {label}")
            model, transforms = get_model_with_head(model_name, num_classes, source, freeze=True, m_head=1)
            wrapped_model = wrapper_fn(model)
            transform = transforms["val"]
            try:
                batch_size = get_largest_batch_size(wrapped_model, transform, device=device)
                dummy_img = Image.new("RGB", (512, 512))  # Safe large canvas
                dummy_tensor = transform(dummy_img).unsqueeze(0).to(device)
                dummy_input = dummy_tensor.repeat(batch_size, 1, 1, 1)
                latency, throughput = benchmark_model(wrapped_model, dummy_input, device)
                print(f"{model_name} ({label}): {latency:.4f}s | {throughput:.2f} samples/sec")
                results.append([model_name, source, label, batch_size, latency, throughput])
            except RuntimeError as e:
                print(f"OOM on {model_name} ({label}) — skipping")
                torch.cuda.empty_cache()
            finally:
                del wrapped_model, model
                torch.cuda.empty_cache()
        end_time = time.time()
        print(f"number {i}/{total_count} complete, time spent {end_time-start_time}")

    return results

def profile_duo_models(duo_csv, num_classes, device):  # batch_size no longer needed
    df = pd.read_csv(duo_csv)
    results = []
    total_count = df.shape[0]
    for i, row in df.iterrows():
        start_time = time.time()
        model_large = row["model_large"]
        source_large = row["source_large"]
        model_small = row["model_small"]
        source_small = row["source_small"]

        print(f"Benchmarking Duo: {model_large} + {model_small}")

        m_large, transforms = get_model_with_head(model_large, num_classes, source_large, freeze=True, m_head=1)
        m_small, _ = get_model_with_head(model_small, num_classes, source_small, freeze=True, m_head=1)
        transform = transforms["val"]
        # Wrap with temp scaling and combine
        model = DuoWrapper(TempScaleWrapper(m_large), TempScaleWrapper(m_small)).to(device)

        try:
            # Dynamically determine batch size based on memory
            batch_size = get_largest_batch_size(model, transform, device=device)
            dummy_img = Image.new("RGB", (512, 512))  # Safe large canvas
            dummy_tensor = transform(dummy_img).unsqueeze(0).to(device)
            dummy_input = dummy_tensor.repeat(batch_size, 1, 1, 1)

            latency, throughput = benchmark_model(model, dummy_input, device)
            print(f"Duo: {model_large}+{model_small}: {latency:.4f}s | {throughput:.2f} samples/sec")

            results.append([f"{model_large}+{model_small}", "duo", batch_size, latency, throughput])
        except RuntimeError as e:
            print(f"OOM on Duo: {model_large}+{model_small} — skipping")
            torch.cuda.empty_cache()
        finally:
            del m_large, m_small, model
            torch.cuda.empty_cache()
            end_time = time.time()
            print(f"number {i}/{total_count} complete, time spent {end_time-start_time}")

    return results


def main(single_csv, duo_csv, num_classes,only_single, only_duo):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("result", exist_ok=True)
    csv_path = "evaluation/eval_res/throughput.csv"
    write_header = not os.path.exists(csv_path)

    results = []
    if not only_duo:
        results += profile_single_models(single_csv, num_classes, device)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["model_name", "source_or_wrapper", "wrapper", "batch_size", "latency_sec", "throughput_samples_per_sec"])
            writer.writerows(results)
        
    if not only_single:
        results = profile_duo_models(duo_csv, num_classes, device)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(results)

    print(f"📦 Saved throughput results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_csv", type=str, required=True)
    parser.add_argument("--duo_csv", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=257)
    parser.add_argument("--only_single", action="store_true", help="Only benchmark single models")
    parser.add_argument("--only_duo", action="store_true", help="Only benchmark duo models")
    args = parser.parse_args()
    main(args.single_csv, args.duo_csv, args.num_classes, args.only_single, args.only_duo)