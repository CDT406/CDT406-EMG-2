import os
import re
import csv
import matplotlib.pyplot as plt
from run_train import run_training

# ----------- Configuration -----------
base_data_dir = "output/ProcessedData/NormalizedData1000Hz"
base_model_dir = "output/SavedModels/RNN/NormalizedData1000Hz/WindowTest50%overlap"
sequence_length = 3
val_ids = {1, 8}  # ✅ Validate on persons 1 and 8

# Process both 2-state and 4-state models
for state_type in ["2state", "4state"]:
    data_dir = os.path.join(base_data_dir, state_type)
    model_dir = os.path.join(base_model_dir, state_type)
    csv_log_path = os.path.join(model_dir, f"{state_type}_window_test_results.csv")
    plot_path = os.path.join(model_dir, f"{state_type}_accuracy_vs_window_size.png")
    
    print(f"\n=== Processing {state_type} models ===")
    
    os.makedirs(model_dir, exist_ok=True)
    results = []

    # ----------- Training Loop -----------
    for fname in os.listdir(data_dir):
        if fname.endswith(".pkl"):
            data_path = os.path.join(data_dir, fname)
            model_name = fname.replace(".pkl", ".tflite")
            tflite_output_path = os.path.join(model_dir, model_name)

            print(f"\n🔁 Training model for: {fname}")
            try:
                accuracy, n_train, n_val = run_training(
                    data_path=data_path,
                    sequence_length=sequence_length,
                    tflite_output_path=tflite_output_path,
                    val_ids=val_ids,
                    num_classes=4 if state_type == "4state" else 2
                )
            except Exception as e:
                print(f"❌ Skipping {fname} due to error: {e}")
                continue

            match = re.search(r"W(\d+)_O(\d+)", fname)
            window = int(match.group(1)) if match else -1
            overlap = int(match.group(2)) if match else -1

            results.append({
                "window_size": window,
                "overlap": overlap,
                "accuracy": accuracy * 100,
                "train_samples": n_train,
                "val_samples": n_val,
                "model_path": model_name
            })

    # ----------- Save CSV Summary -----------
    with open(csv_log_path, "w", newline="") as csvfile:
        fieldnames = ["window_size", "overlap", "accuracy", "train_samples", "val_samples", "model_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✅ Summary saved to: {csv_log_path}")

    # ----------- Plot Accuracy vs Window Size -----------
    if results:
        results.sort(key=lambda r: r["window_size"])
        window_sizes = [r["window_size"] for r in results]
        accuracies = [r["accuracy"] for r in results]

        plt.figure(figsize=(10, 5))
        plt.plot(window_sizes, accuracies, marker='o')
        plt.title(f"Validation Accuracy vs. Window Size - {state_type}\n(Validation on Persons {', '.join(map(str, val_ids))})")
        plt.xlabel("Window Size")
        plt.ylabel("Validation Accuracy (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"📈 Accuracy plot saved to: {plot_path}")
    else:
        print(f"⚠️ No results to plot for {state_type}")
