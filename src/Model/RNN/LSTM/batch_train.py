import os
import re
import csv
import matplotlib.pyplot as plt
from run_train import run_training

# ----------- Configuration -----------
data_dir = "output/ProcessedData/WindowTest50%overlap"
model_dir = "output/SavedModels/RNN/WindowTest50%overlap"
csv_log_path = os.path.join(model_dir, "window_test_results.csv")
plot_path = os.path.join(model_dir, "accuracy_vs_window_size.png")
sequence_length = 3
val_ids = {1, 8}  # ✅ Validate on persons 1 and 2

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
                val_ids=val_ids
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
results.sort(key=lambda r: r["window_size"])
window_sizes = [r["window_size"] for r in results]
accuracies = [r["accuracy"] for r in results]

plt.figure(figsize=(10, 5))
plt.plot(window_sizes, accuracies, marker='o')
plt.title("Validation Accuracy vs. Window Size (Validation on Persons 1 & 2)")
plt.xlabel("Window Size")
plt.ylabel("Validation Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"📈 Accuracy plot saved to: {plot_path}")
