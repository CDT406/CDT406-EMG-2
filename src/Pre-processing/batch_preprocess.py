from run_preprocessing import run_preprocessing

# ----------- Constants -----------
input_folder = "datasets/official/unprocessed/slow"
output_dir = r"output/ProcessedData/WindowTest50%overlap"
sampling_rate = 5000
low_cut = 20
high_cut = 450
filter_order = 4
wamp_threshold = 0.02

# ----------- Loop through window sizes 32 to 300 (step 32) -----------
for window_size in range(32, 301, 32):
    overlap = window_size // 2  # 50% overlap
    print(f"\n🔁 Preprocessing with window={window_size}, overlap={overlap}")
    
    run_preprocessing(
        input_folder=input_folder,
        output_dir=output_dir,
        window_size=window_size,
        overlap=overlap,
        sampling_rate=sampling_rate,
        low_cut=low_cut,
        high_cut=high_cut,
        filter_order=filter_order,
        wamp_threshold=wamp_threshold
    )
