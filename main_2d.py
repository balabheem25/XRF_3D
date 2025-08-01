import os
import numpy as np
import matplotlib.pyplot as plt
from pixel_level_Bg_Sub_Curvefit import read_histograms_from_files, extract_signals, visualize_signals_with_hypermet #visualize_stacked_histograms
from file_io import file_format
file_format = os.path.splitext(spectrum_file_path)[-1].lower()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_2d(pixel_dir, intervals, intervals_2, x, y, r, output_pixel_dir = None):
    # Set working directory to pixel_dir
    if output_pixel_dir is None: output_pixel_dir = os.path.join(pixel_dir, "output")
    os.makedirs(output_pixel_dir, exist_ok = True)

    histograms = read_histograms_from_files(pixel_dir, file_format = file_format)
    signals = extract_signals(histograms, intervals)

    #output_dir = os.path.join(output_pixel_dir, "output")
    #os.makedirs(output_dir, exist_ok=True)

    # === Run Hypermet and get raw results with z_scores and fit types ===
    raw_results = visualize_signals_with_hypermet(histograms, intervals, signals, intervals_2, output_dir = output_pixel_dir, x = x, y = y, rot_idx  = r)

    # === Apply z_score threshold and build 12 final signal maps ===
    final_signal_maps = {
        "bkg_subtract": [],
        "polyfit": [],
        "bg_global": []
    }

    for method in ["bkg_subtract", "polyfit", "bg_global"]:
        for interval_index in range(len(intervals)):
            result = raw_results[method][interval_index]
            z = result["z_score"]
            fit = result["fit_type"]
            signal = result["signal"]

            if fit == "Hypermet" or z >= 3:
                final_signal = signal
            else:
                final_signal = 0.0

            final_signal_maps[method].append(final_signal)

    # === Save 12 signal maps: npy files ===
    for method, signal_list in final_signal_maps.items():
        for i, signal in enumerate(signal_list):
            filename = f"signal_map_{method}_interval{i+1}.npy"
            np.save(os.path.join(output_pixel_dir, filename), np.array(signal))

    # Save signals as a matrix (n_intervals x features)
    #signals_matrix = np.array([[s['Amptek_signal'], s['Amptek_avg_bg']] for s in signals])
    #signals_matrix = torch.tensor(signals_matrix) #, device = device)
    #np.save(os.path.join(pixel_dir, "output", "signals_matrix.npy"), signals)

    # Save plots
    #visualize_stacked_histograms(histograms, intervals, intervals_2)
    #plt.savefig(os.path.join(output_dir, "stacked_histograms.png"))
    #plt.close()

    visualize_signals_with_hypermet(histograms, intervals, signals, intervals_2, output_dir = output_pixel_dir, x = x, y = y, rot_idx = r)
    # (Already saves plot inside the function)

    return final_signal_maps #signals_matrix