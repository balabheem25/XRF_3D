import os
import numpy as np
import pandas as pd
import csv
from utils import find_pixel_dirs
from concurrent.futures import ThreadPoolExecutor
from file_io import read_hist_file, set_file_format
from pixel_level_Bg_Sub_Curvefit import extract_signals
from hdf5_io import save_signal_data_hdf5
import matplotlib.pyplot as plt
from main_2d import main_2d
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_format = os.path.splitext(spectrum_file_path)[-1].lower()


def process_row(row, intervals, intervals_2, output_root):
    x, y, r, path = int(row["X"]), int(row["Y"]), int(row["Rot_idx"]), row["spectrum_file_path"]
    idx = row.name
    
    try:
        counts, edges = read_hist_file(path, file_format)
        pixel_dir = os.path.dirname(path)

        out_dir = os.path.join(output_root, f"spectrum_{idx}_plots")
        os.makedirs(out_dir, exist_ok = True)
        signal_map = main_2d(pixel_dir, intervals, intervals_2, x, y, r, output_pixel_dir=out_dir)
        plt.figure()
        plt.plot(edges[:-1], counts)
        plt.title(f"Spectrum @ ({x}, {y}, rot{r})")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.savefig(os.path.join(out_dir, "spectrum_plot.png"))
        plt.close()
        
        return x, y, r, signal_map
    except Exception as e:
        print(f"[Error] Failed to process {path}: {e}")
        return None





def main_3d(counts_matrix, intervals, intervals_2, csv_path, voxel_size = 0.1, slice_thickness = 1.0):
    df = pd.read_csv(csv_path)
    n_intervals = len(intervals)
    max_x = df["X"].max() + 1
    max_y = df["Y"].max() + 1
    max_r = df["Rot_idx"].max() + 1
    
    #signal_matrix = np.zeros((n_intervals, max_r, max_x, max_y))
    methods = ["bkg_subtract", "polyfit", "bg_global"]
    signal_matrix = {
        method: [np.zeros((max_r, max_x, max_y)) for _ in range(n_intervals)]
        for method in methods
    }
    output_root = os.path.join(os.path.dirname(csv_path), "output_plots")
    os.makedirs(output_root, exist_ok = True)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row, intervals, intervals_2, output_root) for _, row in df.iterrows()]
        for future in futures:
            result = future.result()
            if result:
                x, y, r, signal_map = result   # dict[method] -> list of 4 interval values
                for method in methods:
                    for i in range(n_intervals):
                        value = signal_map[method][i]
                        signal_matrix[method][i][r, x, y] = value
                        #for i, s in enumerate(signals):
                        #signal_matrix[i, r, x, y] = s
                        #print(f"[DEBUG] signal value at i={i}, r={r}, x={x}, y={y}: {s}")
                        #try:
                        #    signal_matrix[i, r, x, y] = s["Amptek_signal"]
                        #    print(f"[DEBUG] signal value at i={i}, r={r}, x={x}, y={y}: {s}")
                        #except (TypeError, KeyError):
                        #    signal_matrix[i, r, x, y] = s # fallback is s is just a float
                        #    print(f"[DEBUG] signal value at i={i}, r={r}, x={x}, y={y}: {s}")
                        print(f"[DEBUG] method={method}, interval={i}, r={r}, x={x}, y={y}: {value}")
    output_dir = os.path.join(os.path.dirname(csv_path), "signal_matrix")
    os.makedirs(output_dir, exist_ok=True)

    for method in methods:
        for i in range(n_intervals):
            np.save(os.path.join(output_dir, f"{method}_interval{i+1}.npy"), signal_matrix[method][i])
                
    
    #h5_path = os.path.join(os.path.dirname(csv_path), "signal_matrix.h5")
    #with h5py.File(h5_path, "w") as f:
    #for method in signal_matrix:
    #    for i, matrix in enumerate(signal_matrix[method]):
    #        f.create_dataset(f"{method}/interval_{i+1}", data=matrix)

    #save_signal = save_signal_data_hdf5(h5_path, signal_matrix, voxel_size, slice_thickness, intervals)
    #print(f"Signal matrix and metadata saved to: {h5_path}")
    print(f"All signal matrices saved to {output_dir}")
    return signal_matrix


"""
    # Prepare CSV file for writing
    csv_path = os.path.join(main_dir, "spectrum_file_index.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X-index", "Y-index", "Rot-index", "spectrum_file_path"])

        for rot_idx, rot in enumerate(sorted(os.listdir(main_dir))):
            if not rot.startswith("ROT_"):
                continue
            rot_path = os.path.join(main_dir, rot)
            for pixel in sorted(os.listdir(rot_path)):
                if not pixel.startswith("pixel_"):
                    continue
                pixel_path = os.path.join(rot_path, pixel)
                # Parse indices from folder names
                try:
                    x_str, y_str = pixel.replace("pixel_", "").split("x")
                    x_idx = int(x_str)
                    y_idx = int(y_str)
                except Exception:
                    continue
                # Find all spectrum files (.hist or .csv)
                for fname in sorted(os.listdir(pixel_path)):
                    if fname.endswith(".hist") or fname.endswith(".csv"):
                        spectrum_path = os.path.abspath(os.path.join(pixel_path, fname))
                        writer.writerow([x_idx, y_idx, rot_idx, spectrum_path])

    print(f"Spectrum file index written to {csv_path}")
"""



"""
    for rot in rot_dirs:
        rot_path = os.path.join(main_dir, rot)
        pixel_dirs = sorted([os.path.join(rot_path, d) for d in os.listdir(rot_path) if d.startswith("pixel_")])
        rot_signals = []
        for pixel_dir in pixel_dirs:
            signals_matrix = main_2d(pixel_dir, intervals, intervals_2)
            rot_signals.append(signals_matrix)
        all_signals.append(rot_signals)
    # Save the full 3D matrix
    all_signals_array = np.array(all_signals)  # shape: (k, n, features)
    np.save(os.path.join(main_dir, "all_signals.npy"), all_signals_array)
    return all_signals_array
    """
    
def generate_index_csv(main_dir, csv_path):
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X", "Y", "Rot_idx", "spectrum_file_path"])

        for rot_idx, rot in enumerate(sorted(os.listdir(main_dir))):
            if not rot.startswith("ROT_"):
                continue
            rot_path = os.path.join(main_dir, rot)
            for pixel in os.listdir(rot_path):
                if not pixel.startswith("pixel_"):
                    continue
                x_str, y_str = pixel.replace("pixel_", "").split("x")
                x, y = int(x_str), int(y_str)
                pixel_path = os.path.join(rot_path, pixel)
                for fname in os.listdir(pixel_path):
                    if fname.endswith(".hist") or fname.endswith(".csv"):
                        spectrum_path = os.path.abspath(os.path.join(pixel_path, fname))
                        writer.writerow([x, y, rot_idx, spectrum_path])
