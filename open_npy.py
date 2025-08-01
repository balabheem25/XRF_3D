import numpy as np
from pathlib import Path

# Folder that holds your *.npy files
signal_matrix_dir = Path(r"D:\3D_recon\signal_matrix")

# Load each file into a dict keyed by its base‑filename
signal_matrices = {}

for npy_file in signal_matrix_dir.glob("*.npy"):
    # Key becomes, e.g., "bkg_subtract" for "bkg_subtract.npy"
    key = npy_file.stem
    signal_matrices[key] = np.load(npy_file, allow_pickle=False)
    print(f"{key:15s}  ➜  shape {signal_matrices[key].shape}")

# --- example use -----------------------------------------------------------
# access a specific matrix
bkg_sub = signal_matrices["bkg_subtract_interval1"]      # NumPy array
print("Background‑subtracted matrix dtype:", bkg_sub.dtype)
