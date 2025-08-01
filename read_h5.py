import os
import h5py
import pydicom
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
dicom_dir = "D:\3D_recon\dicom_hdf5_output"  # adjust if needed
h5_file_path = "D:\3D_recon\signal_matrix.h5"   # path to HDF5 file

# === Visualize DICOM slices ===
n_intervals = 3
n_slices = 9

fig, axs = plt.subplots(n_intervals, n_slices, figsize=(15, 6))
fig.suptitle("DICOM Slices", fontsize=16)

for interval in range(n_intervals):
    for slice_idx in range(n_slices):
        filename = f"interval_{interval}_slice_{slice_idx}.dcm"
        filepath = os.path.join(dicom_dir, filename)
        try:
            ds = pydicom.dcmread(filepath)
            image = ds.pixel_array
            axs[interval, slice_idx].imshow(image, cmap='gray')
            axs[interval, slice_idx].axis("off")
            axs[interval, slice_idx].set_title(f"I{interval} S{slice_idx}", fontsize=8)
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            axs[interval, slice_idx].axis("off")
            axs[interval, slice_idx].set_title("Error")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# === Load HDF5 File Separately ===
print("\n--- Contents of all_signals.h5 ---\n")
with h5py.File(h5_file_path, "r") as f:
    print("Root keys:", list(f.keys()))
    signal_data = f["signal_matrix"]
    print("Min:", np.min(signal_data))
    print("Max:", np.max(signal_data))
    print("Mean:", np.mean(signal_data))
    for name in f:
        dataset = f[name]
        print(f"\nDataset: {name}")
        print("Shape:", dataset.shape)
        print("Data type:", dataset.dtype)
        if dataset.ndim > 0:
            print("Sample slice values:\n", dataset[0])
        else:
            print("Value:", dataset[()])

"""
Optional: Visualize slices from HDF5 signal data
with h5py.File(h5_file_path, "r") as f:
    signal_data = f["signal_data"]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        axs[i].imshow(signal_data[i, 0], cmap="hot")  # first rotation of each interval
        axs[i].set_title(f"HDF5 Interval {i}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()
"""
