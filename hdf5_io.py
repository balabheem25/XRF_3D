import h5py
import numpy as np

def save_signal_data_hdf5(
    filepath: str,
    signal_data,                      # dict  OR  np.ndarray
    voxel_size: tuple | None = None,  # (x_mm, y_mm, z_mm)
    slice_thickness: float | None = None,
    intervals: list | None = None,    # e.g. [40, 60, 80, 100]
):
    """
    Save a 5‑D signal matrix to HDF5.
        dims = (methods, intervals, rot, y, x)

    `signal_data` may be
        • dict: {"bkg_subtract":[4 arrays], "polyfit":[…], "bg_global":[…]}
        • ndarray with shape (3, 4, rot, y, x)
    """

    # ── 1. ensure we have a NumPy array ───────────────────────────────────
    if isinstance(signal_data, dict):
        method_order   = ["bkg_subtract", "polyfit", "bg_global"]
        n_intervals    = len(signal_data[method_order[0]])
        rot, y, x      = signal_data[method_order[0]][0].shape

        data_array = np.zeros(
            (len(method_order), n_intervals, rot, y, x), dtype=np.float32
        )
        for m_idx, m in enumerate(method_order):
            for i_idx in range(n_intervals):
                data_array[m_idx, i_idx] = signal_data[m][i_idx]

        signal_data = data_array   # overwrite with ndarray

    # ── 2. write to HDF5 ──────────────────────────────────────────────────
    with h5py.File(filepath, "w") as f:
        f.create_dataset("signal_matrix", data=signal_data, compression="gzip")

        if voxel_size is not None:
            f.attrs["voxel_size_mm"] = voxel_size
        if slice_thickness is not None:
            f.attrs["slice_thickness_mm"] = slice_thickness
        if intervals is not None:
            f.create_dataset("intervals", data=np.asarray(intervals))

    print(f"HDF5 saved → {filepath}")
