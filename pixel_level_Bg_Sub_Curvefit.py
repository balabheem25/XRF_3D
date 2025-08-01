import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import make_interp_spline, UnivariateSpline

import os
import sys
import glob
import torch
from file_io import read_histograms_from_files, read_hist_file


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian (x, a, mu, sigma):
    #return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return a * np.exp(-(x-mu)**2 / (2 * sigma**2))

def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) +
            a2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2))

def hypermet(x, A, mu, sigma, B, beta, C, Kp, Kw):
    """
    Hypermet function for Amptek detector, combining Gaussian, plateau, exponential tail, and box-like function.
    :param x: Energy values (keV).
    :param A: Amplitude of the Gaussian.
    :param mu: Mean of the Gaussian.
    :param sigma: Standard deviation of the Gaussian.
    :param B: Scaling factor for the exponential tail.
    :param beta: Parameter scaling the exponential tail.
    :param C: Scaling factor for the plateau.
    :param Kp: Parameter for the box-like function.
    :param Kw: Parameter for the box-like function width.
    :return: Hypermet function values.
    """
    # Gaussian term
    G = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Plateau term
    P = A * B * 0.5 * (1 - erf((x - mu) / (np.sqrt(2) * sigma)))

    # Exponential tail term
    D = A * C * 0.5 * np.exp((x - mu) / beta) * (1 - erf(((x - mu) / (np.sqrt(2) * sigma)) + (sigma / (np.sqrt(2) * beta))))

    # Box-like function term
    H = A * D * (1 + erf((x - mu) *Kp / Kw)) * erf((x - mu) / (np.sqrt(2) * sigma))

    hypermet_c = G + P + D + H

    return hypermet_c


def hypermet_copy(x, A, mu, sigma, B, beta, C, Kp, Kw):
    """
    Hypermet function for Amptek detector, combining Gaussian, plateau, exponential tail, and box-like function.
    :param x: Energy values (keV).
    :param A: Amplitude of the Gaussian.
    :param mu: Mean of the Gaussian.
    :param sigma: Standard deviation of the Gaussian.
    :param B: Scaling factor for the exponential tail.
    :param beta: Parameter scaling the exponential tail.
    :param C: Scaling factor for the plateau.
    :param Kp: Parameter for the box-like function.
    :param Kw: Parameter for the box-like function width.
    :return: Hypermet function values.
    """

    # Gaussian term
    G = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Plateau term
    P = A * B * 0.5 * (1 - erf((x - mu) / (np.sqrt(2) * sigma)))

    # Exponential tail term
    D = A * C * 0.5 * np.exp((x - mu) / beta) * (1 - erf(((x - mu) / (np.sqrt(2) * sigma)) + (sigma / (np.sqrt(2) * beta))))

    # Box-like function term
    H = A * D * (1 + erf((x - mu) *Kp / Kw)) * erf((x - mu) / (np.sqrt(2) * sigma))

    hypermet_c = G + P + D + H

    return G, P, D, H, hypermet_c


def reduced_chi_square(y_obs, y_fit, num_params):
    residual = y_obs - y_fit
    chi2 = np.sum((residual ** 2) / (y_fit + 1e-6))  # avoid division by zero
    dof = len(y_obs) - num_params
    return chi2 / dof if dof > 0 else np.inf

"""
def read_hist_file(file_name):
    #""
    Reads a .hist file and extracts bin edges and counts.
    :param file_name: Path to the .hist file.
    :return: Tuple (counts, edges) as NumPy arrays.
    #""
    with open(file_name, "r") as f:
        lines = f.readlines()

    # Parse metadata
    bin_width = float(lines[0].split()[1])
    emin = float(lines[1].split()[1])
    emax = float(lines[2].split()[1])

    # Parse histogram data
    data = np.array([list(map(float, line.split())) for line in lines[3:]])
    bin_centers = data[:, 0]
    counts = data[:, 1]

    # Calculate bin edges from bin centers and bin width
    edges = np.append(bin_centers -  / 2, bin_centers[-1] + bin_width / 2)

    return counts, edges
        """

"""
def read_histograms_from_files(pixel_dir):
    
    Reads histograms from .hist files and stores them in a dictionary.
    :return: Dictionary of histograms as (counts, edges) tuples.
    
    histograms = {
        "Amptek": read_hist_file("C:/Users/karth/Downloads/Code_to_convert/hist_rebin/Amptek.hist"), #Bragg.root.h1.4.hist
        "C1": read_hist_file(os.path.join(pixel_dir,"C1.hist")), #Bragg.root.h1.11.hist
        #change line 73 as well
        "C2": read_hist_file("C:/Users/karth/Downloads/Code_to_convert/hist_rebin/C2.hist"), #Bragg.root.h1.12.hist"
        "C3": read_hist_file("C:/Users/karth/Downloads/Code_to_convert/hist_rebin/C3.hist"), #Bragg.root.h1.13.hist"
        "Rayl": read_hist_file("C:/Users/karth/Downloads/Code_to_convert/hist_rebin/Rayl.hist"), #Bragg.root.h1.14.hist"
        #"totWOxrf": read_hist_file("C:/Users/karth/Downloads/Code_to_convert/hist_rebin/totWOxrf.hist"), #Bragg.root.h1.15.hist"
        "pFluo": read_hist_file("C:/Users/karth/Downloads/Code_to_convert/hist_rebin/pFluo.hist"), #Bragg.root.h1.10.hist"
    }
    filenames = {
        "Amptek": "Amptek.hist",
        #"C1": "C1.hist",
        #"C2": "C2.hist",
        #"C3": "C3.hist",
        #"Rayl": "Rayl.hist",
        #"pFluo": "pFluo.hist",
    }
    histograms = {}
    for key, fname in filenames.items():
        path = os.path.join(pixel_dir, fname)
        histograms[key] = read_hist_file(path)
    return histograms
"""


def get_interval_counts_1(hist, edges, min_energy, max_energy):
    """
    Calculates the total counts in a specified energy range.
    :param hist: NumPy array of histogram counts.
    :param edges: NumPy array of bin edges.
    :param min_energy: Minimum energy of the range.
    :param max_energy: Maximum energy of the range.
    :return: Total counts in the range.
    """
    bin_min = np.searchsorted(edges, min_energy, side="left")
    bin_max = np.searchsorted(edges, max_energy, side="right") - 1
    total_counts = np.sum(hist[bin_min:bin_max + 1])
    range_width = edges[bin_max + 1] - edges[bin_min]
    return total_counts / range_width


def get_signal_counts(x_vals, y_vals, min_energy, max_energy):
    signal_mask = (x_vals >= min_energy) & (x_vals <= max_energy)
    return np.sum(y_vals[signal_mask])

def extract_signals(histograms, intervals, use_gpu = False):
    """
    Extracts the pFluo and Amptek signals for the given intervals by calculating
    the average background and subtracting it from the signal region.
    :param histograms: Dictionary of histograms as NumPy arrays.
    :param intervals: List of intervals for analysis (central_energy, fwhm).
    :return: List of extracted signals for pFluo and Amptek.
    """
    extracted_signals = []
    #device = "cuda" if (use_gpu and torch.cuda.is_available() else "cpu")
    counts_Amptek, edges_Amptek = histograms["Amptek"]


    for i, (central_energy, fwhm) in enumerate(intervals):
        sigma = fwhm / 2.355
        min_energy = central_energy - 3 * sigma
        max_energy = central_energy + 3 * sigma

        # Calculate background regions
        left_bg_min = min_energy - 3 * sigma
        left_bg_max = min_energy
        right_bg_min = max_energy
        right_bg_max = max_energy + 3 * sigma

        # Calculate average background for Amptek
        left_bg_Amptek = get_interval_counts_1(counts_Amptek, edges_Amptek, left_bg_min, left_bg_max)
        right_bg_Amptek = get_interval_counts_1(counts_Amptek, edges_Amptek, right_bg_min, right_bg_max)
        avg_bg_Amptek = (left_bg_Amptek + right_bg_Amptek) / 2

        gross_counts = get_interval_counts_1(counts_Amptek, edges_Amptek,min_energy, max_energy)
        signal_Amptek = gross_counts - avg_bg_Amptek

        # Z-score calculation
        sigma_bg = np.sqrt(avg_bg_Amptek) if avg_bg_Amptek> 0 else 1e-6
        z_score = (signal_Amptek- avg_bg_Amptek) / sigma_bg

        # Discard signal if not significant
        if z_score < 3:
            signal_Amptek= 0.0
            
        # Store the results
        extracted_signals.append({
            "interval": f"[{min_energy:.1f} - {max_energy:.1f}] keV",
            "Amptek_signal": signal_Amptek,
            "Amptek_avg_bg": avg_bg_Amptek,
            "z_score": z_score,
            "left_bg_min": left_bg_min,
            "left_bg_max": left_bg_max,
            "right_bg_min": right_bg_min,
            "right_bg_max": right_bg_max,
            "signal_min": min_energy,
            "signal_max": max_energy
        })

    return extracted_signals

# subtraction method from c++ file
def bkg_subtract_method_1(x_data, y_data, counts_Amptek, min_energy, max_energy, sigma):
    # Define adjacent background regions (width = max_energy - min_energy)
    dE = max_energy - min_energy
    left_bg_min = min_energy - dE
    left_bg_max = min_energy
    right_bg_min = max_energy
    right_bg_max = max_energy + dE

    # Masks for left and right background
    left_mask = (x_data >= left_bg_min) & (x_data < left_bg_max)
    right_mask = (x_data >= right_bg_min) & (x_data < right_bg_max)

    # Average background from both sides
    left_bg = np.mean(counts_Amptek[left_mask]) if np.any(left_mask) else 0
    right_bg = np.mean(counts_Amptek[right_mask]) if np.any(right_mask) else 0
    avg_bg = (left_bg + right_bg) / 2

    # Estimate background under the signal region
    baseline = np.full_like(y_data, avg_bg)
    signal_counts = np.sum(y_data - baseline)
    background_counts = np.sum(baseline)

    # Chi-square (assuming Poisson errors)
    chi2 = np.sum((y_data - baseline) ** 2 / (baseline + 1e-6))
    # Z-score
    z_score = (signal_counts) / np.sqrt(background_counts) if background_counts > 0 else 0

    return chi2, z_score, avg_bg


def bkg_subtract_method_2(x_data, y_data, counts_Amptek, central_energy, nSigmaB, fwhm, use_left=True, use_right=True,
                        min_energy = None, max_energy = None):
    """
    Background subtraction method matching the C++ analyzeEnergyInterval logic.
    - x_data: bin centers
    - y_data: counts in the signal region (should be from counts_Amptek)
    - counts_Amptek: full histogram counts
    - central_energy: center of the peak
    - nSigmaB: number of sigma for background region width (e.g., 3)
    - fwhm: FWHM of the peak
    - use_left, use_right: whether to use left/right background regions
    """
    sigma = fwhm / 2.355
    if min_energy is None:
        min_energy = central_energy - 3 * sigma
    if max_energy is None:
        max_energy = central_energy + 3 * sigma
    dE = 2 * nSigmaB * sigma  # width of background region

    # Signal region
    signal_mask = (x_data >= min_energy) & (x_data <= max_energy)
    s_plus_b = np.sum(counts_Amptek[signal_mask])
    #num_signal_bins = np.sum(signal_mask)
    signal_width = max_energy - min_energy

    # Left background region
    left_bg_min = min_energy - dE
    left_bg_max = min_energy
    left_mask = (x_data >= left_bg_min) & (x_data < left_bg_max)
    left_bg_counts = np.sum(counts_Amptek[left_mask])
    #left_bg_bins = np.sum(left_mask)
    #left_bg_per_bin = left_bg_counts/ left_bg_bins if left_bg_bins > 0 else 0
    left_bg_width = left_bg_max - left_bg_min

    # Right background region
    right_bg_min = max_energy
    right_bg_max = max_energy + dE
    right_mask = (x_data >= right_bg_min) & (x_data < right_bg_max)
    right_bg_counts = np.sum(counts_Amptek[right_mask])
    #right_bg_bins = np.sum(right_mask)
    #ight_bg_per_bin = right_bg_counts / right_bg_bins if right_bg_bins > 0 else 0
    right_bg_width = right_bg_max - right_bg_min

    b_left = (left_bg_counts / left_bg_width) * signal_width if left_bg_width > 0 else 0
    b_right = (right_bg_counts / right_bg_width) *  signal_width if right_bg_width > 0 else 0

    if use_left and use_right and (left_bg_width > 0 and right_bg_width > 0):
        b = (b_left + b_right) / 2
    elif use_left and left_bg_width > 0:
        b = b_left
    elif use_right and right_bg_width > 0:
        b = b_right
    else:
        b = 0

    s = s_plus_b - b

    # --- Z-score (C++ style) ---
    Z = s / np.sqrt(b) if b > 0 else 0.0
    if Z < 0:
        Z = 0.0

    # --- Chi-square: use expected background per bin in the signal region ---
    num_signal_bins = np.sum(signal_mask)
    avg_bg_per_bin = b / num_signal_bins if num_signal_bins > 0 else 0
    y_signal = counts_Amptek[signal_mask]
    baseline = np.full(num_signal_bins, avg_bg_per_bin)
    chi2 = np.sum((y_signal - baseline) ** 2 / (baseline + 1e-6))

    return chi2, Z, avg_bg_per_bin, s_plus_b, b, s

def bkg_subtract_method(x_data, y_data, counts_Amptek, central_energy, nSigmaB, fwhm, use_left=True, use_right=True,min_energy=None, max_energy=None, edges_Amptek=None):
    sigma = fwhm / 2.355
    if min_energy is None:
        min_energy = central_energy - 3 * sigma
    if max_energy is None:
        max_energy = central_energy + 3 * sigma
    dE = 2 * nSigmaB * sigma  # width of background region
    
    # --- Signal region: use bin edges for exact match ---
    bin_min = np.searchsorted(edges_Amptek, min_energy, side="left")
    bin_max = np.searchsorted(edges_Amptek, max_energy, side="right") - 1
    signal_width = edges_Amptek[bin_max + 1] - edges_Amptek[bin_min]
    s_plus_b = np.sum(counts_Amptek[bin_min:bin_max + 1]) / signal_width if signal_width > 0 else 0
    num_signal_bins = bin_max - bin_min + 1

    # --- Left background region ---
    left_bg_min = min_energy - dE
    left_bg_max = min_energy
    left_bin_min = np.searchsorted(edges_Amptek, left_bg_min, side="left")
    left_bin_max = np.searchsorted(edges_Amptek, left_bg_max, side="right") - 1
    left_bg_width = edges_Amptek[left_bin_max + 1] - edges_Amptek[left_bin_min]
    left_bg_counts = np.sum(counts_Amptek[left_bin_min:left_bin_max + 1]) / left_bg_width if left_bg_width > 0 else 0

    # --- Right background region ---
    right_bg_min = max_energy
    right_bg_max = max_energy + dE
    right_bin_min = np.searchsorted(edges_Amptek, right_bg_min, side="left")
    right_bin_max = np.searchsorted(edges_Amptek, right_bg_max, side="right") - 1
    right_bg_width = edges_Amptek[right_bin_max + 1] - edges_Amptek[right_bin_min]
    right_bg_counts = np.sum(counts_Amptek[right_bin_min:right_bin_max + 1]) / right_bg_width if right_bg_width > 0 else 0

    # --- Scale background counts to signal region width ---
    b_left = left_bg_counts#* signal_width
    b_right = right_bg_counts#* signal_width

    if use_left and use_right and (left_bg_width > 0 and right_bg_width > 0):
        b = (b_left + b_right) / 2
    elif use_left and left_bg_width > 0:
        b = b_left
    elif use_right and right_bg_width > 0:
        b = b_right
    else:
        b = 0

    s = s_plus_b - b

    # --- Z-score (C++ style) ---
    Z = s / np.sqrt(b) if b > 0 else 0.0
    if Z < 0:
        Z = 0.0

    # --- Chi-square: use expected background per bin in the signal region ---
    avg_bg_per_bin = b / num_signal_bins if num_signal_bins > 0 else 0
    y_signal = counts_Amptek[bin_min:bin_max + 1]
    baseline = np.full(num_signal_bins, avg_bg_per_bin)
    chi2 = np.sum((y_signal - baseline) ** 2 / (baseline + 1e-6))

    return chi2, Z, avg_bg_per_bin, s_plus_b, b, s


def estimate_background_methods(x_data, y_data, min_energy, max_energy, left_bg_min, left_bg_max, right_bg_min, right_bg_max, poly_deg=2):
    """
    Returns a dictionary with background arrays for the signal region using different methods.
    """
    # Masks
    left_mask = (x_data >= left_bg_min) & (x_data < left_bg_max)
    right_mask = (x_data > right_bg_min) & (x_data <= right_bg_max)
    signal_mask = (x_data >= min_energy) & (x_data <= max_energy)
    x_signal = x_data[signal_mask]
    y_signal = y_data[signal_mask]

    # 1. Two-sided polyfit + interpolation
    coeff_left = np.polyfit(x_data[left_mask], y_data[left_mask], poly_deg)
    coeff_right = np.polyfit(x_data[right_mask], y_data[right_mask], poly_deg)
    bg_left = np.polyval(coeff_left, x_signal)
    bg_right = np.polyval(coeff_right, x_signal)
    if len(x_signal) > 1:
        weights = (x_signal - x_signal[0]) / (x_signal[-1] - x_signal[0])
    else:
        weights = np.zeros_like(x_signal)
    bg_interp = (1 - weights) * bg_left + weights * bg_right

    # 2. Spline background fit (using all background points)
    x_bg = np.concatenate([x_data[left_mask], x_data[right_mask]])
    y_bg = np.concatenate([y_data[left_mask], y_data[right_mask]])
    if len(x_bg) > poly_deg:
        spline = UnivariateSpline(x_bg, y_bg, k=poly_deg, s=0)
        bg_spline = spline(x_signal)
    else:
        bg_spline = np.full_like(x_signal, np.mean(y_bg) if len(y_bg) > 0 else 0)

    # 3. Global polyfit (using all background points)
    if len(x_bg) > poly_deg:
        coeff_global = np.polyfit(x_bg, y_bg, poly_deg)
        bg_global = np.polyval(coeff_global, x_signal)
    else:
        bg_global = np.full_like(x_signal, np.mean(y_bg) if len(y_bg) > 0 else 0)

    # 4. Flat background (mean of both sides)
    mean_bg = (np.mean(y_data[left_mask]) + np.mean(y_data[right_mask])) / 2 if (np.any(left_mask) and np.any(right_mask)) else 0
    bg_flat = np.full_like(x_signal, mean_bg)

    return {
        "x_signal": x_signal,
        "y_signal": y_signal,
        "bg_interp": bg_interp,
        "bg_spline": bg_spline,
        "bg_global": bg_global,
        "bg_flat": bg_flat,
    }

results_summary = []

def visualize_signals_with_hypermet(histograms, intervals, signals, intervals_2, output_pixel_dir = None, x = None,  y = None, rot_idx = None):
    """
    Visualizes the Amptek histogram values within the interval and the 3-sigma regions on each side,
    along with Hypermet fits for each interval.
    :param histograms: Dictionary of histograms as NumPy arrays.
    :param intervals: List of intervals for analysis (central_energy, fwhm).
    :param signals: List of extracted signals for Amptek.
    """

    counts_Amptek, edges_Amptek = histograms["Amptek"]
    x_data = (edges_Amptek[:-1] + edges_Amptek[1:]) / 2
    fit_results = []
    all_score_results = {
    "bkg_subtract": [],
    "polyfit": [],
    "bg_global": [],
    }
    filename_prefix = f"{x}_{y}_rot{rot_idx}" if None not in (x, y, rot_idx) else "plot"
    # plot 1
    plt.figure(figsize=(12, 8))
    # Plot the full Amptek histogram
    plt.step(edges_Amptek[:-1], counts_Amptek, where="post", label="Amptek (Full Histogram)", color="blue", linewidth=1.5)

    # Create a global exclusion mask for all intervals
    global_intervals_mask = np.zeros_like(x_data, dtype=bool)  # Initialize mask for intervals
    for central_energy, fwhm in intervals:
        sigma = fwhm / 2.355
        min_energy = central_energy - 3 * sigma
        max_energy = central_energy + 3 * sigma
        global_intervals_mask |= (x_data >= min_energy) & (x_data <= max_energy)  # Combine masks for all intervals

    # Create a global exclusion mask for intervals_2
    global_intervals_2_mask = np.zeros_like(x_data, dtype=bool)  # Initialize mask for intervals_2
    for central_energy_2, fwhm_2 in intervals_2:
        intervals_2_sigma = fwhm_2 / 2.355
        intervals_2_min = central_energy_2 - 3 * intervals_2_sigma
        intervals_2_max = central_energy_2 + 3 * intervals_2_sigma
        global_intervals_2_mask |= (x_data >= intervals_2_min) & (x_data <= intervals_2_max)  # Combine masks for intervals_2

    # Combine both masks
    global_exclusion_mask = global_intervals_mask | global_intervals_2_mask

        # Define extended background regions for all intervals
    global_background_mask = np.zeros_like(x_data, dtype=bool)  # Initialize mask for background

    for central_energy, fwhm in intervals:
        sigma = fwhm / 2.355
        min_energy = central_energy - 3 * sigma
        max_energy = central_energy + 3 * sigma
        left_bg_min = min_energy - 9 * sigma
        right_bg_max = max_energy + 9 * sigma

        # Add background regions excluding Gaussian regions
        global_background_mask |= ((x_data >= left_bg_min) & (x_data < min_energy)) | ((x_data > max_energy) & (x_data <= right_bg_max))

    # Exclude Gaussian regions from intervals and intervals_2
    global_background_mask &= ~global_exclusion_mask  # Exclude Gaussian regions from all intervals
    # Overlay Hypermet fits and extracted values for each interval
    for i, (central_energy, fwhm) in enumerate(intervals):
        #plt.figure(figsize = (12,8))

        chosen_fit = "None"
        chosen_chi2 = np.inf

        sigma = fwhm / 2.355

        if i ==1:
            min_energy = central_energy - 4.5 * sigma #for Gd interval, using 5 sigma to get full signal 
        else:
            min_energy = central_energy - 3 * sigma
        max_energy = central_energy + 3 * sigma

        # Extract histogram data for the interval and 3-sigma regions
        bin_min = np.searchsorted(edges_Amptek, min_energy, side="left")
        bin_max = np.searchsorted(edges_Amptek, max_energy, side="right")
        ##x_data = (edges_Amptek[bin_min:bin_max] + edges_Amptek[bin_min + 1:bin_max + 1]) / 2  # Bin centers
        x_interval = x_data[bin_min:bin_max]
        y_data = counts_Amptek[bin_min:bin_max]

        #calculate background regions
        left_bg_min = min_energy - 9 * sigma
        left_bg_max = min_energy
        right_bg_min = max_energy
        right_bg_max = max_energy + 9 * sigma
        """
        for i, (central_energy, fwhm) in enumerate(intervals):
            # Calculate sigma and energy ranges
            sigma = fwhm / 2.355
            min_energy = central_energy - 3 * sigma
            max_energy = central_energy + 3 * sigma

            # Define extended background regions for the current interval
            left_bg_min = min_energy - 9 * sigma
            right_bg_max = max_energy + 9 * sigma

            # Create exclusion mask for the current interval and all other intervals
            global_exclusion_mask = np.zeros_like(x_data, dtype=bool)
            for other_energy, other_fwhm in intervals:
                other_sigma = other_fwhm / 2.355
                other_min = other_energy - 3 * other_sigma
                other_max = other_energy + 3 * other_sigma
                global_exclusion_mask |= (x_data >= other_min) & (x_data <= other_max)

            for central_energy_2, fwhm_2 in intervals_2:
                intervals_2_sigma = fwhm_2 / 2.355
                intervals_2_min = central_energy_2 - 3 * intervals_2_sigma
                intervals_2_max = central_energy_2 + 3 * intervals_2_sigma
                global_exclusion_mask |= (x_data >= intervals_2_min) & (x_data <= intervals_2_max)

            # Create the background mask for the current interval
            background_mask = ((x_data >= left_bg_min) & (x_data < min_energy)) | ((x_data > max_energy) & (x_data <= right_bg_max))
            background_mask &= ~global_exclusion_mask  # Exclude Gaussian regions from all intervals

            # Extract background data
            background_x = x_data[background_mask]
            background_y = counts_Amptek[background_mask]

            # Fit a polynomial curve to the background data
            degree = 2  # Degree of the polynomial (1 for linear, 2 for quadratic, etc.)
            coefficients = np.polyfit(background_x, background_y, degree)

            # Generate fitted curve points
            fitted_x = np.linspace(left_bg_min, right_bg_max, 500)  # Generate x values for the curve
            fitted_y = np.polyval(coefficients, fitted_x)  # Evaluate the polynomial at fitted_x

            # Plot the fitted curve
            plt.plot(fitted_x, fitted_y, label=f"Baseline (Interval {i+1}, Polynomial Degree {degree})", color="yellow", linestyle="-")

            # Highlight the Gaussian region for the current interval
            plt.axvspan(min_energy, max_energy, color="green", alpha=0.1, label="Signal Region" if i == 0 else None)
        """
        # Create exclusion mask for the current interval and all other intervals
        global_exclusion_mask = np.zeros_like(x_data, dtype=bool)
        for other_energy, other_fwhm in intervals:
            other_sigma = other_fwhm / 2.355
            other_min = other_energy - 3 * other_sigma
            other_max = other_energy + 3 * other_sigma
            global_exclusion_mask |= (x_data >= other_min) & (x_data <= other_max)

        for central_energy_2, fwhm_2 in intervals_2:
            intervals_2_sigma = fwhm_2 / 2.355
            intervals_2_min = central_energy_2 - 3 * intervals_2_sigma
            intervals_2_max = central_energy_2 + 3 * intervals_2_sigma
            global_exclusion_mask |= (x_data >= intervals_2_min) & (x_data <= intervals_2_max)

        # Create the background mask for the current interval
        background_mask = ((x_data >= left_bg_min) & (x_data < min_energy)) | ((x_data > max_energy) & (x_data <= right_bg_max))
        background_mask &= ~global_exclusion_mask  # Exclude Gaussian regions from all intervals

        # Extract background data
        background_x = x_data[background_mask]
        background_y = counts_Amptek[background_mask]

        # Fit a polynomial curve to the background data
        degree = 2  # Degree of the polynomial (1 for linear, 2 for quadratic, etc.)
        coefficients = np.polyfit(background_x, background_y, degree)

        # Generate fitted curve points
        fitted_x = np.linspace(left_bg_min, right_bg_max, 500)  # Generate x values for the curve
        fitted_y = np.polyval(coefficients, fitted_x)  # Evaluate the polynomial at fitted_x

        # Plot the fitted curve
        plt.plot(fitted_x, fitted_y, label=f"Baseline (Interval {i+1}, Polynomial Degree {degree})", color="yellow", linestyle="-")
        # Highlight the Gaussian region for the current interval
        plt.axvspan(min_energy, max_energy, color="green", alpha=0.1, label="Signal Region" if i == 0 else None)
        ##############################################################################################################
        left_bg_mask = (x_data >= left_bg_min) & (x_data < left_bg_max)
        right_bg_mask = (x_data > right_bg_min) & (x_data <= right_bg_max)
        left_avg_bg = np.mean(counts_Amptek[left_bg_mask]) if np.any(left_bg_mask) else 0
        right_avg_bg = np.mean(counts_Amptek[right_bg_mask]) if np.any(right_bg_mask) else 0

        baseline_avg = np.interp(x_interval, [x_interval[0], x_interval[-1]], [left_avg_bg, right_avg_bg])
        ################################################################################################################

        # Fit Hypermet function
        try:
            p0_hypermet = [np.max(y_data), central_energy, sigma, 0.1, 1.0, 0.1, 1.0, 1.0]
            print(f"x_interval for interval {i+1}: {x_interval}")
            print(f"y_data for interval {i+1}: {y_data}")
            popt_hypermet, _ = curve_fit(hypermet, x_interval, y_data, p0=p0_hypermet, maxfev = 10000)
            fitted_hypermet = hypermet(x_interval, *popt_hypermet)
            chosen_fit = "Hypermet"
            chosen_chi2 = reduced_chi_square(y_data, fitted_hypermet, len(p0_hypermet))
            plt.plot(x_interval, fitted_hypermet, label=f"Hypermet Fit (Interval {i+1})", color="green", linestyle="-")
        except RuntimeError:
            print(f"Hypermet fit failed for interval {i+1}")
            #Fallback to gaussian fit
            try:
                p0_gaussian = [np.max(y_data), central_energy, sigma]
                popt_gaussian, _ = curve_fit(gaussian, x_interval, y_data, p0 = p0_gaussian, maxfev = 10000)
                fitted_gaussian = gaussian(x_interval, *popt_gaussian)
                chi2_gaussian = reduced_chi_square(y_data, fitted_gaussian, 3)

                #Double Gaussian fit
                p0_double_gaussian = [np.max(y_data), central_energy, sigma, #first Gauss
                                      np.max(y_data) / 2, central_energy + sigma, sigma #second Gauss
                                      ]
                popt_double_gaussian, _ = curve_fit(double_gaussian, x_interval, y_data, p0 = p0_double_gaussian, maxfev = 10000)
                fitted_double_gaussian = double_gaussian(x_interval, *popt_double_gaussian)
                chi2_double_gaussian = reduced_chi_square(y_data, fitted_double_gaussian, 6)

                if chi2_gaussian < chi2_double_gaussian:
                    y_fit = fitted_gaussian #y_single_fit
                    fit_label = f"Interval {i+1}: SIngle Gaussian (χ²ᵣ, = {chi2_gaussian:.2f})"
                    fit_color = "orange"
                    chosen_fit = "Single Gaussian"
                    chosen_chi2 = chi2_gaussian
                    plt.plot(x_interval, fitted_gaussian, label = f"SingleGaussianFit(interval {i+1})", color = "orange", linestyle = "-")
                else:
                    y_fit = fitted_double_gaussian if fitted_double_gaussian is not None else fitted_gaussian
                    fit_label = f"Interval  {i+1}: Double_Gaussian (χ²ᵣ = {chi2_double_gaussian:.2f})"
                    fit_color = "purple" if fitted_double_gaussian is not None else "orange"
                    chosen_fit = "Double Gaussian"
                    chosen_chi2 = chi2_double_gaussian
                    print(f"x_interval for interval {i+1}: {x_interval}")
                    print(f"y_data for interval {i+1}: {y_data}")
                    plt.plot(x_interval, fitted_double_gaussian,  label = f"DoubleGaussFit(interval {i+1})", color = "orange",  linestyle = "-")
            except RuntimeError:
                print(f" Both the Gaaussian fits also failed for the interval {i+1}. skipping this interval.")
                y_fit_bg_subtracted = np.zeros_like(x_interval)
                continue
        baseline_avg = np.interp(x_interval, [x_interval[0], x_interval[-1]], [left_avg_bg, right_avg_bg])
        y_fit = np.zeros_like(x_interval)
        y_fit_bg_subtracted = y_fit - baseline_avg
        signal_counts = np.sum(y_fit_bg_subtracted)
        background_counts = np.sum(baseline_avg) * len(x_interval)
        if background_counts > 0:
            Z_score = (signal_counts -background_counts) / np.sqrt(background_counts)
            Z_score = max(0, Z_score)
        else:
            Z_score = 0
        fit_results.append ({
            "interval": i + 1,
            "central_energy": central_energy,
            "fit_type": chosen_fit,
            "chi+square": chosen_chi2,
            "Z_score": Z_score,
        })
    # Define extended background regions
    left_bg_min = central_energy - 9 * sigma   
    right_bg_max = central_energy + 9 * sigma

    # Create exclusion masks for intervals and intervals_2
    intervals_mask = (x_data >= min_energy) & (x_data <= max_energy)  # Exclude Gaussian regions from intervals
    # for intervals_2, a list of tuples, e.g., [(48.70, 1.1)]
    for central_energy_2, fwhm_2 in intervals_2:
        intervals_2_sigma = fwhm_2 / 2.355
        intervals_2_min = central_energy_2 - 3 * intervals_2_sigma
        intervals_2_max = central_energy_2 + 3 * intervals_2_sigma

        # Create exclusion mask for intervals_2
        intervals_2_mask = (x_data >= intervals_2_min) & (x_data <= intervals_2_max)
        
    # Combine exclusion masks
    exclusion_mask = intervals_mask | intervals_2_mask

    # Create the background mask, excluding Gaussian regions
    background_mask = ((x_data >= left_bg_min) & (x_data < min_energy)) | ((x_data > max_energy) & (x_data <= right_bg_max))
    background_mask = background_mask & ~exclusion_mask  # Exclude Gaussian regions from intervals and intervals_2

    # Extract background data
    background_x = x_data[background_mask]
    background_y = counts_Amptek[background_mask]

    # Fit a polynomial curve to the background data
    degree = 2  # Degree of the polynomial (1 for linear, 2 for quadratic, etc.)
    coefficients = np.polyfit(background_x, background_y, degree)

    # Generate fitted curve points
    fitted_x = np.linspace(left_bg_min, right_bg_max, 500)  # Generate x values for the curve
    fitted_y = np.polyval(coefficients, fitted_x)  # Evaluate the polynomial at fitted_x

    # Plot the fitted curve
    plt.plot(fitted_x, fitted_y, label=f"Baseline (Polynomial Degree {degree})", color="yellow", linestyle="-")

    # Highlight the Gaussian region for intervals
    plt.axvspan(min_energy, max_energy, color="green", alpha=0.3, label="Signal Region" if i == 0 else None)

    # Highlight the Gaussian region for intervals_2
    plt.axvspan(intervals_2_min, intervals_2_max, color="gray", alpha=0.3, label="Intervals_2 Region" if i == 0 else None)


        # Overlay Hypermet fits and extracted values for each interval
    for i, (central_energy, fwhm) in enumerate(intervals_2):
        #plt.figure(figsize = (12,8))

        chosen_fit = "None"
        chosen_chi2 = np.inf

        sigma = fwhm / 2.355
        min_energy = central_energy - 3 * sigma
        max_energy = central_energy + 3 * sigma

        # Extract histogram data for the interval and 3-sigma regions
        bin_min = np.searchsorted(edges_Amptek, min_energy, side="left")
        bin_max = np.searchsorted(edges_Amptek, max_energy, side="right")
        ##x_data = (edges_Amptek[bin_min:bin_max] + edges_Amptek[bin_min + 1:bin_max + 1]) / 2  # Bin centers
        x_interval = x_data[bin_min:bin_max]
        y_data = counts_Amptek[bin_min:bin_max]

        #calculate background regions
        left_bg_min = min_energy - 9 * sigma
        left_bg_max = min_energy
        right_bg_min = max_energy
        right_bg_max = max_energy + 9 * sigma

        #calculate left and right background average
        left_bg_mask = (x_data >= left_bg_min) & (x_data < left_bg_max)
        right_bg_mask = (x_data > right_bg_min) & (x_data <= right_bg_max)
        left_avg_bg = np.mean(counts_Amptek[left_bg_mask]) if np.any(left_bg_mask) else 0
        right_avg_bg = np.mean(counts_Amptek[right_bg_mask]) if np.any(right_bg_mask) else 0

        # Highlight the interval and 3-sigma regions
        #plt.axvspan(min_energy, max_energy, color="green", alpha=0.3, label="Signal Region" if i == 0 else None)
        #plt.axvspan(min_energy - 3 * sigma, min_energy, color="red", alpha=0.3, label="Left 3σ Region" if i == 0 else None)
        #plt.axvspan(max_energy, max_energy + 3 * sigma, color="orange", alpha=0.3, label="Right 3σ Region" if i == 0 else None)

        # Fit Hypermet function
        try:
            p0_hypermet = [np.max(y_data), central_energy, sigma, 0.1, 1.0, 0.1, 1.0, 1.0]
            popt_hypermet, _ = curve_fit(hypermet, x_interval, y_data, p0=p0_hypermet, maxfev = 10000)
            fitted_hypermet = hypermet(x_interval, *popt_hypermet)
            chosen_fit = "Hypermet"
            chosen_chi2 = reduced_chi_square(y_data, fitted_hypermet, len(p0_hypermet))
            ###################################
            interval_mask = (x_interval >= min_energy) & (x_interval <= max_energy)
            x_interval_masked = x_interval[interval_mask]

            # Get individual components of the Hypermet curve
            G, P, D, H, hypermet_c = hypermet_copy(x_interval_masked, *popt_hypermet)

            # Plot individual components
            plt.plot(x_interval_masked, G, label="Gaussian Component", color="blue", linestyle="--")
            plt.plot(x_interval_masked, P, label="Plateau Component", color="orange", linestyle="--")
            plt.plot(x_interval_masked, D, label="Exponential Tail Component", color="green", linestyle="--")
            plt.plot(x_interval_masked, H, label="Box-like Function Component", color="purple", linestyle="--")


            plt.plot(x_interval_masked, hypermet_c, "--", color = "green", label = f"Hypermet Fit (Interval {i+1})" )
            # Calculate Z-score
            baseline_avg = np.interp(x_interval, [x_interval[0], x_interval[-1]], [np.mean(background_y), np.mean(background_y)])
            y_fit_bg_subtracted = fitted_hypermet - baseline_avg
            signal_counts = np.sum(y_fit_bg_subtracted)
            background_counts = np.sum(baseline_avg) * len(x_interval)
            if background_counts > 0:
                Z_score = (signal_counts) / np.sqrt(background_counts)
                Z_score = max(0, Z_score)
            #####################################
        
            #plt.plot(x_interval, fitted_hypermet, label=f"Hypermet Fit (Interval {i+1})", color="black", linestyle="-")
        except RuntimeError:
            print(f"Hypermet fit failed for interval {i+1}")
            #Fallback to gaussian fit
            try:
                p0_gaussian = [np.max(y_data), central_energy, sigma]
                popt_gaussian, _ = curve_fit(gaussian, x_interval, y_data, p0 = p0_gaussian)
                fitted_gaussian = gaussian(x_interval, *popt_gaussian)
                chi2_gaussian = reduced_chi_square(y_data, fitted_gaussian, 3)

                #Double Gaussian fit
                p0_double_gaussian = [np.max(y_data), central_energy, sigma, #first Gauss
                                      np.max(y_data) / 2, central_energy + sigma, sigma #second Gauss
                                      ]
                popt_double_gaussian, _ = curve_fit(double_gaussian, x_interval, y_data, p0 = p0_double_gaussian)
                fitted_double_gaussian = double_gaussian(x_interval, *popt_double_gaussian)
                chi2_double_gaussian = reduced_chi_square(y_data, fitted_double_gaussian, 6)

                if chi2_gaussian < chi2_double_gaussian:
                    y_fit = fitted_gaussian #y_single_fit
                    fit_label = f"Interval {i+1}: SIngle Gaussian (χ²ᵣ, = {chi2_gaussian:.2f})"
                    fit_color = "orange"
                    chosen_fit = "Single Gaussian"
                    chosen_chi2 = chi2_gaussian
                    plt.plot(x_interval, fitted_gaussian, label = f"SingleGaussianFit(interval {i+1})", color = "red", linestyle = "-")
                else:
                    y_fit = fitted_double_gaussian if fitted_double_gaussian is not None else fitted_gaussian
                    fit_label = f"Interval  {i+1}: Double_Gaussian (χ²ᵣ = {chi2_double_gaussian:.2f})"
                    fit_color = "red" if fitted_double_gaussian is not None else "orange"
                    chosen_fit = "Double Gaussian"
                    chosen_chi2 = chi2_double_gaussian
                    plt.plot(x_interval, fitted_double_gaussian,  label = f"DoubleGaussFit(interval {i+1})", color = "red",  linestyle = "-")


            except RuntimeError:
                print(f" Both the Gaaussian fits also failed for the interval {i+1}. skipping this interval.")
                y_fit_bg_subtracted = np.zeros_like(x_interval)
                continue
        baseline_avg = np.interp(x_interval, [x_interval[0], x_interval[-1]], [left_avg_bg, right_avg_bg])
        y_fit = np.zeros_like(x_interval)
        y_fit_bg_subtracted = y_fit - baseline_avg
        signal_counts = np.sum(y_fit_bg_subtracted)
        background_counts = np.sum(baseline_avg) * len(x_interval)
        if background_counts > 0:
            Z_score = (signal_counts -background_counts) / np.sqrt(background_counts)
            Z_score = max(0, Z_score)
        else:
            Z_score = 0
        fit_results.append ({
            "interval": i + 1,
            "central_energy": central_energy,
            "fit_type": chosen_fit,
            "chi+square": chosen_chi2,
            "Z_score": Z_score,
        })
    
    # Define extended background regions
    left_bg_min = central_energy - 9 * sigma   
    right_bg_max = central_energy + 9 * sigma

    # Create exclusion masks for intervals and intervals_2
    intervals_mask = (x_data >= min_energy) & (x_data <= max_energy)  # Exclude Gaussian regions from intervals
    # for intervals_2, a list of tuples, e.g., [(48.70, 1.1)]
    for central_energy_2, fwhm_2 in intervals_2:
        intervals_2_sigma = fwhm_2 / 2.355
        intervals_2_min = central_energy_2 - 3 * intervals_2_sigma
        intervals_2_max = central_energy_2 + 3 * intervals_2_sigma

        # Create exclusion mask for intervals_2
        intervals_2_mask = (x_data >= intervals_2_min) & (x_data <= intervals_2_max)

    # Combine exclusion masks
    exclusion_mask = intervals_mask | intervals_2_mask

    # Create the background mask, excluding Gaussian regions
    background_mask = ((x_data >= left_bg_min) & (x_data < min_energy)) | ((x_data > max_energy) & (x_data <= right_bg_max))
    background_mask = background_mask & ~exclusion_mask  # Exclude Gaussian regions from intervals and intervals_2

    # Extract background data
    background_x = x_data[background_mask]
    background_y = counts_Amptek[background_mask]

    # Fit a polynomial curve to the background data
    degree = 2  # Degree of the polynomial (1 for linear, 2 for quadratic, etc.)
    coefficients = np.polyfit(background_x, background_y, degree)

    # Generate fitted curve points
    fitted_x = np.linspace(left_bg_min, right_bg_max, 1000)  # Generate x values for the curve
    fitted_y = np.polyval(coefficients, fitted_x)  # Evaluate the polynomial at fitted_x

    # Plot the fitted curve
    #plt.plot(fitted_x, fitted_y, label=f"Baseline (Polynomial Degree {degree})", color="yellow", linestyle="-")

    # Highlight the Gaussian region for intervals
    plt.axvspan(min_energy, max_energy, color="green", alpha=0.3, label="Signal Region" if i == 0 else None)

    # Highlight the Gaussian region for intervals_2
    plt.axvspan(intervals_2_min, intervals_2_max, color="gray", alpha=0.3, label="Intervals_2 Region" if i == 0 else None)


        
    # Finalize the plot
    plt.yscale("log")
    plt.ylim(1, 1e4)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.title("Amptek Histogram with Extracted Signals and Hypermet Fits")
    #plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    if output_pixel_dir:
        os.makedirs(output_pixel_dir, exist_ok=True)
        method_plot_path = os.path.join(output_pixel_dir, f"{filename_prefix}_full.png")
        plt.savefig(method_plot_path)
    plt.close()
    #plt.show()

    fit_results = []

    #plot 2-5 seperate plots for  each interval
    for i, (central_energy, fwhm) in enumerate(intervals):
        plt.figure(figsize =(10,6))
        sigma = fwhm / 2.355

        if i  == 1:
            min_energy = central_energy - 4.5 * sigma
        else:
            min_energy = central_energy - 3 * sigma

        max_energy = central_energy + 3 * sigma

        #Extract histo data for interval
        bin_min = np.searchsorted(edges_Amptek, min_energy, side = "left")
        bin_max = np.searchsorted(edges_Amptek, max_energy, side = "right")
        x_interval = x_data[bin_min:bin_max]
        y_data = counts_Amptek[bin_min:bin_max]
        bin_widths = edges_Amptek[bin_min + 1:bin_max +1] - edges_Amptek[bin_min:bin_max]

        #calculate background regions
        dE = max_energy - min_energy
        left_bg_min = min_energy - 9 * sigma
        left_bg_max = min_energy
        right_bg_min =max_energy
        right_bg_max = max_energy +  9 * sigma

        # multiple methods or background estimation 
        bg_methods = estimate_background_methods(
            x_data, counts_Amptek,
            min_energy, max_energy,
            left_bg_min, left_bg_max,
            right_bg_min, right_bg_max,
            poly_deg=2
        )

        for method in ["bg_interp", "bg_spline", "bg_global", "bg_flat"]:
            signal_minus_bg = bg_methods["y_signal"] - bg_methods[method]
            print(f"Method: {method}, Signal sum after BG subtraction: {np.sum(signal_minus_bg):.2f}")

        # Use bg_global as background for the signal region
        bg_global = bg_methods["bg_global"]
        y_signal = bg_methods["y_signal"]  # This is the data in the signal region (same x as bg_global)

        # Subtract background
        signal_minus_bg_global = y_signal - bg_global

        # Calculate chi-square (using background as expected value)
        chi2_bg_global = np.sum((y_signal - bg_global) ** 2 / (bg_global + 1e-6)) / (len(y_signal) if len(y_signal) > 0 else 1)

        # Calculate Z-score (signal sum divided by sqrt of background sum)
        signal_sum = np.sum(signal_minus_bg_global)
        background_sum = np.sum(bg_global)
        z_score_bg_global = signal_sum / np.sqrt(background_sum) if background_sum > 0 else 0

        print(f"[bg_global] Signal sum: {signal_sum:.2f}, Background sum: {background_sum:.2f}, Chi2: {chi2_bg_global:.2f}, Z: {z_score_bg_global:.2f}")
        
        # Create exclusion mask for the current interval and all other intervals
        exclusion_mask = np.zeros_like(x_data, dtype=bool)
        for other_energy, other_fwhm in intervals:
            other_sigma = other_fwhm / 2.355
            other_min = other_energy - 3 * other_sigma
            other_max = other_energy + 3 * other_sigma
            exclusion_mask |= (x_data >= other_min) & (x_data <= other_max)

        for central_energy_2, fwhm_2 in intervals_2:
            intervals_2_sigma = fwhm_2 / 2.355
            intervals_2_min = central_energy_2 - 3 * intervals_2_sigma
            intervals_2_max = central_energy_2 + 3 * intervals_2_sigma
            exclusion_mask |= (x_data >= intervals_2_min) & (x_data <= intervals_2_max)

        # Create the background mask for the current interval
        background_mask = ((x_data >= left_bg_min) & (x_data < left_bg_max)) | ((x_data > right_bg_min) & (x_data <= right_bg_max))
        background_mask &= ~exclusion_mask  # Exclude Gaussian regions from all intervals

        # Extract background data
        background_x = x_data[background_mask]
        background_y = counts_Amptek[background_mask]

        # Fit a polynomial curve to the background data
        degree = 2  # Degree of the polynomial (1 for linear, 2 for quadratic, etc.)
        coefficients = np.polyfit(background_x, background_y, degree)

        """  # Generate fitted curve points
        fitted_x = np.linspace(left_bg_min, right_bg_max, 500)  # Generate x values for the curve
        fitted_y = np.polyval(coefficients, fitted_x)  # Evaluate the polynomial at fitted_x 

        #calculate baseline dynamically
        baseline_avg = np.interp(x_interval, fitted_x, fitted_y)
        """
        
        # Plot polyfit on background region only
        fitted_bg_y = np.polyval(coefficients, background_x)
        plt.plot(background_x, fitted_bg_y, color="yellow", linestyle="-", label="Polyfit (background only)")

        # Integrate polyfit under the signal region
        bin_min = np.searchsorted(x_data, min_energy, side="left")
        bin_max = np.searchsorted(x_data, max_energy, side="right")
        x_interval = x_data[bin_min:bin_max]
        y_data = counts_Amptek[bin_min:bin_max]
        bin_widths = edges_Amptek[bin_min + 1:bin_max + 1] - edges_Amptek[bin_min:bin_max]
        poly_bg_signal = np.polyval(coefficients, x_interval)
        avg_bkg = np.sum(poly_bg_signal * bin_widths)


        # Plot the Amptek histogram for the interval
        plt.step(x_interval, y_data, where="post", label="Amptek (Interval Histogram)", color="red", linewidth=1)
        
        # (Optional) Plot polyfit over signal region
        plt.plot(x_interval, poly_bg_signal, color="yellow", linestyle="--", label="Polyfit (signal region)")

        plt.step(bg_methods["x_signal"], bg_methods["y_signal"], where="post", label="Signal Region Data", color="purple")
        #plt.plot(bg_methods["x_signal"], bg_methods["bg_interp"], label="Two-sided Polyfit+Interp", color="black")
        #plt.plot(bg_methods["x_signal"], bg_methods["bg_spline"], label="Spline BG Fit", color="green")
        plt.plot(bg_methods["x_signal"], bg_methods["bg_global"], label="Global Polyfit", color="red")
        #plt.plot(bg_methods["x_signal"], bg_methods["bg_flat"], label="Flat BG (Mean)", color="red")


        # Plot the fitted background curve
        ######## plt.plot(fitted_x, fitted_y, label=f"Baseline (Interval {i+1}, Polynomial Degree {degree})", color="yellow", linestyle="-")

        # Highlight the Gaussian region for the current interval
        plt.axvspan(min_energy, max_energy, color="green", alpha=0.3, label="Signal Region")


            # plot amptek  histo for the interval
            #plt.step(x_interval, y_data, where = "post", label = "Amptek (Interval histo)", color  = "red", linewidth = 1)

        chosen_fit = "None"
        chosen_chi2 = np.inf
        Z_score = 0
        signal_counts = 0
        background_counts = 0
        s_plus_b_curve_fit = 0
        # Check if there are enough data points
        if len(x_interval) < len(p0_hypermet):
            print(f"Interval {i+1} has insufficient data points for Hypermet fit. Skipping.")
            continue
        #fit hypermet func
        try:
            p0_hypermet = [np.max(y_data),  central_energy, sigma, 0.1, 1.0, 0.1, 1.0, 1.0]
            popt_hypermet, _  = curve_fit(hypermet, x_interval, y_data, p0 = p0_hypermet, maxfev = 5000)
            fitted_hypermet = hypermet(x_interval,  *popt_hypermet)
            chosen_fit = "Hypermet"
            chosen_chi2 = reduced_chi_square(y_data, fitted_hypermet, len(p0_hypermet))

            interval_mask = (x_interval >= min_energy) & (x_interval <= max_energy)
            x_interval_masked = x_interval[interval_mask]

            # Get individual components of the Hypermet curve
            G, P, D, H, hypermet_c = hypermet_copy(x_interval_masked, *popt_hypermet)

            # Plot individual components
            plt.plot(x_interval_masked, G, label="Gaussian Component", color="blue", linestyle="--")
            plt.plot(x_interval_masked, P, label="Plateau Component", color="orange", linestyle="--")
            plt.plot(x_interval_masked, D, label="Exponential Tail Component", color="green", linestyle="--")
            plt.plot(x_interval_masked, H, label="Box-like Function Component", color="purple", linestyle="--")


            plt.plot(x_interval_masked, hypermet_c, "--", color = "green", label = f"Hypermet Fit (Interval {i+1})" )
            # Calculate Z-score
            
            # Calculate bin widths for the interval
            bin_widths = edges_Amptek[bin_min + 1:bin_max + 1] - edges_Amptek[bin_min:bin_max]

            # Evaluate the polynomial background at each bin center in the interval
            poly_bg = np.polyval(coefficients, x_interval)

            # Integrate background under the signal region using bin widths
            background_counts = np.sum(poly_bg * bin_widths)

            # Integrate signal (data minus background) using bin widths
            signal_counts = np.sum((y_data - poly_bg) * bin_widths)

            # For s+b (total counts in the region, for annotation)
            s_plus_b_curve_fit =  np.sum(y_data * bin_widths)

            # Z-score (C++ style)
            if background_counts > 0:
                Z_score = signal_counts / np.sqrt(background_counts)
                Z_score = max(0, Z_score)
            else:
                Z_score = 0
        except RuntimeError:
            print(f"Hypermet fit failed for interval {i+1}")
            #Fallback to gaussian fit
            try:
                p0_gaussian = [np.max(y_data), central_energy, sigma]
                popt_gaussian, _ = curve_fit(gaussian, x_interval, y_data, p0 = p0_gaussian)
                fitted_gaussian = gaussian(x_interval, *popt_gaussian)
                chi2_gaussian = reduced_chi_square(y_data, fitted_gaussian, 3)

                #DOuble Gaussian fit
                p0_double_gaussian = [np.max(y_data), central_energy, sigma, #first Gauss
                                      np.max(y_data) / 2, central_energy + sigma, sigma #second Gauss
                                      ]
                popt_double_gaussian, _ = curve_fit(double_gaussian, x_interval, y_data, p0 = p0_double_gaussian)
                fitted_double_gaussian = double_gaussian(x_interval, *popt_double_gaussian)
                A1, mu1, sigma1, A2, mu2, sigma2 = popt_double_gaussian
                G1 = A1 * np.exp(-((x_interval -mu1) ** 2) / (2 * sigma1 ** 2))
                G2 = A2 * np.exp(-((x_interval - mu2) ** 2) / (2 * sigma2 ** 2))
                print(f"A1: {A1}, mu1: {mu1}, sigma1: {sigma1}")
                print(f"A2: {A2}, mu2: {mu2}, sigma2: {sigma2}")
                chi2_double_gaussian = reduced_chi_square(y_data, fitted_double_gaussian, 6)

                if chi2_gaussian < chi2_double_gaussian:
                    y_fit = fitted_gaussian #y_single_fit
                    fit_label = f"Interval {i+1}: SIngle Gaussian (χ²ᵣ, = {chi2_gaussian:.2f})"
                    fit_color = "orange"
                    chosen_fit = "single Gauss"
                    chosen_chi2 = chi2_gaussian
                    plt.plot(x_interval, fitted_gaussian, label = f"SingleGaussianFit(interval {i+1})", color = "orange", linestyle = "--")
                    y_fit_bg_subtracted = fitted_gaussian - baseline_avg
                else:
                    y_fit = fitted_double_gaussian if fitted_double_gaussian is not None else fitted_gaussian
                    fit_label = f"Interval  {i+1}: Double_Gaussian (χ²ᵣ = {chi2_double_gaussian:.2f})"
                    fit_color = "purple" if fitted_double_gaussian is not None else "orange"
                    chosen_fit = "Double Gauss"
                    chosen_chi2 = chi2_double_gaussian
                    plt.plot(x_interval, G1, label = "First Gaussian", color = "blue", linestyle = "--")
                    plt.plot(x_interval, G2, label = "Second Gaussian", color = "orange", linestyle = "--")
                    plt.plot(x_interval, fitted_double_gaussian,  label = f"DoubleGaussFit(interval {i+1})", color = "purple",  linestyle = "--")
                    y_fit_bg_subtracted = fitted_double_gaussian - baseline_avg
                # plot the closed curve
                plt.plot(x_interval, y_fit, "--", color = fit_color, label = fit_label)
                
                # --- Always calculate these after the fit ---
                bin_widths = edges_Amptek[bin_min + 1:bin_max + 1] - edges_Amptek[bin_min:bin_max]
                poly_bg = np.polyval(coefficients, x_interval)
                background_counts = np.sum(poly_bg * bin_widths)
                s_plus_b_curve_fit = np.sum(y_data * bin_widths)
                signal_counts = np.sum((y_data - poly_bg) * bin_widths)
                if background_counts > 0:
                    Z_score = (signal_counts) / np.sqrt(background_counts)
                    Z_score = max(0, Z_score)
                else:
                    Z_score = 0
            except RuntimeError:
                print(f"Gaaussian fit also failedfor  interval {i+1}. skipping this interval.")
                signal_counts = np.nan
                background_counts = np.nan
                s_plus_b_curve_fit = np.nan
                continue
        
        # background subtraction
        #chi2_bkg_subtract, z_bkg_subtract, avg_bg = bkg_subtract_method(x_data, y_data, counts_Amptek, min_energy, max_energy, sigma)
        
        # background subtraction
        if i == 1:
            min_energy_bkg = central_energy - 4.5 * sigma
        else:
            min_energy_bkg = central_energy - 3 * sigma
        max_energy_bkg = central_energy + 3 * sigma

        #chi2_bkg_subtract, z_bkg_subtract, avg_bg, s_plus_b, b, s = bkg_subtract_method(x_data, y_data, counts_Amptek, central_energy, nSigmaB=3, fwhm=fwhm, use_left=True, use_right=True,
        #                                                               min_energy = min_energy_bkg, max_energy = max_energy_bkg)

        chi2_bkg_subtract, z_bkg_subtract, avg_bg, s_plus_b, b, s = bkg_subtract_method(x_data, y_data, counts_Amptek, central_energy, nSigmaB=3, fwhm=fwhm, use_left=True, use_right=True,
                                                                                         min_energy=min_energy_bkg, max_energy=max_energy_bkg, edges_Amptek=edges_Amptek)

        # Annotate
        text_x = central_energy
        text_y = np.max(y_fit) *1.2
        plt.text(
            text_x,
            text_y,
            f"Interval: {i+1}\n"
            f"Fit: {chosen_fit}\n"
            f"s_curve_fit: {signal_counts:.0f}\n"
            f"b_curve_fit: {background_counts:.0f}\n"
            f"s+b_curve_fit: {s_plus_b_curve_fit:.0f}\n"
            f"Curvefit χ²: {chosen_chi2:.2f}\n"
            f"Curvefit Z: {Z_score:.2f}\n"
            f"s+b: {s_plus_b:.0f}\n"
            f"b: {b:.0f}\n"
            f"s: {s:.0f}\n"
            f"Bkg_Sub χ²: {chi2_bkg_subtract:.2f}\n"
            f"Bkg_Sub Z: {z_bkg_subtract:.2f}",
            fontsize = 10,
            bbox = dict(facecolor = "white", alpha = 0.8),
            ha = "left",
        )
        """# Annotate
        text_x = central_energy
        text_y = np.max(y_fit) *1.2
        plt.text(
            text_x,
            text_y,
            f"Interval: {i+1}\nFit: {chosen_fit}\nχ²: {chosen_chi2:.2f}\nZ: {Z_score:.2f}",
            fontsize = 10,
            bbox = dict(facecolor = "white", alpha = 0.8),
            ha = "left",
        )"""
        plt.step(x_interval, y_data, where = "post", label = "Amptek (Interval Histogram)", color = "red", linewidth = 1)
        #Finalize the interval plot
        plt.yscale("log")
        plt.ylim(1, 1e4)
        plt.xlim(0, 120)
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.title(f"Interval {i+1}: [{min_energy:.1f} - {max_energy:.1f}] keV")
        #plt.legend(loc = "upper right")
        plt.grid(True, which = "both", linestyle ="--", linewidth = 0.5)
        plt.tight_layout()
        #plt.show()
        if output_pixel_dir:
            os.makedirs(output_pixel_dir, exist_ok=True)
            method_plot_path = os.path.join(output_pixel_dir, f"{filename_prefix}_curve_fit_interval{i+1}.png")
            plt.savefig(method_plot_path)
        plt.close()

        # For background_sub (bkg_subtract_method)
        results_summary.append({
            "fit_type": chosen_fit,
            "interval": i+1,
            "method": "background_sub",
            "s_plus_b": s_plus_b,
            "signal": s,
            "background": b,
            "chi2": chi2_bkg_subtract,
            "z_score_bg_subtract": z_bkg_subtract
        })
        
        # For curve_fit (polyfit baseline)
        results_summary.append({
            "fit_type": chosen_fit,
            "interval": i+1,
            "method": "curve_fit",
            "s_plus_b": s_plus_b_curve_fit,
            "signal": signal_counts,
            "background": background_counts,
            "chi2": chosen_chi2,
            "z_score_polyfit": Z_score
        })

        # For bg_methods.bg_global
        results_summary.append({
            "fit_type": chosen_fit,
            "interval": i+1,
            "method": "bg_global",
            "s_plus_b": np.sum(y_signal),
            "signal": np.sum(signal_minus_bg_global),
            "background": np.sum(bg_global),
            "chi2": chi2_bg_global,
            "z_score_bg_global": z_score_bg_global
        })
        
        all_score_results["bkg_subtract"].append({
            "z_score": z_bkg_subtract,
            "fit_type": chosen_fit,
            "signal": s
        })

        all_score_results["polyfit"].append({
            "z_score": Z_score,
            "fit_type": chosen_fit,
            "signal": s
        })
        
        all_score_results["bg_global"].append({
            "z_score": z_score_bg_global,
            "fit_type": chosen_fit,
            "signal": s
        })
    return all_score_results

def save_stats_summary(output_pixel_dir, results_summary):
    stats_path = os.path.join(output_pixel_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write("Summary of all intervals and methods:\n")
        f.write(f"{'Interval':>8} {'Method':>15} {'S+B':>12} {'Signal':>12} {'Bkg':>12} {'Chi2':>10} {'Z-score':>10}\n")
        for res in results_summary:
            f.write(f"{res['interval']:>8} {res['method']:>15} {res['s_plus_b']:12.2f} {res['signal']:12.2f} {res['background']:12.2f} {res['chi2']:10.2f} {res['z_score']:10.2f}\n")

def process_pixel(pixel_dir, intervals, intervals_2):
    # Load histograms for this pixel
    histograms = read_histograms_from_files(pixel_dir)
    signals = extract_signals(histograms, intervals)

    # Create output directory
    output_pixel_dir = os.path.join(pixel_dir, "output_pixel_dir")
    os.makedirs(output_pixel_dir, exist_ok=True)

    # Save signals as .npy
    np.save(os.path.join(output_pixel_dir, "signals.npy"), signals)

    # Save plots
    #visualize_stacked_histograms(histograms, intervals, intervals_2, output_pixel_dir=output_pixel_dir, plot_name="signals_with_stacked_hists.png")
    visualize_signals_with_hypermet(histograms, intervals, signals, intervals_2, output_pixel_dir=output_pixel_dir, plot_name="signals_with_hypermet.png")
    save_stats_summary(output_pixel_dir, results_summary)





# Example usage
if __name__ == "__main__":

    #for only running once uncomment this line, else comment this
    #pixel_dir = "C:/Users/karth/Downloads/Code_to_convert/hist_rebin"

    # Usage: python pixel_level_Bg_Sub_Curvefit.py /path/to/pixel_dir
    if len(sys.argv) < 2:
        print("Usage: python pixel_level_Bg_Sub_Curvefit.py /path/to/pixel_dir")
        sys.exit(1)

    pixel_dir = sys.argv[1]
    # Define intervals for analysis
    intervals = [
        (11.6, 0.5), #(central energy, FWHM)
        (42.6, 0.9),
        (55.1, 1.28),
        (67.7, 1.55),
    ]

    # Define intervals_2 for analysis to exclude (for signal extraction from background)
    intervals_2 = [
        (48.70, 1.1),  #(central energy, FWHM)
    ]

    # Read histograms from .hist files
    histograms = read_histograms_from_files(pixel_dir)

    # Extract signals
    signals = extract_signals(histograms, intervals)

    process_pixel(pixel_dir, intervals, intervals_2)


    # Print extracted signals
    for signal in signals:
        print(f"Interval: {signal['interval']}")
        print(f"  Amptek Signal: {signal['Amptek_signal']:.2f}")
        
        print()

print("\nSummary of all intervals and methods:")
print(f"{'Interval':>8} {'Method':>15} {'S+B':>12} {'Signal':>12} {'Bkg':>12} {'Chi2':>10} {'Z-score':>10}")
for res in results_summary:
    print(f"{res['interval']:>8} {res['method']:>15} {res['s_plus_b']:12.2f} {res['signal']:12.2f} {res['background']:12.2f} {res['chi2']:10.2f} {res['z_score']:10.2f}")


#output_pixel_dir = os.path.join(pixel_dir, "output")


#visualize_stacked_histograms(histograms, intervals, intervals_2, output_pixel_dir = output_pixel_dir, plot_name = "signals_with_stacked_hists.png" )

#visualize_signals_with_hypermet(histograms,  intervals, signals, intervals_2, output_pixel_dir = output_pixel_dir, plot_name = "signals_with_hypermet.png")

#plt.show()
"""
if output_pixel_dir:
    os.makedirs(output_pixel_dir, exist_ok=True)
    plt.savefig(os.path.join(output_pixel_dir, plot_name))
plt.close()
"""