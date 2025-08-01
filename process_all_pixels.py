import os
from pixel_level_Bg_Sub_Curvefit import (
    read_histograms_from_files,
    extract_signals,
    #visualize_stacked_histograms,
    visualize_signals_with_hypermet,
)

main_dir = "C:\Users\karth\Downloads\Code_to_convert\trail\3D_Recon"

from concurrent.futures import ProcessPoolExecutor

def process_pixel_dir(pixel_dir):
    histograms = read_histograms_from_files(pixel_dir)
    # Define intervals (you may want to load these from a config)
    intervals = [
        (11.6, 0.5),
        (42.6, 0.9),
        (55.1, 1.28),
        (67.7, 1.55),
    ]
    intervals_2 = [
        (48.70, 1.1),
    ]
    signals = extract_signals(histograms, intervals)
    output_pixel_dir = os.path.join(pixel_dir, "output")
    os.makedirs(output_pixel_dir, exist_ok=True)
    #visualize_stacked_histograms(histograms, intervals, intervals_2, output_pixel_dir=output_pixel_dir, plot_name="signals_with_stacked_hists.png")
    visualize_signals_with_hypermet(histograms, intervals, signals, intervals_2, output_pixel_dir=output_pixel_dir, plot_name="signals_with_hypermet.png")
    return True

pixel_dirs = []
for rot in os.listdir(main_dir):
    rot_path = os.path.join(main_dir, rot)
    if not os.path.isdir(rot_path): continue
    for pixel in os.listdir(rot_path):
        pixel_dir = os.path.join(rot_path, pixel)
        if os.path.isdir(pixel_dir):
            pixel_dirs.append(pixel_dir)

with ProcessPoolExecutor() as executor:
    list(executor.map(process_pixel_dir, pixel_dirs))








"""
# Loop over all rotations and pixels
for rot in os.listdir(main_dir):
    rot_path = os.path.join(main_dir, rot)
    if not os.path.isdir(rot_path):
        continue
    for pixel in os.listdir(rot_path):
        pixel_dir = os.path.join(rot_path, pixel)
        if not os.path.isdir(pixel_dir):
            continue

        # Change working directory to pixel_dir so .hist files are found
        #os.chdir(pixel_dir)
        histograms = read_histograms_from_files(pixel_dir)

        # Define intervals (you may want to load these from a config)
        intervals = [
            (11.6, 0.5),
            (42.6, 0.9),
            (55.1, 1.28),
            (67.7, 1.55),
        ]
        intervals_2 = [
            (48.70, 1.1),
        ]

        signals = extract_signals(histograms, intervals)
        output_pixel_dir = os.path.join(pixel_dir, "output")
        os.makedirs(output_pixel_dir, exist_ok=True)

        #visualize_stacked_histograms(histograms, intervals, intervals_2, output_pixel_dir=output_pixel_dir, plot_name="signals_with_stacked_hists.png")
        visualize_signals_with_hypermet(histograms, intervals, signals, intervals_2, output_pixel_dir=output_pixel_dir, plot_name="signals_with_hypermet.png")
"""