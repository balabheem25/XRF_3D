from main_3d import main_3d
from file_io import set_file_format

intervals = [
    (11.6, 0.5),
    (42.6, 0.9),
    (55.1, 1.28),
    (67.7, 1.55),
]
intervals_2 = [
    (48.70, 1.1),
]



main_dir = "DATA"
if __name__ == "__main__":

    set_file_format(".hist") # or ".hist" based on the GUI
    csv_path = "C:/Users/karth/Downloads/Code_to_convert/trail/3D_Recon/spectrum_data.csv"

    main_3d(csv_path, intervals, intervals_2)
    
    

#  run    source /Users/balabheem/Downloads/trail/fitenv/bin/activate