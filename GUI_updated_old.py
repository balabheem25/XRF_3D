import os

import logging, pathlib

LOG_PATH = pathlib.Path(__file__).with_name("log.txt")

logging.basicConfig(
    filename = LOG_PATH,      # only a FileHandler, no console handler
    filemode = "w",           # overwrite each run (“a” to append)
    level    = logging.DEBUG, # or INFO/WARNING …
    format   = "[%(asctime)s %(levelname).1s:%(name)s] %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S"
)

# convenience logger for each module
logger = logging.getLogger(__name__)

import sys
import numpy as np
import pandas as pd
import pydicom
import datetime
import shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QComboBox,
    QStackedWidget, QVBoxLayout, QWidget, QMessageBox, QProgressBar, QListWidget,
    QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QTextEdit, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtCore import Qt, QTimer, QPoint
from concurrent.futures import ThreadPoolExecutor
import torch
from main_2d import main_2d
from main_3d import main_3d
from main import intervals, intervals_2
from file_io import set_file_format, read_histograms_from_files, auto_set_file_format_from_path, get_file_format, _normalise_columns
from pixel_level_Bg_Sub_Curvefit import read_histograms_from_files
from hdf5_io import save_signal_data_hdf5

from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

import sys, pathlib, logging
log_file = open(pathlib.Path(__file__).with_name("log.txt"), "w", buffering=1, encoding="utf‑8")
sys.stdout = sys.stderr = log_file        # every plain print → log.txt


default_file_path = "C:/Users/karth/Downloads/Code_to_convert/trail/3D_Recon/spectrum_data.csv"         #spectrum_data.csv"
global file_format
"""class FileFormatSelector(QWidget):
    def __init__(self, on_next):
        super().__init__()
        layout = QVBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems([".hist", ".csv"])

        layout.addWidget(QLabel("Select File Format:"))
        layout.addWidget(self.format_combo)

        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: on_next(self.format_combo.currentText()))
        layout.addWidget(next_btn)

        self.setLayout(layout)
"""

def dict2array(signal_dict, method_order=("bkg_subtract", "polyfit", "bg_global")):
    n_intervals = len(signal_dict[method_order[0]])
    rot, y, x   = signal_dict[method_order[0]][0].shape
    arr = np.zeros((len(method_order), n_intervals, rot, y, x), dtype=np.float32)
    for m_idx, m in enumerate(method_order):
        for i_idx in range(n_intervals):
            arr[m_idx, i_idx] = signal_dict[m][i_idx]
    return arr

class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.voxel_size_input = QLineEdit("0.1")
        self.slice_thickness_input = QLineEdit("1.0")

        layout.addWidget(QLabel("Voxel Size (mm):"))
        layout.addWidget(self.voxel_size_input)
        layout.addWidget(QLabel("Slice Thickness (mm):"))
        layout.addWidget(self.slice_thickness_input)

        self.setLayout(layout)

    def get_settings(self):
        return float(self.voxel_size_input.text()), float(self.slice_thickness_input.text())


class FolderSelector(QWidget):
    def __init__(self, file_format, on_run):
        super().__init__()
        self.file_format = file_format
        self.main_folder = None
        self.layout = QVBoxLayout()
        layout = self.layout

        self.file_label = QLabel("No file selected")
        layout.addWidget(QLabel(f"Selected Format: {file_format}"))
        
        # Add a text input field for the file path
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Enter the path to the CSV file")
        layout.addWidget(self.file_path_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        layout.addWidget(browse_btn)

        layout.addWidget(self.file_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.settings_panel = SettingsPanel()
        layout.addWidget(self.settings_panel)

        run_btn = QPushButton("Run Reconstruction")
        run_btn.clicked.connect(lambda: self.run_with_debug(on_run))
        layout.addWidget(run_btn)

        self.setLayout(layout)
        self.setAcceptDrops(True)

    """
    def process_dragged_folder(dragged_folder_path):
        # Print the folder path for debugging
        print(f"Folder path received: {dragged_folder_path}")
        # Construct the path to the "spectrum_data.csv" file
        csv_file_path = os.path.join(dragged_folder_path, "spectrum_data.csv")
    
        # Check if the file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"'spectrum_data.csv' not found in the folder: {dragged_folder_path}")
    
        # Read the CSV file
        data = pd.read_csv(csv_file_path)
    
        # Process the data (replace this with your logic)
        print(f"Successfully read the file: {csv_file_path}")
        print(data.head())  # Example: Display the first few rows of the CSV
    """
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select csv file", "", "csv files (*.csv)")
        if file_path:
            self.file_path_input.setText(file_path)
        
        
    def run_with_debug(self, on_run):
        #Debugging; print the file path before running
        input_path = self.file_path_input.text()
        print(f"file path entered: {input_path}")
        on_run(input_path, self.file_format, self.progress_bar, self.settings_panel.get_settings())
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            #event.acceptProposedAction()
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()  # Get the dropped folder path
            # Check if the dropped item is a .csv file
            if file_path.endswith(".csv"):
                #self.file_label = QLabel("No file selected")  # Update the label to show the file path
                #layout.addWidget(self.file_label)
                self.file_label.setText(file_path)

                # Debugging: Print the file path
                print(f"Selected CSV file path: {file_path}")

            # Process the CSV file
                try:
                    self.process_csv_file(file_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to process the file: {e}")
                    print(f"Error processing the file: {e}")
            else:
                QMessageBox.warning(self, "Invalid File", "Please drop a valid .csv file.")
                print(f"Invalid input: {file_path} is not a .csv file.")
                # Process the default file
                try:
                    self.file_label.setText(default_file_path)  # Update the label to show the default file path
                    self.process_csv_file(default_file_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to process the default file: {e}")
                    print(f"Error processing the default file: {e}")
        else:
            # If no valid file is dropped, use the default file path
            QMessageBox.warning(self, "Invalid Input", f"No valid file dropped. Using default file: {default_file_path}")
            print("No valid file dropped. Falling back to default file path.")

            # Process the default file
            try:
                self.file_label.setText(default_file_path)  # Update the label to show the default file path
                self.process_csv_file(default_file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process the default file: {e}")
                print(f"Error processing the default file: {e}")

                
    
    #def process_csv_file(self, file_path):
        # Pass the file path to the main processing function
    #    self.run_main_processing(file_path, "csv", self.progress_bar, self.settings)
        
    
    def process_csv_file(self, file_path):
        """
        Process the CSV file to generate a 5D matrix of counts.
        """
        import traceback
        traceback.print_exc()
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            print("normalized columns:", df.columns.tolist())

            df.columns = df.columns.str.strip() # Strip whitespace from column names

            # NEW: Detect file format from the 5th row (index 4)
            spectrum_file_path = str(df.loc[4, "spectrum_file_path"]).strip()
            file_ext = os.path.splitext(spectrum_file_path)[1].lower()
            if file_ext not in (".hist", ".csv"):
                raise ValueError(f"Unknown file extension: {file_ext}")
            set_file_format(file_ext)
            print(f"[INFO] Detected and set file format to: {file_ext}")
            file_format = file_ext
            # ✅ Step 3: Print first 5 lines of the detected file for inspection
            try:
                with open(spectrum_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    print(f"\n[DEBUG] First 5 lines of: {spectrum_file_path}")
                    for i in range(5):
                        print(f.readline().strip())
            except Exception as preview_err:
                print(f"[WARN] Could not preview file {spectrum_file_path}: {preview_err}")

            # ── NEW ── skip the first *data* row if Rot_idx is missing or non‑numeric
            if not pd.to_numeric(df.iloc[0]["rot_idx"], errors="coerce") >= 0:
                df = df.iloc[1:].reset_index(drop=True)
            # ─────────


            #print("Columns in the CSV file:", df.columns)
            # Ensure the required columns exist
            required_columns = ['x', 'x', 'rot_idx', 'spectrum_file_path']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Ensure numeric columns are properly converted
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['rot_idx'] = pd.to_numeric(df['rot_idx'], errors='coerce')

            # Drop rows with invalid data
            df = df.dropna(subset=['x', 'y', 'rot_idx', 'spectrum_file_path'])

            # Get the dimensions of the 5D matrix
            x_max = int(df['x'].max()) + 1
            y_max = int(df['y'].max()) + 1
            rot_max = int(df['rot_idx'].max()) + 1
            num_methods = 3
            num_intervals = len(intervals)

            # Initialize the 3D matrix
            counts_matrix = np.zeros((num_methods, num_intervals, x_max, y_max, rot_max), dtype = np.float32)

            # Multithreaded processing of spectrum files
            def process_row(row):
                x = int(row['x'])
                y = int(row['y'])
                rot_idx = int(row['rot_idx'])
                spectrum_file_path = str(row['spectrum_file_path']).strip()
                # Debugging log for permissions
                if not os.path.exists(spectrum_file_path):
                    print(f"[Missing] File not found: {spectrum_file_path}")
                    return x, y, rot_idx, np.zeros((num_methods, num_intervals), dtype=np.float32)
                elif not os.access(spectrum_file_path, os.R_OK):
                    print(f"[Permission Denied] Cannot read file: {spectrum_file_path}")
                    return x, y, rot_idx, np.zeros((num_methods, num_intervals), dtype=np.float32)
                else:
                    print(f"[OK] Reading file: {spectrum_file_path}")
                #counts = self.process_spectrum_file(spectrum_path)
                #signals = self.process_spectrum_file(spectrum_path, intervals, intervals_2) #returns (num_methods x num_intervals)

                # main_2d returns a dict with 3 methods × N intervals
                #pixel_dir = os.path.dirname(spectrum_path)
                signal_maps = main_2d(spectrum_file_path, intervals, intervals_2)

                #histograms = read_histograms_from_files(spectrum_path)

                # convert to a (methods, intervals) numpy array
                method_order = ["bkg_subtract", "polyfit", "bg_global"]
                signals = np.asarray([signal_maps[m] for m in method_order], dtype = np.float32)
                return x, y, rot_idx, signals

            with ThreadPoolExecutor() as ex:
                results = list(ex.map(process_row, df.to_dict('records')))

            # Populate the 3D matrix with the results
            for x, y, rot_idx, signals in results:
                counts_matrix[:, :, x, y, rot_idx] = signals 
                #counts_matrix[x, y, rot_idx] = counts

            # Save the 5D matrix for further processing
            np.save("counts_matrix.npy", counts_matrix)
            print("5‑D counts matrix saved to counts_matrix.npy")
            return counts_matrix
            #self.save_counts_matrix(counts_matrix)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process the CSV file: {e}")
            print(f"Error processing the CSV_main_except file: {e}")

    def save_counts_matrix(self, counts_matrix):
        """
        Save the 5D matrix to a file or pass it to the next stage of processing.
        """
        output_path = "counts_matrix.npy"
        np.save(output_path, counts_matrix)
        print(f"5D counts matrix saved to {output_path}")
        return counts_matrix

class ImagePreview(QWidget):
    def __init__(self, dicom_folder):
        super().__init__()
        self.dicom_folder = dicom_folder
        self.layout = QHBoxLayout()

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        self.layout.addWidget(self.image_list)

        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.layout.addWidget(self.graphics_view)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)

        self.setLayout(self.layout)
        self.populate_image_list()

    def populate_image_list(self):
        files = sorted([f for f in os.listdir(self.dicom_folder) if f.endswith(".dcm")])
        self.image_list.addItems(files)

    def load_image(self, item):
        filepath = os.path.join(self.dicom_folder, item.text())
        self.log_box.append(f"Loaded: {filepath}")
        ds = pydicom.dcmread(filepath)
        image = ds.pixel_array
        height, width = image.shape
        qimage = QImage(image.tobytes(), width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(512, 512, Qt.KeepAspectRatio)
        self.scene.clear()
        self.scene.addPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D XRF Image Reconstruction GUI")
        self.setMinimumSize(1000, 700)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        #self.file_format_screen = FileFormatSelector(self.go_to_folder_selection)
        #self.stack.addWidget(self.file_format_screen)
        file_format = ".hist"
        self.folder_selector = FolderSelector(file_format, self.run_main_processing)
        self.stack.addWidget(self.folder_selector)
        self.stack.setCurrentWidget(self.folder_selector)

    def go_to_folder_selection(self, file_format):
        self.folder_selector = FolderSelector(file_format, self.run_main_processing)
        self.stack.addWidget(self.folder_selector)
        self.stack.setCurrentWidget(self.folder_selector)

    def run_main_processing(self, input_path, file_format, progress_bar, settings):
        # Validate the input path
        if not input_path or not isinstance(input_path, str):
            QMessageBox.warning(self, "Warning", "Please provide a valid CSV file path.")
            print("Error: No file path provided or invalid input.")
            input_path =default_file_path  # Set to default file path if input is invalid
            #return

        # Check if the input path is a valid file
        if os.path.isfile(input_path):
            if not input_path.endswith(".csv"):
                QMessageBox.warning(self, "Warning", "Invalid file format. Please provide a CSV file.")
                print(f"Error: Invalid file format for file: {input_path}")
                return

            # Debugging: Print the file path
            print(f"Processing file: {input_path}")
        else:
            QMessageBox.warning(self, "Warning", "Invalid input. Please provide a valid CSV file path.")
            print(f"Error: {input_path} is not a valid file.")
            return
        
        # Process CSV and get 5D counts matrix
        counts_matrix = self.folder_selector.process_csv_file(input_path)

        #counts_matrix = np.load("counts_matrix.npy")  # shape: (methods, intervals, x, y, rot_idx)

        # Now call main_3d to reconstruct images based on counts_matrix
        intervals = [(11.6, 0.5), (42.6, 0.9), (55.1, 1.28), (67.7, 1.55)]
        intervals_2 = [(48.70, 1.1)]

        progress_bar.setValue(0)
        QApplication.processEvents()

        folder = os.path.dirname(input_path)   # gets the directory of thee csv file
        output_dict = main_3d(counts_matrix, intervals, intervals_2, input_path)
        output_array = dict2array(output_dict)
        progress_bar.setValue(80)
        QApplication.processEvents()

        self.save_as_dicom_hdf5(output_array, folder, progress_bar, settings, intervals)
        QMessageBox.information(self, "Done", "Reconstruction and DICOM export completed!")
        folder = os.path.dirname(input_path)  # gets the directory of the csv file
        dicom_folder = os.path.join(folder, "dicom_hdf5_output")
        self.preview_widget = ImagePreview(dicom_folder)
        self.stack.addWidget(self.preview_widget)
        self.stack.setCurrentWidget(self.preview_widget)

    def save_as_dicom_hdf5(self, data, folder, progress_bar, settings, intervals):
        voxel_size, slice_thickness = settings
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)
        output_dir = os.path.join(folder, "dicom_hdf5_output")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        # data shape: (methods, intervals, slices, height, width)
        num_methods = data.shape[0]
        num_intervals = data.shape[1]
        total = num_methods * num_intervals * data.shape[2]  # slices count
        #total = data.shape[0] * data.shape[1]
        count = 0
        for method_idx in range(num_methods):
            for interval_idx in range(num_intervals):
                interval_data = data[method_idx, interval_idx]  # shape: (slices, height, width)
                for slice_idx in range(interval_data.shape[0]):
                    slice_data = interval_data[slice_idx]  # shape: (height, width)
                    filename = os.path.join(output_dir, f"method_{method_idx}_interval_{interval_idx}_slice_{slice_idx}.dcm")
                    #max_val = np.max(data)
                    #if max_val == 0 or np.isnan(max_val):
                    #    slice_data = np.zeros_like(data[interval_idx, slice_idx], dtype=np.uint8)
                    #else:
                    #    slice_data = (data[interval_idx, slice_idx] * 255.0 / max_val).astype(np.uint8)
                    #slice_data = (data[interval_idx, slice_idx] * 255.0 / np.max(data)).astype(np.uint8)

                    file_meta = pydicom.dataset.FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = generate_uid()
                    file_meta.MediaStorageSOPInstanceUID = generate_uid()
                    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
                    ds.Modality = "OT"
                    ds.StudyDate = datetime.date.today().strftime('%Y%m%d')
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds.SOPInstanceUID = generate_uid()
                    ds.StudyInstanceUID = generate_uid()
                    ds.SliceThickness = slice_thickness
                    ds.PixelSpacing = voxel_size
                    ds.Rows, ds.Columns = slice_data.shape
                    #ds.PatientName = "Test object"
                    #ds.ImagePositionPatient = [0, 0, slice_idx * ds.SliceThickness]
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    ds.BitsAllocated = 16
                    ds.BitsStored = 16
                    ds.SamplesPerPixel = 1
                    ds.HighBit = 15
                    ds.PixelRepresentation = 0
                    ds.InstanceNumber = slice_idx
                    ds.PixelData = slice_data.astype(np.uint16).tobytes()
                    

                    filename = os.path.join(output_dir, f"method_{method_idx}_interval_{interval_idx}_slice_{slice_idx}.dcm")
                    ds.save_as(filename)

                    count += 1
                    progress = int((count / total) * 100)
                    progress_bar.setValue(progress)
                    QApplication.processEvents()
                    print(f"Saved slice {slice_idx} for interval {interval_idx}, method {method_idx}")

            hdf5_path = os.path.join(folder, "all_signals_h5")
            save_signal_data_hdf5(
                filepath=r"C:\Users\karth\Downloads\Code_to_convert\trail\3D_Recon\signal_matrix.h5",
                signal_data=data,
                voxel_size=(0.1,0.1,0.1),
                slice_thickness=0.1,
                intervals=intervals,
            )
            progress_bar.setValue(100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


