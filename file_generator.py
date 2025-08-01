import os
import shutil
import csv
import pandas as pd
from pathlib import Path

def create_spectrum_data():
    # Configuration
    source_file = "source_spectrum.hist"  # Change this to your source .hist file path
    output_folder = "3d_spectrum_data"
    csv_filename = "spectrum_data.csv"
    
    # Dimensions
    X_range = 10    #400
    Y_range = 10    #400
    Rot_range = 10   #20
    
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' not found!")
        print("Please make sure the source .hist file exists and update the 'source_file' variable.")
        return
    
    print(f"Starting to copy {X_range * Y_range * Rot_range:,} files...")
    
    # Prepare CSV data
    csv_data = []
    file_counter = 1
    
    # Generate files and CSV data
    for x in range(1, X_range + 1):  # X: 1 to 400
        for y in range(1, Y_range + 1):  # Y: 1 to 400
            for rot in range(Rot_range):  # Rot_idx: 0 to 19
                # Create new filename
                new_filename = f"spectrum_{file_counter}.hist"
                new_filepath = output_path / new_filename
                
                # Copy the source file with new name
                try:
                    shutil.copy2(source_file, new_filepath)
                except Exception as e:
                    print(f"Error copying file {file_counter}: {e}")
                    continue
                
                # Add to CSV data
                csv_data.append({
                    'X': x,
                    'Y': y, 
                    'Rot_idx': rot,
                    'spectrum_file_path': str(new_filepath.absolute())
                })
                
                file_counter += 1
                
                # Progress indicator
                if file_counter % 10000 == 0:
                    print(f"Processed {file_counter:,} files...")
    
    # Create CSV file
    csv_path = Path(csv_filename)
    
    print(f"Creating CSV file: {csv_path.absolute()}")
    
    # Write CSV using pandas for better performance
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Successfully created:")
    print(f"   - {len(csv_data):,} spectrum files in '{output_folder}' folder")
    print(f"   - CSV file '{csv_filename}' with {len(csv_data):,} rows")
    print(f"   - Total files created: {X_range * Y_range * Rot_range:,}")
    print(f"   - X range: 1 to {X_range}")
    print(f"   - Y range: 1 to {Y_range}")
    print(f"   - Rotation range: 0 to {Rot_range - 1}")
    print(f"   - Total storage used: ~{(X_range * Y_range * Rot_range * 23 / 1024 / 1024):.1f} GB")
    
    # Display first few rows of CSV for verification
    print(f"\nFirst 5 rows of CSV:")
    print(df.head())
    
    print(f"\nLast 5 rows of CSV:")
    print(df.tail())

def create_spectrum_data_alternative():
    """
    Alternative version using standard CSV writer (if pandas is not available)
    """
    # Configuration
    source_file = "source_spectrum.hist"  # Change this to your source .hist file path
    output_folder = "3d spectrum data"
    csv_filename = "spectrum_data.csv"
    
    # Dimensions
    X_range = 400
    Y_range = 400
    Rot_range = 20
    
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' not found!")
        print("Please make sure the source .hist file exists and update the 'source_file' variable.")
        return
    
    print(f"Starting to copy {X_range * Y_range * Rot_range:,} files...")
    
    # Open CSV file for writing
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['X', 'Y', 'Rot_idx', 'spectrum_file_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        file_counter = 1
        
        # Generate files and CSV data
        for x in range(1, X_range + 1):  # X: 1 to 400
            for y in range(1, Y_range + 1):  # Y: 1 to 400
                for rot in range(Rot_range):  # Rot_idx: 0 to 19
                    # Create new filename
                    new_filename = f"spectrum_{file_counter}.hist"
                    new_filepath = output_path / new_filename
                    
                    # Copy the source file with new name
                    try:
                        shutil.copy2(source_file, new_filepath)
                    except Exception as e:
                        print(f"Error copying file {file_counter}: {e}")
                        continue
                    
                    # Write to CSV
                    writer.writerow({
                        'X': x,
                        'Y': y,
                        'Rot_idx': rot,
                        'spectrum_file_path': str(new_filepath.absolute())
                    })
                    
                    file_counter += 1
                    
                    # Progress indicator
                    if file_counter % 10000 == 0:
                        print(f"Processed {file_counter:,} files...")
    
    print(f"✅ Successfully created:")
    print(f"   - {(X_range * Y_range * Rot_range):,} spectrum files in '{output_folder}' folder")
    print(f"   - CSV file '{csv_filename}' with {(X_range * Y_range * Rot_range):,} rows")

if __name__ == "__main__":
    print("3D Spectrum Data Generator")
    print("=" * 50)
    
    # Update the source_file variable with your actual .hist file path
    print("⚠️  IMPORTANT: Update the 'source_file' variable with your actual .hist file path")
    print("   Current setting: 'source_spectrum.hist'")
    print()
    
    # Ask user which version to use
    choice = input("Choose version:\n1. Standard version (requires pandas)\n2. Alternative version (no pandas required)\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            create_spectrum_data()
        except ImportError:
            print("Pandas not found. Using alternative version...")
            create_spectrum_data_alternative()
    elif choice == "2":
        create_spectrum_data_alternative()
    else:
        print("Invalid choice. Using standard version...")
        try:
            create_spectrum_data()
        except ImportError:
            print("Pandas not found. Using alternative version...")
            create_spectrum_data_alternative()