import os
import shutil
from sample_tracer import main, allowed_modes

# Directory to move the generated files to
output_directory = "/home/salnahari/testing_dir/tt-metal/models/"
os.makedirs(output_directory, exist_ok=True)

# Iterate through each mode
for mode in allowed_modes:
    print(f"Processing mode: {mode}")
    
    # Prepare arguments for the main function
    args_dict = {
        "model": mode,
        "input_shape": [[1, 3, 1280, 800]],  # Example input shape
        "input_dtype": ["float32"],  # Default data type
    }
    
    try:
        # Call the main function from sample_tracer.py
        main(args_dict)
        print("Successfully processed mode:", mode)
    except Exception as e:
        print(f"Error while processing mode {mode}: {e}")
        continue

    # Move the generated graph.py file to the output directory
    source_file = "graph.xlsx"
    if os.path.exists(source_file):
        destination_file = os.path.join(output_directory, f"graph_{mode}.xlsx")
        shutil.move(source_file, destination_file)
        print(f"Moved {source_file} to {destination_file}")
    else:
        print(f"{source_file} not found for mode {mode}")