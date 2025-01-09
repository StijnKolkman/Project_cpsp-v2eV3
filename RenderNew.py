from v2ecore.renderer import EventRenderer
import h5py
import numpy as np  # Ensure numpy is imported for handling arrays


file_path = 'output/eventsH5.h5' # file path for the h5 file containing the events


renderer = EventRenderer(
    full_scale_count=1,
    output_path="output",
    dvs_vid="dvs_video.avi",
    preview=True,
    #exposure_mode=ExposureMode.DURATION,
    exposure_value=1/30.0,  # 30 FPS
    avi_frame_rate=30,
)

with h5py.File(file_path, 'r') as hdf:
    dataset_name = 'events'
    if dataset_name in hdf:
        dataset = hdf[dataset_name]
        print(f"\nShape of the dataset '{dataset_name}': {dataset.shape}")
        print(f"Data type of the dataset: {dataset.dtype}")

        # Load the entire dataset into memory
        all_data = dataset[:]  # This loads all the data into a NumPy array

        # Convert the relevant columns to float if necessary
        all_data = all_data.astype(np.float32)  # Convert entire array to float32

        all_data[:, 1] = all_data[:, 1] / 1e6 #change time to seconds
        print(f"Loaded {len(all_data)} entries from '{dataset_name}'.")

        # Now you can process all_data as needed
        # For example, printing the first few rows:
        print(all_data[:10])  # Print the first 10 entries to verify

    else:
        print(f"Dataset '{dataset_name}' not found in the file.")

#test = all_data[1][1]
#print(test)

# Filter rows where the first column is equal to 0
filtered_data = all_data[all_data[:, 0] == 0]
# Divide the second column by 1000
filtered_data_no_first_col = filtered_data[:, 1:]  # Select all columns except the first
height, width = 720, 1280
renderer.render_events_to_frames(filtered_data_no_first_col, height, width)


'''
events = np.array([
    [0.01, 10, 20, 1],
    [0.02, 15, 25, -1],  ...
])  
# Events with [timestamp, x, y, polarity]
height, width = 128, 128
renderer.render_events_to_frames(events, height, width)
'''
