import h5py

file_path = 'output/eventsH5.h5'

with h5py.File(file_path, 'r') as hdf:
    #print("Keys in the HDF5 file:")
    #for key in hdf.keys():
    #    print(key)

    dataset_name = 'events'
    if dataset_name in hdf:
        dataset = hdf[dataset_name]
        print(f"\nShape of the dataset '{dataset_name}': {dataset.shape}")
        print(f"Data type of the dataset: {dataset.dtype}")

        data = dataset[:10]  #Get a part of the data
        print(f"Data from '{dataset_name}':")
        print(data)
    else:
        print(f"Dataset '{dataset_name}' not found in the file.")
