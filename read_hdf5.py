import h5py

def print_hdf5_contents(filename):
    """Reads an HDF5 file and prints its contents."""
    def recursively_print_group(name, obj):
        """Recursively prints the contents of a group or dataset."""
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data: {obj[()]}")

    with h5py.File(filename, "r") as f:
        print(f"Reading HDF5 file: {filename}")
        f.visititems(recursively_print_group)

def print_results_group(filename):
    """Reads an HDF5 file and prints the contents of the 'results' group."""
    with h5py.File(filename, "r") as f:
        if 'results' in f:
            print("Reading 'results' group:")
            results_group = f['results']
            for name, obj in results_group.items():
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Data: {obj[()]}")
                elif isinstance(obj, h5py.Group):
                    print(f"Nested Group: {name}")
        else:
            print("'results' group not found in the HDF5 file.")

def extract_dataset_from_results(filename, dataset_name):
    """
    Extracts a NumPy array from a specific dataset in the 'results' group of an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to extract.

    Returns:
        numpy.ndarray: The data stored in the specified dataset.
    """
    with h5py.File(filename, "r") as f:
        if 'results' in f:
            results_group = f['results']
            if dataset_name in results_group:
                dataset = results_group[dataset_name]
                return dataset[()]  # Extract and return as a NumPy array
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in the 'results' group.")
        else:
            raise KeyError("'results' group not found in the HDF5 file.")

if __name__ == "__main__":
    # Example usage
    filename = "Output/aero_7dim_asym/N6_volmax/opt_vars.hdf5"
    print_results_group(filename)

    # or extract specific dataset
    dataset_name = "x"  # Replace with the name of the dataset you want to extract
    try:
        data = extract_dataset_from_results(filename, dataset_name)
        print(f"Extracted data from dataset '{dataset_name}':\n{data}")
    except KeyError as e:
        print(e)