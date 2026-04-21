import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}")
    else:
        print(f"Group: {name}")

with h5py.File('weights.weights.h5', 'r') as f:
    f.visititems(print_structure)
