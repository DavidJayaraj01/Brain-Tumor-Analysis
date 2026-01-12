import h5py
import numpy as np

def explore_hdf5(name, obj):
    """Recursively explore HDF5 structure"""
    if isinstance(obj, h5py.Dataset):
        print(f'  {name}: Dataset, shape={obj.shape}, dtype={obj.dtype}')
        if obj.size < 10:  # Print small values
            print(f'    Value: {obj[...]}')
    elif isinstance(obj, h5py.Group):
        print(f'  {name}: Group')

# Load first .mat file to inspect structure
print('Inspecting: datasets/brainTumorDataPublic_1-766/1.mat\n')
with h5py.File('datasets/brainTumorDataPublic_1-766/1.mat', 'r') as f:
    print('Top-level keys:', list(f.keys()))
    print('\nFull structure:')
    f.visititems(explore_hdf5)
    
    # Access cjdata specifically
    if 'cjdata' in f:
        print('\n\nAccessing cjdata directly:')
        cjdata = f['cjdata']
        for key in cjdata.keys():
            item = cjdata[key]
            print(f'  {key}: {type(item)}')
            if isinstance(item, h5py.Dataset):
                print(f'    Shape: {item.shape}, Dtype: {item.dtype}')
                if item.size == 1:
                    try:
                        ref = item[0, 0]
                        if isinstance(ref, h5py.h5r.Reference):
                            actual_data = f[ref]
                            print(f'    -> Reference to: shape={actual_data.shape}, dtype={actual_data.dtype}')
                    except:
                        pass
