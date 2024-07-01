# %%
# fix_datasets.py
# Taken from https://github.com/zju3dv/LoFTR/issues/276#issuecomment-1600921374

# %%
import numpy as np
from numpy import load
import os

directory = 'data/megadepth/index/scene_info_0.1_0.7'
directory2 = 'data/megadepth/index/scene_info_0.1_0.7_no_sfm/'
os.mkdir(directory2)

for filename in os.listdir(directory):
    f_npz = os.path.join(directory,filename)
    data = load(f_npz, allow_pickle = True)
    for count, image_path in enumerate(data['image_paths']):
        if image_path is not None:
            if 'Undistorted_SfM' in image_path:
                data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')
    
    data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
    new_file = directory2 + filename
    np.savez(new_file, **data)
    print("Saved to ", new_file)

# %%
directory = 'data/megadepth/index/scene_info_val_1500'
directory2 = 'data/megadepth/index/scene_info_val_1500_no_sfm/'
os.mkdir(directory2)

for filename in os.listdir(directory):
    f_npz = os.path.join(directory,filename)
    data = load(f_npz, allow_pickle = True)
    for count, image_path in enumerate(data['image_paths']):
        if image_path is not None:
            if 'Undistorted_SfM' in image_path:
                data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')
    
    data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
    new_file = directory2 + filename
    np.savez(new_file, **data)
    print("Saved to ", new_file)

# %%
