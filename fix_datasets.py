# %%
# fix_datasets.py
# Taken from https://github.com/zju3dv/LoFTR/issues/276#issuecomment-1600921374

# %%
import numpy as np
from numpy import load
import os
import shutil

def fix_scene_info(scene_info_dir):
    scene_info_dir_no_sfm = f'{scene_info_dir}_no_sfm/'
    if os.path.exists(scene_info_dir_no_sfm):
        shutil.rmtree(scene_info_dir_no_sfm)
    os.mkdir(scene_info_dir_no_sfm)

    for filename in os.listdir(scene_info_dir):
        f_npz = os.path.join(scene_info_dir,filename)
        scene_info = load(f_npz, allow_pickle=True)

        for i, image_path in enumerate(scene_info['image_paths']):
            if image_path is not None:
                if 'Undistorted_SfM' in image_path:
                    ext = image_path.split('.')[-1]
                    depth_path = scene_info['depth_paths'][i]
                    scene_info['image_paths'][i] = depth_path.replace('depths', 'imgs').replace('h5', ext)
        
        scene_info['pair_infos'] = np.asarray(scene_info['pair_infos'], dtype=object)
        new_file = os.path.join(scene_info_dir_no_sfm, filename)
        np.savez(new_file, **scene_info)
        print("Saved to ", new_file)

fix_scene_info('data/megadepth/index/scene_info_0.1_0.7')
fix_scene_info('data/megadepth/index/scene_info_val_1500')
