from ruamel import yaml
import os
import numpy as np
data_root = '/tmp2/r09944001/data/pdc/logs_proto'
data_folder = 'insertion_xyzrot_eye_close_2022-01-12'
anno_data = data_folder + '/processed'
im_data = data_folder + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v, norm
    return v / norm, norm


with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
    data = yaml.load(f_r)

min_size = 99
max_size = 0
for key, value in data.items():
    delta_xyz = data[key] ['delta_translation']
    unit_delta_xyz, step_size = normalize(delta_xyz)
    print(step_size)
    min_size = min(min_size, step_size)
    max_size = max(min_size, step_size)
print('min size', min_size)
print('max size', max_size)
    
    
'''
with open(os.path.join(anno_data_path, 'peg_in_hole_translation.yaml'), 'w') as f_w:
    yaml.dump(data_new, f_w, Dumper=yaml.RoundTripDumper)
'''