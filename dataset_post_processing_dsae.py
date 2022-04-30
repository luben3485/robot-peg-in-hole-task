from ruamel import yaml
import os
from tqdm import tqdm
import numpy as np
data_root = '/tmp2/r09944001/data/pdc/logs_proto'
data_name = 'fine_insertion_square_7x12x12_2022-04-30-notilt'
anno_data = data_name + '/processed'
im_data = data_name + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
    data = yaml.load(f_r)
print(type(data))
print(len(data))

t_xyz = []

for key,value in tqdm(data.items()):
    delta_xyz = data[key]['delta_translation']
    t_xyz.append(delta_xyz)

t_xyz = np.array(t_xyz)
print('min:', np.min(t_xyz, 0))
print('max:', np.max(t_xyz, 0))
print('mean:', np.mean(t_xyz, 0))
print('median:', np.median(t_xyz, 0))

''' for track
new_data = {}
for key,value in tqdm(data.items()):
    length = len(value)
    if length > 10:
        new_data[key] = value

with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'w') as f_w:
    yaml.dump(new_data, f_w, Dumper=yaml.RoundTripDumper)
'''
