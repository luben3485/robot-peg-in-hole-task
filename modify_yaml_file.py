from ruamel import yaml
import os
from tqdm import tqdm
data_root = '/home/luben/data/pdc/logs_proto'
data_folder = 'fine_insertion_square_7x12x12_2022-05-02-notilt'
anno_data = data_folder + '/processed'
im_data = data_folder + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
    data = yaml.load(f_r)
print(type(data))
print(len(data))
new_data = {}
for key, value in tqdm(data.items()):
    # do something here
    if key < 10:
        new_data[key] = value
with open(os.path.join(anno_data_path, 'peg_in_hole_small.yaml'), 'w') as f_w:
    yaml.dump(new_data, f_w, Dumper=yaml.RoundTripDumper)
