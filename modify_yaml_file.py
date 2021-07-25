from ruamel import yaml
import os
data_root = '/home/luben/data/pdc/logs_proto'
date = '2021-07-19'
anno_data = 'insertion_xyzrot_eye_' + date + '/processed'
im_data = 'insertion_xyzrot_eye_' + date + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
    data = yaml.load(f_r)
print(type(data))
print(len(data))
for i in range(len(data)):
    data[i]['bbox_bottom_right_xy'] = [255,255]
    data[i]['bbox_top_left_xy'] = [0,0]

with open(os.path.join(anno_data_path, 'peg_in_hole_test.yaml'), 'w') as f_w:
    yaml.dump(data, f_w, Dumper=yaml.RoundTripDumper)
