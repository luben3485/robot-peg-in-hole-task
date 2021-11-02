from ruamel import yaml
import os
data_root = '/tmp2/r09944001/data/pdc/logs_proto'
date = '2021-10-15'
anno_data = 'insertion_xyzrot_eye_' + date + '/processed'
im_data = 'insertion_xyzrot_eye_' + date + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
    data = yaml.load(f_r)
print(type(data))
print(len(data))
for i in range(len(data)):
    # do something here
    r_euler = data[i]['r_euler']
    z = r_euler[0]
    y = r_euler[1]
    x = r_euler[2]

    z_cls = max(0, min(int(round(z,-1)/10)+6, 12))        
    y_cls = max(0, min(int(round(y,-1)/10)+6, 12))        
    x_cls = max(0, min(int(round(x,-1)/10)+2, 4))        

    cls = [z_cls, y_cls, x_cls]
    data[i]['cls'] = cls
with open(os.path.join(anno_data_path, 'peg_in_hole_new.yaml'), 'w') as f_w:
    yaml.dump(data, f_w, Dumper=yaml.RoundTripDumper)
