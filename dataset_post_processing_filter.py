from ruamel import yaml
import os
data_root = '/tmp2/r09944001/data/pdc/logs_proto'
date = '2021-11-13'
anno_data = 'insertion_xyzrot_eye_' + date + '/processed'
im_data = 'insertion_xyzrot_eye_' + date + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)
class_count = 150

with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
    data = yaml.load(f_r)

filtered_data = {}
y_cls_count = {}
z_cls_count = {}
for key, value in data.items():
    
    r_euler = data[key]['r_euler']
    z = r_euler[0]
    y = r_euler[1]
    x = r_euler[2]

    if z < -60 or z > 60:
        continue
    if y < -60 or y > 60:
        continue
    if x < -25 or x > 25:
        continue
    
    # 5 13 13
    
    z_cls = max(0, min(int(round(z,-1)/10)+6, 12))        
    y_cls = max(0, min(int(round(y,-1)/10)+6, 12))        
    x_cls = max(0, min(int(round(x,-1)/10)+2, 4))        
    
    # 5 25 25   
    '''
    z_cls = max(0, min(int(round(z,-1)/5)+12, 24))        
    y_cls = max(0, min(int(round(y,-1)/5)+12, 24))        
    x_cls = max(0, min(int(round(x,-1)/10)+2, 4))        
    '''
    cls = [z_cls, y_cls, x_cls]
    if y_cls in y_cls_count:
        y_cls_count[y_cls] += 1
        if y_cls_count[y_cls] > class_count:
            continue
    else:
        y_cls_count[y_cls] = 1
    
    if z_cls in z_cls_count:
        z_cls_count[z_cls] += 1
        if z_cls_count[z_cls] > class_count:
            continue
    else:
        z_cls_count[z_cls] = 1
    
    filtered_data[key] = value
    filtered_data[key]['cls'] = cls

print(len(data))
print(len(filtered_data))

with open(os.path.join(anno_data_path, 'peg_in_hole_filter_' + str(class_count) + '.yaml'), 'w') as f_w:
    yaml.dump(filtered_data, f_w, Dumper=yaml.RoundTripDumper)
