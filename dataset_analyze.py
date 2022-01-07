from ruamel import yaml
import os
data_root = '/tmp2/r09944001/data/pdc/logs_proto'
date = '2021-11-13'
anno_data = 'insertion_xyzrot_eye_' + date + '/processed'
im_data = 'insertion_xyzrot_eye_' + date + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

with open(os.path.join(anno_data_path, 'peg_in_hole_filter_150.yaml'), 'r') as f_r:
    data = yaml.load(f_r)
print(type(data))
print(len(data))
x_count = {}
y_count = {}
z_count = {}
#for i in range(len(data)):
for i, value in data.items():
    # do something here
    r_euler = data[i]['r_euler']
    x = int(r_euler[2])
    y = int(r_euler[1])
    z = int(r_euler[0])
    if x in x_count:
        x_count[x] += 1
    else:
        x_count[x] = 1
    if y in y_count:
        y_count[y] += 1
    else:
        y_count[y] = 1
    if z in z_count:
        z_count[z] += 1
    else:
        z_count[z] = 1

fx = open('x_count.txt', 'w')
print('x_count')
k_list = []
v_list = []
for k,v in x_count.items():
    k_list.append(k)
    v_list.append(v)
print('key')
fx.write("key\n")
for k in k_list:
    fx.write(str(k)+"\n")
    print(k)
fx.write("value\n")
print('value')
for v in v_list:
    fx.write(str(v)+"\n")
    print(v)
fx.close()

fy = open('y_count.txt', 'w')
print('y_count')
k_list = []
v_list = []
for k,v in y_count.items():
    k_list.append(k)
    v_list.append(v)
print('key')
fy.write("key\n")
for k in k_list:
    fy.write(str(k)+"\n")
    print(k)
fy.write("value\n")
print('value')
for v in v_list:
    fy.write(str(v)+"\n")
    print(v)
fy.close()

fz = open('z_count.txt', 'w')
print('z_count')
k_list = []
v_list = []
for k,v in z_count.items():
    k_list.append(k)
    v_list.append(v)
print('key')
fz.write("key\n")
for k in k_list:
    fz.write(str(k)+"\n")
    print(k)
fz.write("value\n")
print('value')
for v in v_list:
    fz.write(str(v)+"\n")
    print(v)
fz.close()
#with open(os.path.join(anno_data_path, 'peg_in_hole_new.yaml'), 'w') as f_w:
#    yaml.dump(data, f_w, Dumper=yaml.RoundTripDumper)
