import numpy as np
import os
import yaml
import tqdm
def main():

    data_root = '/tmp2/r09944001/data/pdc/logs_proto/insertion_xyzrot_eye_close_2022-01-12/processed'
    data_small = {}
    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    #for key, value in data.items():
    for key, value in tqdm.tqdm(data.items()):
        if key <= 10:
            data_small[key] = data[key]
    with open(os.path.join(data_root, 'peg_in_hole_small.yaml'), 'w') as f_w:
        yaml.dump(data_small, f_w)

if __name__ == '__main__':
    main()
