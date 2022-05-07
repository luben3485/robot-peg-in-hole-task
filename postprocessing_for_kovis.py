from ruamel import yaml
import os
import cv2
import time
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='fine_insertion_square_7x12x12_2022-05-04-notilt-test', help='folder path')
    return parser.parse_args()

def convert_to_kovis(data, data_root, data_type):
    image_folder_path = os.path.join(data_root, 'images')
    seg_folder_path = [os.path.join(data_root, data_type, 'left/segme'), os.path.join(data_root, data_type, 'right/segme')]
    color_folder_path = [os.path.join(data_root, data_type, 'left/color'), os.path.join(data_root, data_type, 'right/color')]
    depth_folder_path = [os.path.join(data_root, data_type, 'left/depth'), os.path.join(data_root, data_type, 'right/depth')]

    # create folder
    for i in range(2):
        if not os.path.exists(seg_folder_path[i]):
            os.makedirs(seg_folder_path[i])
        if not os.path.exists(color_folder_path[i]):
            os.makedirs(color_folder_path[i])
        if not os.path.exists(depth_folder_path[i]):
            os.makedirs(depth_folder_path[i])

    gt = []
    number = 0
    for key, value in tqdm(data.items()):
        rgb_name_list = data[key]['rgb_image_filename']
        depth_name_list = data[key]['depth_image_filename']
        seg_peg_name_list = data[key]['seg_peg_image_filename']
        seg_hole_name_list = data[key]['seg_hole_image_filename']
        delta_translation = data[key]['delta_translation']
        length = 2
        assert (len(rgb_name_list)==length and len(depth_name_list)==length and len(seg_peg_name_list)==length and len(seg_hole_name_list)==length)

        for idx in range(len(rgb_name_list)):
            file_name = str(number).zfill(6) + '.png'
            rgb = cv2.imread(os.path.join(image_folder_path, rgb_name_list[idx]))
            seg_peg = cv2.imread(os.path.join(image_folder_path, seg_peg_name_list[idx]), cv2.IMREAD_GRAYSCALE)
            seg_hole = cv2.imread(os.path.join(image_folder_path, seg_hole_name_list[idx]), cv2.IMREAD_GRAYSCALE)
            depth = cv2.imread(os.path.join(image_folder_path, depth_name_list[idx]), cv2.IMREAD_ANYDEPTH)
            # for rgb
            #rgb = cv2.resize(rgb, (128, 128))
            cv2.imwrite(os.path.join(color_folder_path[idx], file_name), rgb)

            # for seg
            #seg_peg = cv2.resize(seg_peg, (128, 128))
            #seg_hole = cv2.resize(seg_hole, (128, 128))
            y_idx, x_idx = np.nonzero(seg_hole)
            seg = np.zeros((128, 128))
            seg[y_idx, x_idx] = 2
            y_idx, x_idx = np.nonzero(seg_peg)
            seg[y_idx, x_idx] = 1
            cv2.imwrite(os.path.join(seg_folder_path[idx], file_name), seg)
            # for depth (set 255 for background)

            y_idx, x_idx = np.where(seg == 0)
            depth[y_idx, x_idx] = 255 # Depth image's data type is uint8, and its maximum value is 255. Set background to 255
            cv2.imwrite(os.path.join(depth_folder_path[idx], file_name), depth)

        # motion
        unit_delta_translation = delta_translation / np.linalg.norm(delta_translation)
        speed = np.linalg.norm(delta_translation) / 0.0049
        gt.append([float(x) for x in unit_delta_translation.tolist()] + [0, 0, 0, float(speed)])
        number += 1
    with open(os.path.join(data_root, data_type, 'gt.yaml'), 'w') as f_w:
        yaml.dump(gt, f_w, Dumper=yaml.RoundTripDumper)

def main(args):
    data_root = os.path.join('/tmp2/r09944001/data/pdc/logs_proto', args.folder_path, 'processed')
    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    s = pd.Series(data)
    training_data, test_data = [i.to_dict() for i in train_test_split(s, train_size=0.8)]

    convert_to_kovis(training_data, data_root, 'train')
    convert_to_kovis(test_data, data_root, 'test')

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time) / 3600))
