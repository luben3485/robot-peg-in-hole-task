#!/bin/sh
python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_square_7x12x12_squarehole_2022-06-20-crop-noiseaug3-rotscale --offset kpts --aug_data --crop_pcd
python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_square_7x13x13_squarehole_2022-06-20-crop-noiseaug3-rotscale --offset kpts --aug_data --crop_pcd
python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_rectangle_7x9x12_squarehole_2022-06-20-crop-noiseaug3-rotscale --offset kpts --aug_data --crop_pcd
python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_rectangle_7x10x13_squarehole_2022-06-20-crop-noiseaug3-rotscale --offset kpts --aug_data --crop_pcd

#python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_square_7x12x12_squarehole_2022-06-20-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data
#python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_square_7x13x13_squarehole_2022-06-20-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data
#python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_rectangle_7x9x12_squarehole_2022-06-20-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data
#python depth2pcd.py --folder_path 2022-06-20-fine-squarehole/fine_insertion_rectangle_7x10x13_squarehole_2022-06-20-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data

#python depth2pcd.py --folder_path 2022-06-03-fine/fine_insertion_square_7x12x12_2022-06-03-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data
#python depth2pcd.py --folder_path 2022-06-03-fine/fine_insertion_square_7x13x13_2022-06-03-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data
#python depth2pcd.py --folder_path 2022-06-03-fine/fine_insertion_rectangle_7x10x13_2022-06-03-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data
#python depth2pcd.py --folder_path 2022-06-03-fine/fine_insertion_rectangle_7x9x12_2022-06-03-crop-noiseaug3-noscale --offset kpts --crop_pcd --aug_data

#python depth2pcd.py --folder_path coarse_insertion_square_7x12x12_2022-05-24-aug --offset kpts --aug_data
#python 2depth2pcd_seg_heatmap_kpt_dir.py --folder_path fine_insertion_square_2022-04-19 --offset kpts
#python 2depth2pcd_seg_heatmap_kpt_dir.py --folder_path fine_insertion_square_2022-04-14-crop --offset kpts --crop_pcd
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/coarse_insertion_circle_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/coarse_insertion_square_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/coarse_insertion_small_square_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/coarse_insertion_rectangle_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/coarse_insertion_triangle_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/fine_insertion_circle_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/fine_insertion_square_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/fine_insertion_small_square_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/fine_insertion_rectangle_2022-02-25
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-25/fine_insertion_triangle_2022-02-25
