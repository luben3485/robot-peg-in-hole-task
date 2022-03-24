#!/bin/sh
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/coarse_insertion_circle_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/coarse_insertion_square_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/coarse_insertion_small_square_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/coarse_insertion_rectangle_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/coarse_insertion_triangle_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/fine_insertion_circle_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/fine_insertion_square_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/fine_insertion_small_square_2022-02-26-test
python 2depth2pcd_seg_heatmap_3kpt.py --folder_path 2022-02-26-test/fine_insertion_rectangle_2022-02-26-test
#python 2depth2pcd_seg_heatmap_kpt.py --folder_path 2022-02-26-test/fine_insertion_triangle_2022-02-26-test
