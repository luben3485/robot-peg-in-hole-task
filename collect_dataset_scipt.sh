#!/bin/sh
#python collect_rollout_data_for_kovis.py --hole square_7x12x12 --date 2022-05-12-tilt --iter 300 --data_type test --tilt
#python collect_rollout_data_for_kovis.py --hole square_7x12x12 --date 2022-05-12-tilt --iter 7000 --data_type train --tilt
python collect_coarse_insertion_data_nomove.py --hole square_7x12x12 --date 2022-05-15 --iter 20000
python collect_fine_insertion_data.py --hole square_7x12x12 --date 2022-05-15 --iter 20000
#python collect_fine_insertion_data.py --hole circle --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole rectangle --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole triangle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole square --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole small_square --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole circle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole rectangle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole triangle --date 2022-02-26-test --iter 30
