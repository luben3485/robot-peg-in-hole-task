#!/bin/sh
python collect_rollout_data_for_kovis.py --hole square_7x12x12 --date 2022-05-07-notilt --iter 300 --data_type test
python collect_rollout_data_for_kovis.py --hole square_7x12x12 --date 2022-05-07-notilt --iter 7000 --data_type train
#ython collect_coarse_insertion_data_nomove.py --hole square --date 2022-03-28-test --iter 100
#python collect_fine_insertion_data.py --hole small_square --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole circle --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole rectangle --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole triangle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole square --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole small_square --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole circle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole rectangle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole triangle --date 2022-02-26-test --iter 30
