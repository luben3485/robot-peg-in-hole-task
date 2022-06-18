#!/bin/sh
### DSAE
#python collect_rollout_data_for_dsae.py --date 2022-06-16-notilt --iter 75 --data_type test
#python collect_rollout_data_for_dsae.py --date 2022-06-16-notilt --iter 5000 --data_type train

### KOVIS
### round hole
#python collect_rollout_data_for_kovis.py --date 2022-06-01-tilt --iter 1750 --data_type train --tilt
#python collect_rollout_data_for_kovis.py --date 2022-06-01-tilt --iter 75 --data_type test --tilt
### square hole
python collect_rollout_data_for_kovis.py --date 2022-06-16-notilt-yaw --iter 35 --data_type test --yaw
python collect_rollout_data_for_kovis.py --date 2022-06-16-notilt-yaw --iter 1000 --data_type train --yaw
python collect_rollout_data_for_kovis.py --date 2022-06-16-tilt-yaw --iter 35 --data_type test --yaw --tilt
python collect_rollout_data_for_kovis.py --date 2022-06-16-tilt-yaw --iter 1000 --data_type train --yaw --tilt


### ours
#python collect_coarse_insertion_data_nomove.py --hole square_7x12x12_squarehole --date 2022-06-16 --iter 10000
#python collect_coarse_insertion_data_nomove.py --hole square_7x13x13 --date 2022-06-03 --iter 5000
#python collect_coarse_insertion_data_nomove.py --hole rectangle_7x9x12 --date 2022-06-03 --iter 5000
#python collect_coarse_insertion_data_nomove.py --hole rectangle_7x10x13 --date 2022-06-03 --iter 5000
#python collect_fine_insertion_data.py --hole square_7x12x12_squarehole --date 2022-06-16 --iter 10000
#python collect_fine_insertion_data.py --hole square_7x13x13 --date 2022-06-03 --iter 5000
#python collect_fine_insertion_data.py --hole rectangle_7x9x12 --date 2022-06-03 --iter 5000
#python collect_fine_insertion_data.py --hole rectangle_7x10x13 --date 2022-06-03 --iter 5000
#python collect_fine_insertion_data.py --hole circle --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole rectangle --date 2022-02-26-test --iter 30
#python collect_fine_insertion_data.py --hole triangle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole square --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole small_square --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole circle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole rectangle --date 2022-02-26-test --iter 30
#python collect_coarse_insertion_data.py --hole triangle --date 2022-02-26-test --iter 30
