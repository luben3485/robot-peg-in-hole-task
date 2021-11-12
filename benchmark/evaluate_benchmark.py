
score = []
threshold = 10
with open('benchmark_1015_no_kpt_reg_filter_400_2_cp_200_d_05_60.txt', 'r') as f:
    for line in f:
        degree = float(line)
        if degree < threshold:
            score.append(1)
        else:
            score.append(0)
print('Score: ' + str(sum(score) / len(score) * 100) + '%')
