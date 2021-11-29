
score = []
threshold = 20
with open('benchmark_1119_no_kpt_reg_cp_160_d_00_60_fix.txt', 'r') as f:
    for line in f:
        degree = float(line)
        if degree < threshold:
            score.append(1)
        else:
            score.append(0)
print('Score: ' + str(sum(score) / len(score) * 100) + '%')
