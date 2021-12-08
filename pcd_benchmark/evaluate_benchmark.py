
score = []
threshold = 1
filename = 'benchmark_1119_reg_e_185_d_00_60_vision_fix_front.txt'
with open(filename, 'r') as f:
    for line in f:
        degree = float(line)
        if degree < threshold:
            score.append(1)
        else:
            score.append(0)
print('Score: ' + str(sum(score) / len(score) * 100) + '%')
