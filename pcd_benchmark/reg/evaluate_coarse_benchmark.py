
score = []
threshold = 2
filename = 'benchmark_1228_reg_e_80_vision_eye_concat.txt'
with open(filename, 'r') as f:
    for line in f:
        degree = float(line)
        if degree < threshold:
            score.append(1)
        else:
            score.append(0)
print('Score: ' + str(sum(score) / len(score) * 100) + '%')
