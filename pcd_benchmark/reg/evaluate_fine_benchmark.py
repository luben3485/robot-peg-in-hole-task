

filename = 'benchmark_0103_0103_reg_vision_eye_concat_c2f_closeloop.txt'
insertion_succ = []
with open(filename, 'r') as f:
    for line in f:
        tmp = line.split(' ')
        degree = tmp[0]
        dist = float(tmp[1])
        if dist < 0.010:
            insertion_succ.append(1)
        else:
            insertion_succ.append(0)
        
print('Insertion success rate: ' + str(sum(insertion_succ) / len(insertion_succ) * 100) + '%')
