import os
from collections import defaultdict

fold_path = 'data/nFold/'
output_path = 'data/avgFold/'

if __name__ == '__main__':
    pick_dict = {}
    for pick_id in range(136,235):
        fil_name = 'results-' + str(pick_id)
        fin_avg_fil = '#!MLF!#' + '\n'
        fin_avg_fil += '"/tmp/ext/data/picklist_' + str(pick_id) + '.rec"' + '\n'
        all_lines = []
        ctr = 0
        for fold in range(1,21):
            print(fold)
            first_occurence = False
            cur_fold = 'fold'+str(fold)
            cur_fil_path = fold_path + cur_fold + '/' + fil_name
            if len(all_lines) == 0:
                first_occurence = True
            if os.path.exists(cur_fil_path):
                with open(cur_fil_path, 'r') as f:
                    data = f.readlines()
                    ctr += 1
                    for idx,lin in enumerate(data[2:-1]):
                        # print(lin)
                        lin = lin[:-1]
                        vals = lin.split(' ')
                        if first_occurence:
                            vals[0] = int(vals[0])
                            vals[1] = int(vals[1])
                            vals[3] = float(vals[3])
                            all_lines.append(vals)
                        else:
                            all_lines[idx][0] = int(all_lines[idx][0]) + int(vals[0])
                            all_lines[idx][1] = int(all_lines[idx][1]) + int(vals[1])
                            all_lines[idx][3] = float(all_lines[idx][3]) + float(vals[3])
        print(all_lines)
        for idx in range(len(all_lines)):
            all_lines[idx][0] = str(int(all_lines[idx][0] / ctr))
            all_lines[idx][1] = str(int(all_lines[idx][1] / ctr))
            all_lines[idx][3] = str(float(all_lines[idx][3] / ctr))
            fin_avg_fil += ' '.join(all_lines[idx]) + '\n'
        fin_avg_fil += '.'
        print(fin_avg_fil) 
        with open(output_path + fil_name, 'w') as f:
            f.write(fin_avg_fil)
        # print(all_lines)
                        

                        




