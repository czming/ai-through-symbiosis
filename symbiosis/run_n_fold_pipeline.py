import shutil
import os
import argparse
from python_on_whales import docker, DockerClient
from collections import defaultdict
from pathlib import Path

def delete_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)

def copy_folder_contents(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)
        elif os.path.isdir(source_file):
            shutil.copytree(source_file, destination_file)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data directory')
parser.add_argument('--labels', type=str, help='Path to the labels directory')
parser.add_argument('--split', type=float, help='Proportion of the data to use for training')
parser.add_argument('--n_folds', type=int, help='Number of folds')
parser.add_argument('--min_picklist', type=int, help='Minimum value of picklist')
parser.add_argument('--max_picklist', type=int, help='Maximum value of picklist')
args = parser.parse_args()

data_path = os.path.abspath(args.data)
labels_path = os.path.abspath(args.labels)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

shutil.rmtree('data', ignore_errors=True)
Path('data').mkdir(exist_ok=True)
delete_in_folder('all-data')
delete_in_folder('all-labels')
delete_in_folder('trainsets')
delete_in_folder('testsets')

for i in range(1, args.n_folds + 1):
    delete_in_folder(f'fold{i}')

copy_folder_contents(data_path, 'all-data')
copy_folder_contents(labels_path, 'all-labels')



fold_path = ''
output_path = 'data/avgFold/'

def get_avg_fold(n_folds, split, feature_vector_file, label_file):
    docker = DockerClient()
    docker.build(".", tags=["aits:latest"])

    docker.run("aits",
               ["/bin/bash", "./scripts/n_fold.sh", "-s", str(split), "-d", "all-data/", "-n", str(n_folds)],
               interactive=True, tty=True, volumes=[(".", "/tmp/")], workdir="/tmp/")

    print(docker.ps())
    for fold in range(1, n_folds + 1):
        print(fold)
        first_occurence = False
        cur_fold = f'fold{fold}'
        cur_fil_path = os.path.join(fold_path, cur_fold, fil_name)
        if len(all_lines) == 0:
            first_occurence = True
        print(cur_fil_path)
        if os.path.exists(cur_fil_path):
            with open(cur_fil_path, 'r') as f:
                data = f.readlines()
                ctr += 1
                for idx, lin in enumerate(data[2:-1]):
                    # print(lin)
                    lin = lin[:-1]
                    vals = lin.split(' ')
                    if first_occurence:
                        vals[0] = int(vals[0])
                        vals[1] = int(vals[1])
                        vals[3] = float(vals[3])
                        all_lines.append(vals)
                    else:
                        # Sometimes, results will be duplicated. Stop if this happens
                        print(all_lines, vals)
                        if len(vals) == 1 and vals[0] == '.':
                            break
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
    Path.mkdir(Path(output_path), exist_ok=True)
    dest_path = os.path.join(output_path, fil_name)
    with open(dest_path, 'w') as f:
        f.write(fin_avg_fil)

if __name__ == '__main__':
    pick_dict = {}
    for pick_id in range(args.min_picklist,args.max_picklist+1):
        fil_name = 'results-' + str(pick_id)
        fin_avg_fil = '#!MLF!#' + '\n'
        fin_avg_fil += '"/tmp/ext/data/picklist_' + str(pick_id) + '.rec"' + '\n'
        all_lines = []
        ctr = 0
        for fold in range(1,args.n_folds+1):
            print(fold)
            first_occurence = False
            cur_fold = f'fold{fold}'
            cur_fil_path = os.path.join(fold_path, cur_fold, fil_name)
            if len(all_lines) == 0:
                first_occurence = True
            print(cur_fil_path)
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
                            # Sometimes, results will be duplicated. Stop if this happens
                            print(all_lines, vals)
                            if len(vals) == 1 and vals[0] == '.':
                                break
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
        Path.mkdir(Path(output_path), exist_ok=True)
        dest_path = os.path.join(output_path, fil_name)
        with open(dest_path, 'w') as f:
            f.write(fin_avg_fil)
        # print(all_lines)