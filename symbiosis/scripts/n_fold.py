import subprocess
import argparse
import random
from decimal import Decimal
import os
import glob
import shutil


def run_folds(num_folds, split, data_dir):

    data_files = os.listdir(data_dir)    
    num_training_files = round(split * len(data_files))
    if not os.path.exists("pca-training-data"): 
        os.mkdir("pca-training-data")
    for fold_num in range(0,num_folds):
        print("Fold " + str(fold_num))
        random.shuffle(data_files)
        training_files = data_files[:num_training_files]
        testing_files = data_files[num_training_files:]
        print("Training Files: " + str(len(training_files)))
        print("Testing Files: " + str(len(testing_files)))

        # False if Testing and Training share no data
        train_test_share_files = bool(set(training_files) & set(testing_files))
        if train_test_share_files:
            print("Data Leakage")
            exit

        files = glob.glob("pca-training-data/*")
        print(files)
        for f in files:
            os.remove(f)
        for training_file in training_files:
            shutil.copy(os.path.join(data_dir, training_file), "./pca-training-data")
            # shutil.copy(os.path.join("all-labels", training_file + ".lab"), "./label")


        # mv ./pca-data/* ./data

        bashCommand = "python3 PCA.py"
        print(bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

        bashCommand = "python3 apply_pca.py"
        print(bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)


        pca_files = glob.glob("pca-data/*")
        for pca_file in pca_files:
            shutil.copy(pca_file, "./data")

        with open('./trainsets/training-extfiles0', 'w') as outFile:
            for training_file in training_files:
                outFile.write("/PhraseLevel-general-36/ext/data/" + training_file + ".ext\n")

        bashCommand = "./scripts/prepare_files.sh"
        print(bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

        bashCommand = "./scripts/train_parallel.sh ./scripts/options.sh"
        print(bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

        for testing_file in testing_files:
            print(testing_file)
            num_index = testing_file.index("_")
            testing_file_num = testing_file[num_index+1:]
            bashCommand = "HParse grammar/grammar_letter_isolated_ai_general-" + testing_file_num + " word.lattice-" + testing_file_num
            print(bashCommand)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)

            bashCommand = "HVite -a -b sil -p 0 -t 0 -s 0 -A -T 1 -H /PhraseLevel-general-36/models/hmm0.19/newMacros -w /PhraseLevel-general-36/word.lattice-" + testing_file_num + " -S testsets/testing-extfile-" + testing_file_num + " -I /PhraseLevel-general-36/mlf/labels.mlf_tri_internal -i /PhraseLevel-general-36/ext/result.mlf_letter0 /PhraseLevel-general-36/dict/dict_letter2letter_ai_general /PhraseLevel-general-36/commands/commands_letter_isolated_ai_general; mv ext/result.mlf_letter0 results-" + testing_file_num
            print(bashCommand)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            
        if not os.path.exists("fold" + str(fold_num)): 
            os.mkdir("fold" + str(fold_num))
        
        result_files = glob.glob("./results-*")
        for result_file in result_files:
            shutil.move(result_file, "./fold" + str(fold_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=False, help="")
    parser.add_argument("--dir", required=False, help="")
    parser.add_argument("--n", required=False, help="")

    args = parser.parse_args()

    print()
    print("Training Split: " + args.split)
    print("Data Directory: " + args.dir)
    print("Testing Split: " + str((1-Decimal(args.split)))) 
    print()
    run_folds(int(args.n), Decimal(args.split), args.dir)