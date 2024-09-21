import os
import argparse
import shutil
import subprocess

def setup_arguments():
    parser = argparse.ArgumentParser(description='Process input arguments')
    parser.add_argument('--data-dir', type=str, help='Path to the data directory')
    parser.add_argument('--label-dir', type=str, help='Path to the label directory')

    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    args = setup_arguments()

    data_dir = args.data_dir
    label_dir = args.label_dir

    print(f"Data Directory: {data_dir}")

    data = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            data.append(filepath)

    print("Data Files:", data)

    for i, filepath in enumerate(data):
        print(f"Processing file {i}: {filepath}")
        with open(f'testsets/testing-extfile-{i}', 'a') as f:
            f.write(f'/app/symbiosis/ext/data/picklist_{i}.ext\n')
        subprocess.run(['./third-party/gt2k/utils/prepare', filepath, '1', f'/app/symbiosis/ext/data/picklist_{i}.ext'])
        subprocess.run(['HParse', f'grammar/grammar_letter_isolated_ai_general-{i}', f'word.lattice-{i}'])
        subprocess.run(['HVite', '-a', '-b', 'sil', '-p', '0', '-t', '0', '-s', '0', '-A', '-T', '1', '-H', '/app/symbiosis/models/hmm0.19/newMacros', '-w', f'/app/symbiosis/word.lattice-{i}', '-S', f'testsets/testing-extfile-{i}', '-I', '/app/symbiosis/mlf/labels.mlf_tri_internal', '-i', f'/app/symbiosis/ext/result.mlf_letter0', '/app/symbiosis/dict/dict_letter2letter_ai_general', '/app/symbiosis/commands/commands_letter_isolated_ai_general'])
        subprocess.run(['mkdir', 'results'])
        subprocess.run(['mv', '/app/symbiosis/ext/result.mlf_letter0', f'results/results-{i}'])
    