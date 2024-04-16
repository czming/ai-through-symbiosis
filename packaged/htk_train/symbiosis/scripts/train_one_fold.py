import os
import argparse
import shutil

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
    print(f"Label Directory: {label_dir}")

    data = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            data.append(filepath)

    print("Data Files:", data)

    labels = []
    for filename in os.listdir(label_dir):
        filepath = os.path.join(label_dir, filename)
        if os.path.isfile(filepath):
            labels.append(filepath)

    print("Label Files:", labels)

    # Clear all files in the directory
    for filename in os.listdir('data'):
        filepath = os.path.join('data', filename)
        if os.path.isfile(filepath):
            os.remove(filepath)

    for filename in os.listdir('label'):
        filepath = os.path.join('label', filename)
        if os.path.isfile(filepath):
            os.remove(filepath)

    # Copy files to 'data/' directory
    for file in data:
        filename = os.path.basename(file)
        destination = os.path.join('data', filename)
        shutil.copy(file, destination)

    # Copy files to 'label/' directory
    for file in labels:
        filename = os.path.basename(file)
        destination = os.path.join('label', filename)
        shutil.copy(file, destination)

    print("Data and Label files copied successfully!")

    with open("training-list", "w") as f:
        for filename in os.listdir('data'):
            filepath = os.path.join('data', filename)
            if os.path.isfile(filepath):
                name = os.path.basename(filepath).split(".")[0]
                f.write(f"{name}\n")

    os.system("cp training-list ./trainsets/training-extfiles0")
    os.system("./scripts/prepare_files.sh")
    os.system("./scripts/train_parallel.sh ./scripts/options.sh")

    print("Training completed successfully!")

    