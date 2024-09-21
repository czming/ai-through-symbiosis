import os
import sys
import subprocess
from service import Service

def run_htk_on_dirs(_dir, num_folds, split_ratio=0.7):
    try:        
        x = subprocess.Popen(f"/bin/bash -c \"cd /app/symbiosis && cp -r all-data-1-40 all-data && cp -r all-labels-1-40 all-labels && ./scripts/n_fold.sh -s {split_ratio} -d all-data/ -n {num_folds}\"", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # x = subprocess.Popen(["/bin/bash", "-c", f"\"cd /app/symbiosis && ./scripts/n_fold.sh -s {split_ratio} -d all-data/ -n {num_folds}\""], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # x = subprocess.check_output(["/bin/bash", "./scripts/n_fold.sh", "-s", f"{split_ratio}", "-d", f"all_data/", "-n", f"{num_folds}"])
    except subprocess.CalledProcessError as e:
        print(e.output)
        x = -1
    return x.stdout.readlines() + x.stderr.readlines()

service = Service(
    "htk",
    lambda id, _dir, num_folds, split_ratio: str((id, run_htk_on_dirs(_dir, num_folds, split_ratio))),
    {
        'form': ['id', '_dir', 'num_folds', 'split_ratio'],
    }
).create_service(init_cors=True)

if __name__ == '__main__':
    service.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=os.environ.get("PORT", 5000)
    )
