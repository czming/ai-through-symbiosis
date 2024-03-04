import subprocess
from service import Service
def run_htk_on_dirs(_dir, num_folds, split_ratio=0.7):
    return subprocess.check_output(["/bin/bash", "./scripts/n_folds.sh", "-s", f"{split_ratio}", "-d", f"{_dir}/data", "-n", f"{num_folds}"])

service = Service(
    "htk",
    lambda id, _dir, num_folds, split_ratio: str((id, subprocess.check_output(["/bin/bash", "./symbiosis/scripts/n_folds.sh", "-s", f"{split_ratio}", "-d", f"{_dir}/data", "-n", f"{num_folds}"]))),
    {
		'form': ['id', '_dir', 'num_folds', 'split_ratio'],
	}
).create_service(init_cors=True)

if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
