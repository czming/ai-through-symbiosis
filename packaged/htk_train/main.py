import os
import shutil
import sys
import subprocess
from service import Service

def setupEnv(picklists):
    files = {
        i: {
            'data': os.path.join("/shared/data", f"picklist_{i}"),
            'label': os.path.join("/shared/label", f"picklist_{i}.lab")
        }
        for i in picklists
        if os.path.isfile(
            os.path.join("/shared/data", f"picklist_{i}")
        ) and os.path.isfile(
            os.path.join("/shared/label", f"picklist_{i}.lab")
        )
    }
    ignored = list(set(picklists) - set(files.keys()))

    if ignored == set(picklists):
        raise Exception("All picklists ignored")

    with open("/app/symbiosis/trainsets/training-extfiles0", "w") as f:
        f.write("\n".join([os.path.basename(files[i]['data']) for i in files]))
    print("trainset\n", "\n".join([os.path.basename(files[i]['data']) for i in files]))

    for i in files:
        shutil.copy(files[i]["data"], os.path.join("/app/symbiosis/data", os.path.basename(files[i]["data"])))
        shutil.copy(files[i]["label"], os.path.join("/app/symbiosis/label", os.path.basename(files[i]["label"])))

    return ignored

def runHTK():
    os.system("cd /app/symbiosis && ./scripts/prepare_files.sh")
    os.system("cd /app/symbiosis && ./scripts/train_parallel.sh ./scripts/options.sh")

def run(picklists):
    ignored = setupEnv(picklists)
    runHTK()
    return ignored


service = Service(
    "htk",
    lambda id, picklists: str((id, run([int(i) for i in picklists.split(",")]))),
    {
        'form': ['id', 'picklists'],   
    }
).create_service(init_cors=True)

if __name__ == '__main__':
    service.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=os.environ.get("PORT", 5000)
    )
