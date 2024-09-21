import os
import shutil
import sys
import subprocess
from service import Service
import io
import zipfile

from requests_toolbelt import MultipartEncoder
import flask

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

def run(id, picklists):
    ignored = setupEnv(picklists)
    runHTK()
    root_dir = "./symbiosis"
    files = {
            "mlf": "mlf/labels.mlf_tri_internal",
            "ext": "ext/result.mlf_letter0",
            "dict": "dict/dict_letter2letter_ai_general",
            "models": "models/hmm0.19/newMacros",
            "commands": "commands/commands_letter_isolated_ai_general"
        }

    #ret_data = io.BytesIO()
    #with zipfile.Zipfile(ret_data, mode="w") as zf:
    #    for i in files:
    #        filepath = os.path.join(root_dir, i)
    #        zf.write(filepath, arcname=i)
    #ret_data.seek(0)
    
    fields = {
            'id': ('id', str(id).encode(), 'text/plain', {'Content-Id': 'id'}),
            'ignored': ('ignored', ",".join(str(i) for i in ignored).encode(), 'text/plain', {'Content-Id': 'ignored'}),
            **{
                i: (os.path.basename(files[i]), open(os.path.join(root_dir, files[i]), 'rb'), 'application/octet-stream', {'Content-Id': i})
            for i in files
            }
        }
    ret = MultipartEncoder(
            fields = fields
    )
    return flask.Response(
        ret.to_string(),
        mimetype=ret.content_type
    )

service = Service(
    "htk",
    run,
    {
        'form': ['id', 'picklists'],   
    }
).create_service(init_cors=True)

if __name__ == '__main__':
    service.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=os.environ.get("PORT", 5000)
    )
