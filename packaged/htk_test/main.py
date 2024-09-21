import os
import shutil
import sys
import subprocess
from service import Service
import io
import zipfile

from requests_toolbelt import MultipartEncoder
import flask

def setupEnv(picklists, **kwargs):
    root_dir = "/app/symbiosis"
    filenames = {
        "mlf": "mlf/labels.mlf_tri_internal",
        "ext": "ext/result.mlf_letter0",
        "dict": "dict/dict_letter2letter_ai_general",
        "models": "models/hmm0.19/newMacros",
        "commands": "commands/commands_letter_isolated_ai_general"
    }
    for i in filenames:
        if not os.path.exists(os.path.join(root_dir, i)):
            os.makedirs(os.path.join(root_dir, i))
        with open(os.path.join(root_dir, filenames[i]), "wb") as f:
            f.write(kwargs[i].read())
    print("done")
    return []

def runHTK(picklist):
    if True:
        if not os.path.exists("/app/symbiosis/testsets"):
            os.mkdir("/app/symbiosis/testsets")
        if not os.path.exists("/app/symbiosis/ext/data"):
            os.mkdir("/app/symbiosis/ext/data")
        with open(f'/app/symbiosis/testsets/testing-extfile', 'w') as f:
            f.write(f'./ext/data/picklist_{picklist}.ext')
        # os.system(f"cd /app/symbiosis/ && echo /tmp/ext/data/picklist_{i}.ext >> testsets/testing-extfile-{i}")
        #os.system(f"cd /app/symbiosis/ && ./third-party/gt2k/utils/prepare all-data/picklist_{i} 1 ext/data/picklist_{i}.ext")
        #os.system(f"cd /app/symbiosis/ && HParse grammar/grammar_letter_isolated_ai_general-{i} word.lattice-{i}")
        #os.system(f"cd /app/symbiosis/ && HVite -a -b sil -p 0 -t 0 -s 0 -A -T 1 -H /tmp/models/hmm0.19/newMacros -w /tmp/word.lattice-{i} -S testsets/testing-extfile-{i} -I /tmp/mlf/labels.mlf_tri_internal -i /tmp/ext/result.mlf_letter0 /tmp/dict/dict_letter2letter_ai_general /tmp/commands/commands_letter_isolated_ai_general; mv ext/result.mlf_letter0 results-{i}")
        # os.system(f"cd /app/symbiosis/ && rm word.lattice-{i}")
        os.system(f"cd /app/symbiosis/ && ./third-party/gt2k/utils/prepare /shared/data/picklist_{picklist} 1 ./ext/data/picklist_{picklist}.ext")
        os.system(f"cd /app/symbiosis/ && HParse ./grammar/grammar_letter_isolated_ai_general-{picklist} ./word.lattice-{picklist}")
        os.system(f"cd /app/symbiosis/ && HVite -a -b sil -p 0 -t 0 -s 0 -A -T 1 -H ./models/hmm0.19/newMacros -w ./word.lattice-{picklist} -S ./testsets/testing-extfile -I ./mlf/labels.mlf_tri_internal -i ./ext/result.mlf_letter0 ./dict/dict_letter2letter_ai_general ./commands/commands_letter_isolated_ai_general; mv ./ext/result.mlf_letter0 results-{picklist}")
        os.system(f"cd /app/symbiosis/ && rm ./word.lattice-{picklist}")
    

def run(id, picklist, mlf, ext, dict, models, commands):
    setupEnv(picklist, mlf=mlf, ext=ext, dict=dict, models=models, commands=commands)
    runHTK(picklist)

    ret = MultipartEncoder(
        fields = {
            'id': ('id', id.encode(), 'text/plain', {'Content-Id': 'id'}),
            'result': ('result', open(f"/app/symbiosis/results-{picklist}", 'rb'), 'text/plain', {'Content-Id': 'result'})
        }
    )
    os.system(f"rm /app/symbiosis/results-{picklist}")

    return flask.Response(
        ret.to_string(),
        mimetype=ret.content_type
    )


service = Service(
    "htk_test",
    run,
    {
        'form': ['id', 'picklist'],  
        'files': ['mlf', 'ext', 'dict', 'models', 'commands']
    }
).create_service(init_cors=True)

if __name__ == '__main__':
    service.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=os.environ.get("PORT", 5000)
    )
