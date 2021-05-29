import json
from pathlib import Path


def save(data, file):
    file = str(file)
    with open(file, 'w') as fi:
        json.dump(data, fi)


def load(fi):
    fi = Path.cwd() / fi
    return json.loads(fi.read_text())
