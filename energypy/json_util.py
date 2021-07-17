import json
from pathlib import Path




def save(data, file):
    file = str(file)
    with open(file, 'w') as fi:
        json.dump(data, fi, cls=Encoder)


def load(fi):
    fi = Path.cwd() / fi
    return json.loads(fi.read_text())

class Encoder(json.JSONEncoder):
    def default(self, arg):
        if isinstance(arg, Path):
            return str(arg)
        return arg
