import sys
import json

def read_file(name):
    with open(name, 'r') as fin:
        return json.loads(fin.read())


def write_file(name, data):
    with open(name, 'w') as fout:
        for i, line in enumerate(data):
            if i > 0:
                fout.write('\n')
            fout.write(json.dumps(line))


data = read_file(sys.argv[1])
write_file(sys.argv[2], data)
