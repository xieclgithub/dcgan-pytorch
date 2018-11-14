import os
import json
from collections import namedtuple

def get_task(file_path):
    task_file = os.path.join(file_path, 'task.json')
    if not os.path.isfile(task_file):
        raise ValueError('Can not find the file %s' % task_file)
    with open(task_file, "r") as f:
        config = json.load(f, object_hook=lambda d: namedtuple('config', d.keys())(*d.values()))
        #config = json.load(f)

    return config

if __name__ == "__main__":
    config = get_task("workdir\\0")
    print(config.dataset)