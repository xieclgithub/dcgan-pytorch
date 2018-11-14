import json
import itertools
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', default='workdir', help='working dir')
parser.add_argument('--category', default='test', help='task for specify category')

args = parser.parse_args()
print(args)
try:
    shutil.rmtree(args.workdir)
except FileNotFoundError:
    pass
try:
    os.makedirs(args.workdir)
except OSError:
    pass

'''
[task1, task2, task3]
task1 = {attribute1: value, attribute2: value, ...}
'''
def category_test():
    configs = {
        # 'cifar10', 'lsun', 'imagenet', 'folder', 'fake'
        "dataset":    ['cifar10'],
        "batch_size": [128],
        "lr":         [0.0002, 0.0003],
        "workers":    [8],
        "image_size": [64],
        "nz":         [100],
        "ngf":        [64],
        "ndf":        [64],
        "epochs":     [500],
        "beta1":      [0.5, 0.9],
        "netG":       [''],
        "netD":       [''],
        "nrow":       [16],
        "outf":       ['sample'],
        "manual_seed":[80]
    }
    key_list = []
    value_list = []
    for config in configs:
        key_list.append(config)
        value_list.append(configs[config])
    task_all = list(itertools.product(*value_list)).copy()

    return [{key:value for key, value in zip(key_list, task)} for task in task_all]


def issue_task(tasks):
    '''config: a list for the tasks, a tasks is a dictionary'''
    for i, task in enumerate(tasks):
        task_path = os.path.join(args.workdir, str(i))
        try:
            os.mkdir(task_path)
        except OSError:
            pass
        task_file = os.path.join(task_path, "task.json")
        with open(task_file, "w") as f:
            json.dump(task, f, indent=4, separators=(', ', ': '))


if __name__ == "__main__":
    if args.category == 'test':
        tasks = category_test()
    else:
        raise ValueError("Unknown category %s" % args.category)
    issue_task(tasks)


