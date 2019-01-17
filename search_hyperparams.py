
import os
import argparse
import sys
from subprocess import check_call

PYTHON = sys.executable

parser = argparse.ArgumentParser()


def launch_trainning_job(parent_dir, data_dir, job_name, params):

    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    cmd = "{python} main.py --model_dir={model_dir} --data_dir={data_dir}".format(python=PYTHON,
                                                                                  model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd)


if __name__ == "__main__":


    for learning_rate in 
