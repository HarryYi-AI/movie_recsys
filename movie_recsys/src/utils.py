# src/utils.py
import os
import json

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    create_dir_if_not_exist('output')
    sample_data = {'message': 'Hello, world!'}
    save_json(sample_data, 'output/sample.json')