import os

def is_dir_empty(path):
    return len(os.listdir(path)) == 0