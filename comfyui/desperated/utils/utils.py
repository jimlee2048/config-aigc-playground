import torch
import os


def is_dir_empty(path):
    return len(os.listdir(path)) == 0


def check_gpu():
    if torch.cuda.is_available():
        print(f'device name [0]:', torch.cuda.get_device_name(0))
        print('🟢 CUDA is available')
        return 'cuda'
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print('🟢 MPS is available')
            return 'mps'
    else:
        return False
