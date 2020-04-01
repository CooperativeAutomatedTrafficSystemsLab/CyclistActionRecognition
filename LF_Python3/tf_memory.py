## @package tf_memory
#  Helper functions for memory orchestration in multi gpu systems
import os
import subprocess as sp
import tensorflow as tf
import numpy as np


## A namespcae wrapper class for using the most suitable gpus
#
class gpu_device(object):
    ## Constructor
    #  @param name:   Name for the namespace
    #  @param min_memory:   minimum available memory for sorting the gpus
    #  @param num_gpus:     number available gpus
    def __init__(self, name=None, min_memory=0, num_gpus=1):
        self.name = name
        self.min_memory = min_memory
        self.num_gpus = num_gpus

    ## Enter method for the namespace, does the masking
    def __enter__(self):
        mask_gpus(sort_gpus(min_memory=self.min_memory)[:self.num_gpus])
        if self.name:
            print(self.name + ':', 'virtual gpu device created')

    ## Exit method of the namespace
    def __exit__(self, type, value, traceback):
        if self.name:
            print(self.name + ':', 'virtual gpu device released')


## Gets list of available nvidia-gpus
#  @return: list of available memory of each gpu
def get_gpu_list():
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


## Sorts gpus by available memory
#
def sort_gpus(name=None, min_memory=0, AVAILABLE_DEVICES=None):
    '''sorts GPU's by available memory'''
    if AVAILABLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = AVAILABLE_DEVICES
    memory_free_values = get_gpu_list()
    available_gpus = [i for i, x in enumerate(memory_free_values)]
    try:
        VISIBLE = os.environ["CUDA_VISIBLE_DEVICES"]
        if VISIBLE:
            memory_free_values = [memory_free_values[i] for i in map(int, VISIBLE.split(','))]
            print(memory_free_values)
            available_gpus = [available_gpus[i] for i in map(int, VISIBLE.split(','))]
        else:
            memory_free_values = []
            available_gpus = 0
    except:
        print('No CUDA_VISIBLE_DEVICES set')
    if len(available_gpus) < 1: raise ValueError('No usable GPUs in the system')

    if not all(available_memory >= min_memory for available_memory in memory_free_values):
        if name:
            raise RuntimeError(name + ':' + '  MEMORY ALLOCATION FAILED')
        else:
            raise RuntimeError('MEMORY ALLOCATION FAILED')
    return [available_gpus[i] for i in np.argsort(memory_free_values)][::-1]


## Set a global OS variable to keep the selcted gpus visible
#  @param device_list:  list of selected gpus
def mask_gpus(device_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_list))
