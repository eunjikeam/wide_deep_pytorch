# coding: utf-8
# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Utils Functions """

import os
import json
import random
import logging

import numpy as np
import torch


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(fomatter)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def save(parms):
    ver = parms.version
    folders = ['conf', 'model', 'log', 'preprocess']
    path_list = []
    for folder in folders:
        path = os.path.join('model', ver, folder)
        path_list.append(path)
        os.makedirs(path)
    return path_list

def json_load(path):
    """load for conf json"""
    with open(path, 'r') as r:
        json_file = json.load(r)
    return dotdict(json_file)

def json_save(path, conf):
    """save conf json"""
    with open(path, 'w') as w:
        json.dump(conf, w, indent=4)
