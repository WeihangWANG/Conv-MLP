import os
import random
from utils import *

# DATA_ROOT = '../CASIA-CeFA-Challenge'
DATA_ROOT = '../WMCA'

TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Testing/'
RESIZE_SIZE = 112
# RESIZE_SIZE = 120

def load_train_list():
    print("Loading train data ...")
    list = []
    list_fold = []
    # f = open(DATA_ROOT + '/4@1_train_3_ft.txt')  # CeFA
    f = open(DATA_ROOT + '/replay_train.txt')  # WMCA
    lines = f.readlines()

    for line in lines:
        list_fold = []
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list():
    print("Loading val data ...")
    list = []
    # f = open(DATA_ROOT + '/4@1_test_3_rect.txt')  # CeFA
    f = open(DATA_ROOT + '/replay_test.txt')  # WMCA
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print("# pos : ",len(pos_list))
    print("# neg : ", len(neg_list))
    return [pos_list,neg_list]




