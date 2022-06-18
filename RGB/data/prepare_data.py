import os
import random
from utils import *

# The path of OULU dataset
DATA_ROOT = '../../oulu-npu'
VAL_ROOT = '../../oulu-npu/Test_files/Test_files/'

TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Testing/'
RESIZE_SIZE = 112
# RESIZE_SIZE = 120

def load_train_list():
    print("Loading train data ...")
    list = []
    list_fold = []

    f = open(DATA_ROOT + '/Protocols/Protocols/Protocol_4/Train_1_ft.txt')  # oulu
    lines = f.readlines()

    for line in lines:
        list_fold = []
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list():
    print("Loading val data ...")
    list = []

    f = open(DATA_ROOT + '/Protocols/Protocols/Protocol_4/Test_1.txt')  # oulu
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(',')
        list.append(line)
    return list

def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[1]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print("# pos : ",len(pos_list))
    print("# neg : ", len(neg_list))
    return [pos_list,neg_list]




