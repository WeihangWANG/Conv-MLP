import os
import random
import glob
from utils import *

# DATA_ROOT = r'../CASIA-SURF/phase1'
DATA_ROOT = '../CASIA-CeFA-Cha'
# DATA_ROOT = '../CASIA-CeFA-Challenge'
# DATA_ROOT = '../WMCA'
# DATA_ROOT = r'../3DMAD'
# DATA_ROOT = '../HiFi/phase1'
# DATA_ROOT = r'../dm_pad'

TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Testing/'
RESIZE_SIZE = 112
# RESIZE_SIZE = 128

def load_train_list():
    print("Loading train data ...")
    list = []
    # f = open(DATA_ROOT + '/train_list_ft.txt')  # Casia-surf
    f = open(DATA_ROOT + '/4@1_train_3_ftaug.txt')  # CeFA
    # f = open(DATA_ROOT + '/grand_train_ft.txt')  # WMCA
    # f = open(DATA_ROOT + '/train_new_crop_ft.txt')  # HIFI
    # f = open(DATA_ROOT + '/label_train.txt')     # 3DMAD
    # f = open(DATA_ROOT + '/label_0124.txt')     # dm
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list():
    print("Loading val data ...")
    list = []
    # f = open(DATA_ROOT + '/test_private_list_ft.txt')  # Casia-surf
    # f = open(DATA_ROOT + '/4@1_test_3_rect.txt')  # CeFA
    f = open(DATA_ROOT + '/4@2_test_ft.txt')  # CeFA
    # f = open(DATA_ROOT + '/grand_test_ft.txt')  # WMCA
    # f = open(DATA_ROOT + '/val_new_crop_ft.txt')  # HiFi
    # f = open(DATA_ROOT + '/label_test.txt')       # 3DMAD
    # f = open(DATA_ROOT + '/label_valid_0128.txt')     # dm
    lines = f.readlines()

    for line in lines:
        fold = []
        line = line.strip().split(' ')
        for ii in glob.glob('%s/profile/*.jpg'%line[0]):
            rgb_path = ii
            dep_path = ii[:-16] + 'depth/' + ii[-9:]
            ir_path = ii[:-16] + 'ir/' + ii[-9:]
            path = tuple([rgb_path, dep_path, ir_path])
            fold.append(path)

        list.append(tuple([fold, line[1]]))
    return list

def load_test_list():
    list = []
    f = open(DATA_ROOT + '/test_new_crop_ft.txt')
    # f = open(DATA_ROOT + '/label_valid_0128.txt')
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

def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list



