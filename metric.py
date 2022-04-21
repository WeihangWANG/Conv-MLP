import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
import numpy as np
import torch
from scipy import interpolate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)

    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def ACER(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer, apcer, npcer, tp, fp, tn,fn

def TPR_FPR( dist, actual_issame, fpr_target = 0.001):

    thresholds = np.arange(0.0, 1.0, fpr_target)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)

        fpr[threshold_idx] = FPR

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds)
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)

    apcer = fp / (tn * 1.0 + fp * 1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0

    # print(str(FPR)+' '+str(TPR))
    # return acer, apcer, npcer
    return TPR, FPR

import torch.nn.functional as F
def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    correct = correct.data.cpu().numpy()
    return correct, prob

## For Multi-modal Datasets (RGB+D+IR)
def do_valid_test( net, test_loader, criterion ):
    valid_num  = 0
    num = 1
    losses   = []
    corrects = []
    probs = []
    labels = []
    visual = []
    gt = []
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    for i, (input, input_lr, truth) in enumerate(tqdm(test_loader)):
    # for input, truth in test_loader:
        b,n,c,w,h = input.size()

        input = input.cuda()
        input_lr = input_lr.cuda()

        truth = truth.cuda()
        with torch.no_grad():
            res = net(input)
            logit_sum = res

            truth = truth.view(res.shape[0])

            loss = criterion(logit_sum, truth, False)
            correct, prob = metric(logit_sum, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    correct_cat = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct_ = correct_cat.mean()

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    acer,apcer,npcer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([loss, acer, correct_, apcer, npcer])

    return valid_loss,[probs[:, 1], labels]

## For RGB Datasets
def validate(data_loader, model):
    losses   = []
    corrects = []
    probs = []
    labels = []
    valid_num = 0
    criterion = softmax_cross_entropy_criterion
    
    model.eval()

    acer_meter = AverageMeter()
    apcer_meter = AverageMeter()
    bpcer_meter = AverageMeter()
    res_con = []
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        ### 120 is the parameter set in "data_fusion_oulu.py"
        images = images.view(120,9,3,48,48)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        with torch.no_grad():
            res = model(images)
            
            logit_mean = res.mean(dim=0)
            target = target.view(res.shape[0])[0].data.cpu().numpy()
            prob = F.softmax(logit_mean)
            pred = prob[1].data.cpu().numpy()
            res_con.append([pred,target])
            thres = 0.5
            if pred > thres:
                if target == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if target == 1:
                    fn = fn + 1
                else:
                    tn = tn + 1
            apcer = fp / (tn*1.0 + fp*1.0 + 1e-5)
            bpcer = fn / (fn*1.0 + tp*1.0 + 1e-5)
            acer = (apcer + bpcer) / 2.0

            acer_meter.update(acer)
            apcer_meter.update(apcer)
            bpcer_meter.update(bpcer)

    logger.info(f' * ACER {acer_meter.avg:.4f} APCER {apcer_meter.avg:.4f} BPCER {bpcer_meter.avg:.4f}')
    return acer_meter.avg, apcer_meter.avg, bpcer_meter.avg


def infer_test(net, test_loader):
    valid_num  = 0
    probs = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)
        input = input.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())

    probs = np.concatenate(probs)
    return probs[:, 1]



