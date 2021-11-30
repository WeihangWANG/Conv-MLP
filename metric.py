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
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, fpr_target)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)

            # TPR = tp / (tp*1.0 + fn*1.0)

        fpr[threshold_idx] = FPR
        # print("fpr=%s  @thr=%s"%(FPR, threshold))

    # print(np.max(fpr))

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds)
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    # print("thres=", threshold)

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
    # print("prob size", prob.size())
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    # print("top size", top.size())
    # print(top)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    # print(correct)
    correct = correct.data.cpu().numpy()
    # prob = prob.data.cpu().numpy()
    # correct = np.mean(correct)
    return correct, prob


def do_valid_test( net, test_loader, criterion ):
    valid_num  = 0
    num = 1
    losses   = []
    corrects = []
    probs = []
    probs_pos = []
    probs_neg = []
    pos_his = []
    neg_his = []
    labels = []
    visual = []
    gt = []
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    for i, (input, input_lr, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()

        input = input.cuda()
        input_lr = input_lr.cuda()

        truth = truth.cuda()
        with torch.no_grad():

            # res, dep_res = net(input, input_lr)
            res = net(input)
            # logit_sum = res + dep_res * 0.5
            logit_sum = res
            # print("logit_sum:", logit_sum)

            truth = truth.view(res.shape[0])

            loss = criterion(logit_sum, truth, False)
            correct, prob = metric(logit_sum, truth)

            # if i % 50 == 1:
            #     tsne = TSNE(n_components=2)
            #     low_fea = fea.clone()
            #     low_fea = low_fea.cpu().data.numpy()
            #     low_fea = np.array(low_fea, dtype=np.float64)
            #     visual.append(low_fea)
            #     gt.append(truth.data.cpu().numpy())

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
    
#     visual = np.concatenate(visual)
#     gt = np.concatenate(gt)
#
#     visual = tsne.fit_transform(visual)
#     # X_, Y_, Z_ = visual[:, 0], visual[:, 1], visual[:, 2]
#     X_, Y_ = visual[:, 0], visual[:, 1]
#     plt.xlim(X_.min(), X_.max())
#     plt.ylim(Y_.min(), Y_.max())
#     # plt.ylim(Z_.min(), Z_.max())
#
#     # ax = plt.axes(projection='3d')
#     # for x,y,z,g in zip(X_,Y_,Z_, gt):
#     for x,y,g in zip(X_,Y_,gt):
#         if g==0:
#             clr = 'blue'
#         else:
#             clr = 'red'
#         plt.scatter(x,y,color=clr)
#         # ax.scatter3D(x,y,z,color=clr)
# #
#     # plt.title('Visualization TSNE')
#     plt.xticks([])
#     plt.yticks([])
#     # plt.savefig("./pic/tsne_2d_4@2_5_0803.png")
#     plt.savefig("./pic/4@2_50.png")


    acer,apcer,npcer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([loss, acer, correct_, apcer, npcer])

    return valid_loss,[probs[:, 1], labels]

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



