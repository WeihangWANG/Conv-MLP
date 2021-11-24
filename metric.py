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

def do_test( net, test_loader ):
    num = 4646
    # f_out = open('../HiFi/phase1/res_test_062009.txt', 'w+')

    # net = torch.nn.DataParallel(net)
    # net = net.cuda()

    for i, (input, input_lr, input_cs, input_aug, status) in enumerate(tqdm(test_loader)):
        # for input, truth in test_loader:
        ## TTA Cropping
        b, n, c, w, h = input.size()

        input = input.cuda()
        input_lr = input_lr.cuda()
        input_cs = input_cs.cuda()
        input_aug = input_aug.cuda()
        status = status.cuda()

        with torch.no_grad():
            # src
            color_res, ir_res, clr_patch_res, ir_patch_res = net(input)
            clr_patch = torch.mean(clr_patch_res, dim=1)
            ir_patch = torch.mean(ir_patch_res, dim=1)
            # clr_patch = clr_patch_res[:, 0, :] * 0.15 + clr_patch_res[:, 2, :] * 0.15 + clr_patch_res[:, 6, :] * 0.15 + \
            #             clr_patch_res[:, 8, :] * 0.15 + clr_patch_res[:, 7, :] * 0.08 + clr_patch_res[:, 1, :] * 0.08 + \
            #             clr_patch_res[:, 3, :] * 0.08 + clr_patch_res[:, 4, :] * 0.08 + clr_patch_res[:, 5, :] * 0.08
            # ir_patch = ir_patch_res[:, 0, :] * 0.15 + ir_patch_res[:, 2, :] * 0.15 + ir_patch_res[:, 6, :] * 0.15 + \
            #            ir_patch_res[:, 8, :] * 0.15 + ir_patch_res[:, 7, :] * 0.08 + ir_patch_res[:, 1, :] * 0.08 + \
            #            ir_patch_res[:, 3, :] * 0.08 + ir_patch_res[:, 4, :] * 0.08 + ir_patch_res[:, 5, :] * 0.08
            logit = color_res * 0.25 + clr_patch * 0.25 + ir_res * 0.25 + ir_patch * 0.25

            # horizontal flip
            color_res, ir_res, clr_patch_res, ir_patch_res = net(input_lr)
            clr_patch = torch.mean(clr_patch_res, dim=1)
            ir_patch = torch.mean(ir_patch_res, dim=1)
            # clr_patch = clr_patch_res[:, 0, :] * 0.15 + clr_patch_res[:, 2, :] * 0.15 + clr_patch_res[:, 6, :] * 0.15 + \
            #             clr_patch_res[:, 8, :] * 0.15 + clr_patch_res[:, 7, :] * 0.08 + clr_patch_res[:, 1, :] * 0.08 + \
            #             clr_patch_res[:, 3, :] * 0.08 + clr_patch_res[:, 4, :] * 0.08 + clr_patch_res[:, 5, :] * 0.08
            # ir_patch = ir_patch_res[:, 0, :] * 0.15 + ir_patch_res[:, 2, :] * 0.15 + ir_patch_res[:, 6, :] * 0.15 + \
            #            ir_patch_res[:, 8, :] * 0.15 + ir_patch_res[:, 7, :] * 0.08 + ir_patch_res[:, 1, :] * 0.08 + \
            #            ir_patch_res[:, 3, :] * 0.08 + ir_patch_res[:, 4, :] * 0.08 + ir_patch_res[:, 5, :] * 0.08
            logit_lr = color_res * 0.25 + clr_patch * 0.25 + ir_res * 0.25 + ir_patch * 0.25

            # # contrast
            # color_res, ir_res, clr_patch_res, ir_patch_res = net(input_cs)
            # clr_patch = clr_patch_res[:, 0, :] * 0.15 + clr_patch_res[:, 2, :] * 0.15 + clr_patch_res[:, 6, :] * 0.15 + \
            #             clr_patch_res[:, 8, :] * 0.15 + clr_patch_res[:, 7, :] * 0.08 + clr_patch_res[:, 1, :] * 0.08 + \
            #             clr_patch_res[:, 3, :] * 0.08 + clr_patch_res[:, 4, :] * 0.08 + clr_patch_res[:, 5, :] * 0.08
            # ir_patch = ir_patch_res[:, 0, :] * 0.15 + ir_patch_res[:, 2, :] * 0.15 + ir_patch_res[:, 6, :] * 0.15 + \
            #            ir_patch_res[:, 8, :] * 0.15 + ir_patch_res[:, 7, :] * 0.08 + ir_patch_res[:, 1, :] * 0.08 + \
            #            ir_patch_res[:, 3, :] * 0.08 + ir_patch_res[:, 4, :] * 0.08 + ir_patch_res[:, 5, :] * 0.08
            # logit_cs = color_res * 0.25 + clr_patch * 0.25 + ir_res * 0.25 + ir_patch * 0.25
            #
            # # horizontal flip + contrast
            # color_res, ir_res, clr_patch_res, ir_patch_res = net(input_aug)
            # clr_patch = clr_patch_res[:, 0, :] * 0.15 + clr_patch_res[:, 2, :] * 0.15 + clr_patch_res[:, 6, :] * 0.15 + \
            #             clr_patch_res[:, 8, :] * 0.15 + clr_patch_res[:, 7, :] * 0.08 + clr_patch_res[:, 1, :] * 0.08 + \
            #             clr_patch_res[:, 3, :] * 0.08 + clr_patch_res[:, 4, :] * 0.08 + clr_patch_res[:, 5, :] * 0.08
            # ir_patch = ir_patch_res[:, 0, :] * 0.15 + ir_patch_res[:, 2, :] * 0.15 + ir_patch_res[:, 6, :] * 0.15 + \
            #            ir_patch_res[:, 8, :] * 0.15 + ir_patch_res[:, 7, :] * 0.08 + ir_patch_res[:, 1, :] * 0.08 + \
            #            ir_patch_res[:, 3, :] * 0.08 + ir_patch_res[:, 4, :] * 0.08 + ir_patch_res[:, 5, :] * 0.08
            # logit_aug = color_res * 0.25 + clr_patch * 0.25 + ir_res * 0.25 + ir_patch * 0.25

            logit_sum = logit * 0.5 + logit_lr * 0.5# + logit_cs * 0.25 + logit_aug * 0.25

            prob = F.softmax(logit_sum, 1)

            for m in range(len(input)):

                if status[m] == 0:
                    prob[m][1] = 0.001

                # f_out.write("%s.png %f\n"%(num, prob[m][1]))
                num = num + 1


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
    # f_out = open('../HiFi/phase1/res_val_062010.txt', 'w+')
    # f_err = open('res_err_0708.txt', 'w+')

    for i, (input, input_lr, truth) in enumerate(tqdm(test_loader)):
    # for input, truth in test_loader:
        ## TTA Cropping
        b,n,c,w,h = input.size()

        input = input.cuda()
        input_lr = input_lr.cuda()

        truth = truth.cuda()
        with torch.no_grad():
            # src
            # res, dep_res = net(input, input_lr)
            res = net(input)
            # logit_sum = res + dep_res * 0.5
            logit_sum = res
            # print("logit_sum:", logit_sum)
            # norm_max, max_index = res.abs().max(dim=1)
            # res = torch.div(res.t(), norm_max)
            # res = res.t()
            # mask_pos = truth.view(res.shape[0]) > 0
            # mask_neg = truth.view(res.shape[0]) == 0
            # color_p = res[mask_pos,:]
            # color_n = res[mask_neg,:]
            # print("color_p_src = ", color_p)
            # print("color_n_src = ", color_n)
            # logit_sum = res

            truth = truth.view(res.shape[0])
            # truth_tsne = truth.clone().detach()
            # truth[truth==2] = 0
            # print("truth shape", truth.shape)
            loss = criterion(logit_sum, truth, False)
            correct, prob = metric(logit_sum, truth)
            # prob_p = prob[mask_pos, :]
            # prob_n = prob[mask_neg, :]

            # if i % 50 == 1:
            #     tsne = TSNE(n_components=2)
            #     low_fea = fea.clone()
            #     # print("low fea = ", low_fea)
            #     low_fea = low_fea.cpu().data.numpy()
            #     low_fea = np.array(low_fea, dtype=np.float64)
            #     visual.append(low_fea)
            #     gt.append(truth.data.cpu().numpy())

            ### save txt
            # for m in range(len(input)):
            # #
            #     if status[m] == 0:
            # #         prob[m][1] = 0.001
            #     f_out.write("%04d.png %f\n" % (num, prob[m][1]))
                # num = num + 1


        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        # corrects.append(corrects.data.cpu().numpy())
        corrects.append(np.asarray(correct))
        probs.append(prob.data.cpu().numpy())
        # probs.append(np.asarray(correct))
        # probs_pos.append(prob_p.data.cpu().numpy())
        # probs_neg.append(prob_n.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())


    correct_cat = np.concatenate(corrects)
    # print("correct size", correct_cat.shape)
    loss    = np.concatenate(losses)
    # print("loss size", loss.shape)
    loss    = loss.mean()
    correct_ = correct_cat.mean()

    probs = np.concatenate(probs)
    # print("probs:\n", probs)
    # probs_pos = np.concatenate(probs_pos)
    # probs_neg = np.concatenate(probs_neg)
    # probs_pos = probs_pos[:,1]
    # probs_neg = probs_neg[:,1]
    # print("prob_pos : \n", probs_pos)
    # print("prob_neg : \n", probs_neg)
    # for m in range(10):
    #     q = np.sum(list(map(lambda x:x>0.1*m and x<=0.1*(m+1), probs_pos)))
    #     s = np.sum(list(map(lambda x:x>0.1*m and x<=0.1*(m+1), probs_neg)))
    #     pos_his.append(q)
    #     neg_his.append(s)
    # # print("probs size", probs.shape)
    labels = np.concatenate(labels)
    # print("pos history : \n", pos_his)
    # print("neg history : \n", neg_his)
    # print("labels size", labels.shape)
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
#         # else:
#         #     clr = 'green'
#         plt.scatter(x,y,color=clr)
#         # ax.scatter3D(x,y,z,color=clr)
# #
#     # plt.title('Visualization TSNE')
#     plt.xticks([])
#     plt.yticks([])
#     # plt.savefig("./pic/tsne_2d_4@2_5_0803.png")
#     plt.savefig("./pic/4@2_50.png")

    # tpr_1, fpr_1 = TPR_FPR( probs[:,1], labels, fpr_target = 0.01)
    # print("FPR=%s | TPR=%s "%(fpr_1, tpr_1))
    # tpr_1, fpr_1 = TPR_FPR(probs[:, 1], labels, fpr_target=0.001)
    # print("FPR=%s | TPR=%s " % (fpr_1, tpr_1))
    # tpr_1, fpr_1 = TPR_FPR(probs[:, 1], labels, fpr_target=0.0001)
    # print("FPR=%s | TPR=%s " % (fpr_1, tpr_1))

    acer,apcer,npcer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    # valid_loss = np.array([
    #     loss, acer, acc, correct, tpr_0, fnr_0, tpr_1, fnr_1, apcer, npcer
    # ])
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



