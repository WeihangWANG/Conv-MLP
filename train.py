import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
import sys
sys.path.append("..")
import argparse
import math
from process.data_fusion_s import *
from process.augmentation import *
# from metric import *
from metric_tsne import *
# from metric_fold import *
import torch.nn.functional as F
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet

    net = FusionNet(num_class=num_class)
    return net

def cal_theta(m):
    theta = torch.acos(m)
    theta = theta + 0.2
    m = torch.cos(theta)
    m[m<=0] = 0.1
    return m

# def moat_head_add(logit, truth, cos_m, sin_m, thres_add):
#     prob = F.softmax(logit, 1)
#     label = torch.eye(2).cuda()
#     truth = truth.view(truth.shape[0])
#     label = label.index_select(0,truth)
#     # cos_cal = cos_theta.clone().detach()
#     todo = torch.mul(cos_theta, label)
#     dont_need = cos_theta - todo
#     theta = torch.acos(todo)
#     theta = theta + 0.2
#     cos_theta_m = torch.cos(theta) + dont_need
#     cos_theta_m = cos_theta_m * 2.0
#
#     # cos_theta_2 = torch.pow(todo, 2)
#     # sin_theta_2 = 1 - cos_theta_2
#     # sin_theta = torch.sqrt(sin_theta_2)
#     # cond = todo - 0
#     # mask_sin = cond < 0
#     # sin_theta[mask_sin] = (-1) * sin_theta[mask_sin]
#     # cos_theta_m = (todo * cos_m - sin_theta * sin_m)  # cos(theta+m)
#     # # cos_theta_m = cos_theta_m
#     # th = todo - thres_add
#     # mask = th <= 0
#     # cos_theta_m[mask] = cos_theta[mask]
#     # cos_theta_m = cos_theta_m + dont_need
#     # cos_theta_m = cos_theta_m * 2
#     return cos_theta_m

# def moat_head_mns(cos_theta, cos_m, sin_m, thres_mns):
#     cos_cal = cos_theta.clone().detach()
#     cos_theta_2 = torch.pow(cos_cal, 2)
#     sin_theta_2 = 1 - cos_theta_2
#     sin_theta = torch.sqrt(sin_theta_2)
#     cond = cos_cal - 0
#     mask_sin = cond < 0
#     sin_theta[mask_sin] = sin_theta[mask_sin] * (-1)
#     cos_theta_m = (cos_theta * cos_m + sin_theta * sin_m)  # cos(theta+m)
#     cos_theta_m = cos_theta_m * 2
#     th = cos_cal - thres_mns
#     mask = th >= 0
#     cos_theta_m[mask] = cos_theta[mask] * 2.
#     return cos_theta_m

def run_train(config):
    # out_dir = './models_mixer'
    out_dir = './models_wmca'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    criterion          = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'),mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size, fold_index=config.train_fold_index)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size  = config.batch_size, drop_last = True, num_workers = 8)

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                # batch_size  = config.batch_size // 36,     # TTA 36
                                batch_size = config.batch_size,
                                drop_last   = False,
                                num_workers = 8)

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, num_class=2)
    print(net)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)
        # model_dict = net.state_dict()
        # pretrained_dict = {k: v for k, v in initial_checkpoint['state_dict'].items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # net.load_state_dict(model_dict)


    log.write('%s\n'%(type(net)))
    log.write('criterion=%s\n'%criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    train_loss   = np.zeros(6,np.float32)
    valid_loss   = np.zeros(6,np.float32)
    batch_loss   = np.zeros(6,np.float32)
    iter = 0
    i    = 0

    start = timer()
    #-----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=1e-2, momentum=0.9, weight_decay=0.005)

    # sgdr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50, 60, 70, 80], gamma=0.5, last_epoch=-1)
    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    global_min_acer = 1.0
    m = 0.5
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    thres_add = math.cos(math.pi - m)
    thres_mns = math.cos(m)
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.6f}'.format(lr))

            sum_train_loss = 0.0
            sum_train_precision = 0.0
            sum = 0


            for input, depth, truth in train_loader:
                iter = i + start_iter
                l1_reg = 0.0

                # one iteration update  -------------

                input = input.cuda()
                truth = truth.cuda()
                depth = depth.cuda()

                net.train()
                # color_res,  depth_res = net.forward(input, depth)
                color_res = net.forward(input)

                truth = truth.view(color_res.shape[0])

                # logit = color_res + depth_res
                logit = color_res

                # for para in net.parameters():
                #     l1_reg += torch.sum(torch.abs(para))
                # print("truth : \n", truth)
                mask_pos = truth > 0
                # print("mask pos:\n",mask_pos)
                mask_neg = truth == 0
                # print("mask neg:\n", mask_neg)
                # # norm_max, max_index = color_res.abs().max(dim=1)
                # # color_res = torch.div(color_res.t(), norm_max)
                # # logit_d = color_res + 0.5*depth_res
                # color_p = logit[mask_pos,1].cpu().data.numpy()
                # color_n = logit[mask_neg,0].cpu().data.numpy()

                color_p = color_res[mask_pos, 1].cpu().data.numpy()
                # dep_p = depth_res[mask_pos, 1].cpu().data.numpy()
                color_n = color_res[mask_neg, 0].cpu().data.numpy()
                # dep_n = depth_res[mask_neg, 0].cpu().data.numpy()
                # print("color_p_src = ", color_p)
                # print("color_n_src = ", color_n)
                # print("dep_p_src = ", dep_p)
                # print("dep_n_src = ", dep_n)
                # print("pos logit = \n", color_res[mask_pos,:])
                # print("neg logit = \n", color_res[mask_neg,:])
                # ########moat loss
                # color_ct_p = color_p.mean()
                # color_ct_n = color_n.mean()
                # color_dis_p = np.exp(color_p_0 - color_p_1)
                # color_dis_p[color_p_1 - color_p_0 > 2] = 0
                # color_dis_n = np.exp(color_n_1 - color_n_0)
                # color_dis_n[color_n_0 - color_n_1 > 2] = 0
                # color_dis_n = color_dis_n.mean()
                # color_dis_p = color_dis_p.mean()
                # color_dis_p = max(1-color_ct_p, 0)
                # color_dis_n = max(3-color_ct_n, 0)
                color_dis_p = 1 - color_p
                color_dis_p[color_dis_p < 0] = 0
                # print("color_p\n",color_dis_p)
                color_dis_n = 1 - color_n
                color_dis_n[color_dis_n < 0] = 0
                # # print("color_n\n",color_dis_n)
                #
                color_dis_n = color_dis_n.mean()
                color_dis_p = color_dis_p.mean()
                # color_ct_p = color_p.mean()
                # color_ct_n = color_n.mean()
                # color_dis_p = math.exp(max(1-color_ct_p, 0))-1
                # color_dis_n = math.exp(max(1-color_ct_n, 0))-1

                # # # ## moat new
                # color_ct_p = 2 - color_p
                # # color_ct_p[color_ct_p<0] = 0
                # color_ct_n = 2 - color_n
                # # color_ct_n[color_ct_n<0] = 0
                # # color_dis_p = color_ct_p.mean()
                # # color_dis_p = color_ct_p.pow(2).mean()
                # color_dis_p = (color_ct_p).exp()
                # color_dis_ppp = color_dis_p.clone().detach()
                # color_dis_ppp[color_ct_p<=0] = 0
                # color_dis_pp = color_dis_ppp.mean()
                # # color_dis_p = color_ct_p.pow(2).sum()
                # # color_dis_n = color_ct_n.mean()
                # # color_dis_n = color_ct_n.pow(2).mean()
                # color_dis_n = (color_ct_n).exp()
                # color_dis_nnn = color_dis_n.clone().detach()
                # color_dis_nnn[color_ct_n<=0] = 0
                # color_dis_nn = color_dis_nnn.mean()
                # color_dis_n = color_ct_n.pow(2).sum()

                # ### cos loss
                # logit_cos = color_res
                # # logit_max, _ = logit_cos.max(dim=1)
                # # logit_min, _ = logit_cos.min(dim=1)
                # # print(logit_max)
                # # print("size : ", logit_min.shape)
                # # logit_cos = logit_cos.t()
                # # print(logit_cos - logit_min)
                # # logit_cos = torch.div((logit_cos - logit_min) , (logit_max - logit_min))
                # # logit_cos = logit_cos.t()
                # # print(logit_cos)
                # # logit_cos = F.softmax(logit_cos, 1)
                # logit_cos[truth>0,1] -= 0.2
                # logit_cos[truth==0,0] -= 0.2

                ### cos new loss
                color_res[truth>0,1] -= 0.2
                color_res[truth==0,0] -= 0.2


                ## center loss
                # norm_max, _ = color_res.abs().max(dim=1)
                # color_res = torch.div(color_res.t(), norm_max)
                # color_res = color_res.t()
                # color_p = color_res[mask_pos, 1]
                # color_n = color_res[mask_neg, 0]
                # if i == 0:
                #     color_dis_p = 0
                #     color_dis_n = 0
                #     pos_ctr = color_p.mean().detach()
                #     neg_ctr = color_n.mean().detach()
                #     # print("pos_ctr = ", pos_ctr)
                #     # print("neg_ctr = ", neg_ctr)
                # else:
                #     color_dis_p = (color_p.clone().detach() - pos_ctr).pow(2).mean()
                #     color_dis_n = (color_n.clone().detach() + neg_ctr).pow(2).mean()
                #     color_p_im = (color_p.clone().detach() - pos_ctr).mean().clone().detach()
                #     color_n_im = (color_n.clone().detach() - neg_ctr).mean().clone().detach()
                #     pos_ctr = pos_ctr - 0.001*color_p_im
                #     neg_ctr = neg_ctr - 0.001*color_n_im


                # depth_p = depth_res[mask_pos, 1]
                # depth_ct_p = depth_p.mean()
                # depth_n = depth_res[mask_neg, 0]
                # depth_ct_n = depth_n.mean()
                # depth_dis_p = max(1.0 - depth_ct_p, 0)
                # depth_dis_n = max(1.0 - depth_ct_n, 0)


                # loss  = criterion(color_res, truth) + criterion(depth_res, truth) + (color_dis_p + color_dis_n)*0.5 + \
                #         (depth_dis_p + depth_dis_n)*0.2
                # loss = criterion(color_res, truth) + (color_dis_p + color_dis_n) * 0.5
                # print("ce : \n", criterion(logit, truth))
                # loss = criterion(color_res, truth) + criterion(depth_res, truth) + (color_dis_p + color_dis_n)*0.5
                # loss = criterion(logit, truth) + (color_dis_p + color_dis_n)
                # loss = criterion(logit, truth) + color_dis_p * 0.5
                loss = criterion(color_res, truth)
                # loss = criterion(logit, truth)

                precision, _ = metric(logit, truth)
                # prob = protru[:,0]
                # print("prob = ", prob)
                # precision,_ = metric(logit_cos, truth)
                # precision = precision.data.cpu().numpy()
                precision = np.mean(precision)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_train_loss += loss.item()
                sum_train_precision += precision.item()
                sum += 1
                # if iter%iter_smooth == 0:
                #     train_loss = sum_train_loss/sum
                #     sum = 0
                i = i + 1
            # print statistics  ------------
            sum_train_loss = sum_train_loss / sum
            sum_train_precision = sum_train_precision / sum
            batch_loss[:2] = np.array((sum_train_loss, sum_train_precision))

            # if epoch >= config.cycle_inter // 2:
            # if cycle_index < 2:
            # if epoch % 2 == 1:
            if epoch >= 0:
            # if True:
                net.eval()
                valid_loss, _ = do_valid_test(net, valid_loader, criterion)
                net.train()

                if valid_loss[1] < min_acer and epoch >= 0:
                    min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                if valid_loss[1] < global_min_acer and epoch >= 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')

                log.write('APCER=%0.4f    |    NPCER=%0.4f  \n' % (valid_loss[3], valid_loss[4]))

            asterisk = ' '
            log.write(config.model_name+' Cycle %d: %0.6f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch, valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                batch_loss[0], batch_loss[1], time_to_str((timer() - start), 'min')))

            # if cycle_index >= 2 and cycle_index <5:
            #     if epoch >= 0:
            #         # if True:
            #         net.eval()
            #         valid_loss, _ = do_valid_test(net, valid_loader, criterion)
            #         net.train()
            #
            #         if valid_loss[1] < min_acer and epoch > 0:
            #             min_acer = valid_loss[1]
            #             ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
            #             torch.save(net.state_dict(), ckpt_name)
            #             log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')
            #
            #         if valid_loss[1] < global_min_acer and epoch > 0:
            #             global_min_acer = valid_loss[1]
            #             ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
            #             torch.save(net.state_dict(), ckpt_name)
            #             log.write('save global min acer model: ' + str(min_acer) + '\n')
            #
            #         log.write('APCER=%0.4f    |    NPCER=%0.4f  \n' % (valid_loss[3], valid_loss[4]))
            #
            #     asterisk = ' '
            #     log.write(
            #         config.model_name + ' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
            #             cycle_index, lr, iter, epoch,
            #             valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
            #             batch_loss[0], batch_loss[1],
            #             time_to_str((timer() - start), 'min')))

            # if cycle_index >= 5:
            #     if epoch >=0 :
            #         # if True:
            #         net.eval()
            #         valid_loss, _ = do_valid_test(net, valid_loader, criterion)
            #         net.train()
            #
            #         if valid_loss[1] < min_acer and epoch > 0:
            #             min_acer = valid_loss[1]
            #             ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
            #             torch.save(net.state_dict(), ckpt_name)
            #             log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')
            #
            #         if valid_loss[1] < global_min_acer and epoch > 0:
            #             global_min_acer = valid_loss[1]
            #             ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
            #             torch.save(net.state_dict(), ckpt_name)
            #             log.write('save global min acer model: ' + str(min_acer) + '\n')
            #
            #         log.write('APCER=%0.4f    |    NPCER=%0.4f  \n' % (valid_loss[3], valid_loss[4]))
            #
            #     asterisk = ' '
            #     log.write(
            #         config.model_name + ' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
            #             cycle_index, lr, iter, epoch,
            #             valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
            #             batch_loss[0], batch_loss[1],
            #             time_to_str((timer() - start), 'min')))
        ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')

def run_val(config):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = './models_mixer'
    # out_dir = './models_wmca'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    if initial_checkpoint is not None:
        # save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)
        # if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
        #     os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size // 2,
                                drop_last   = False,
                                num_workers=8)

    criterion = softmax_cross_entropy_criterion
    net.eval()
    valid_loss, _ = do_valid_test(net, valid_loader, criterion)
    # print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))
    # print("@FPR=0.001 TPR=%0.6f,  FNR=%0.6f"%(valid_loss[4],valid_loss[5]))
    # print("@FPR=0.01 TPR=%0.6f,  FNR=%0.6f"%(valid_loss[6],valid_loss[7]))
    print('ACER=%0.4f    |     APCER=%0.4f    |    NPCER=%0.4f  \n' % (valid_loss[1], valid_loss[3], valid_loss[4]))
    # # print('infer!!!!!!!!!')
    # out = infer_test(net, test_loader)
    print('done')

def run_test(config):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = './models_mixer'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    if initial_checkpoint is not None:
        # save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        # if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
        #     os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    # valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
    #                           fold_index=config.train_fold_index)
    # valid_loader  = DataLoader( valid_dataset,
    #                             shuffle=False,
    #                             batch_size  = config.batch_size // 8,
    #                             drop_last   = False,
    #                             num_workers=8)

    test_dataset = FDDataset(mode = 'test', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    test_loader  = DataLoader( test_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    # criterion = softmax_cross_entropy_criterion
    net.eval()

    do_test(net, test_loader)
    # print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))
    # print("@FPR=0.001 TPR=%0.6f,  FNR=%0.6f"%(valid_loss[4],valid_loss[5]))
    # print("@FPR=0.01 TPR=%0.6f,  FNR=%0.6f"%(valid_loss[6],valid_loss[7]))
    # print('ACER=%.4f    |     APCER=%0.4f    |    NPCER=%0.4f  \n' % (valid_loss[1], valid_loss[4], valid_loss[5]))
    # # print('infer!!!!!!!!!')
    # out = infer_test(net, test_loader)
    print('done')

    # submission(out,save_dir+'_noTTA.txt', mode='test')

def main(config):
    if config.mode == 'train':
        # config.pretrained_model = r'global_min_acer_model.pth'
        run_train(config)

    if config.mode == 'infer_val':
        config.pretrained_model = config.pretrained_model#r'global_min_acer_model.pth'
        run_val(config)

    if config.mode == 'infer_test':
        config.pretrained_model = config.pretrained_model#r'global_min_acer_model.pth'
        run_test(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--image_mode', type=str, default='fusion')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=2)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_val','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    config = parser.parse_args()
    print(config)
    main(config)