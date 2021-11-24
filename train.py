import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
import sys
sys.path.append("..")
import argparse
import math
from data.data_fusion_s import *
from data.prepare_data import *
from metric import *
import torch.nn.functional as F
from model_fusion.conv_mlp import ConvMLP
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def run_train(config):
    # out_dir = './models_mixer'
    out_dir = './models_wmca'
    config.model_name = 'Conv-MLP_' + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    criterion          = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'),mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = FasDataset(mode = 'train', image_size=config.image_size)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size  = config.batch_size, drop_last = True, num_workers = 8)

    valid_dataset = FasDataset(mode = 'val', image_size=config.image_size)
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

    net = ConvMLP(num_class=2)
    print(net)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)

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
                color_res = net.forward(input)

                truth = truth.view(color_res.shape[0])

                logit = color_res

                mask_pos = truth > 0
                mask_neg = truth == 0
                color_p = color_res[mask_pos, 1].cpu().data.numpy()
                color_n = color_res[mask_neg, 0].cpu().data.numpy()
                color_dis_p = 1 - color_p
                color_dis_p[color_dis_p < 0] = 0
                color_dis_n = 1 - color_n
                color_dis_n[color_dis_n < 0] = 0
                color_dis_n = color_dis_n.mean()
                color_dis_p = color_dis_p.mean()
                loss = criterion(color_res, truth)
                precision, _ = metric(logit, truth)
                precision = np.mean(precision)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_train_loss += loss.item()
                sum_train_precision += precision.item()
                sum += 1
                i = i + 1
            
            sum_train_loss = sum_train_loss / sum
            sum_train_precision = sum_train_precision / sum
            batch_loss[:2] = np.array((sum_train_loss, sum_train_precision))

            if epoch >= 0:
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
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)
        
    valid_dataset = FasDataset(mode = 'val', image_size=config.image_size)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size // 2,
                                drop_last   = False,
                                num_workers=8)

    criterion = softmax_cross_entropy_criterion
    net.eval()
    valid_loss, _ = do_valid_test(net, valid_loader, criterion)
    print('ACER=%0.4f    |     APCER=%0.4f    |    NPCER=%0.4f  \n' % (valid_loss[1], valid_loss[3], valid_loss[4]))
    print('done')


def main(config):
    if config.mode == 'train':
        # config.pretrained_model = r'global_min_acer_model.pth'
        run_train(config)

    if config.mode == 'infer_val':
        config.pretrained_model = config.pretrained_model#r'global_min_acer_model.pth'
        run_val(config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=2)
    parser.add_argument('--cycle_inter', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_val'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    config = parser.parse_args()
    print(config)
    main(config)