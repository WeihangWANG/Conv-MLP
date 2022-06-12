import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils_st import load_checkpoint, save_checkpoint, save_max_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Conv-MLP ImageNet training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def main(config):
    
    data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    # print("*******************")
    # print(config.LOCAL_RANK)
    # print("*******************")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,  find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # if hasattr(model_without_ddp, 'flops'):
    #     flops = model_without_ddp.flops()
    #     logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = softmax_cross_entropy_criterion
    

    max_accuracy = 0.5

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, )
        acer, apcer, npcer, _ = validate(config, data_loader_val, model)
        logger.info(f"ACER of the network on the oulu test images: {acer:.4f} with APCER: {apcer:.4f} and BPCER: {npcer:.4f}.")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)

        acer, apcer, npcer, reslist = validate(config, data_loader_val, model)
        logger.info(f"ACER of the network on the oulu test images: {acer:.4f} with APCER: {apcer:.4f} and BPCER: {npcer:.4f}.")
        
        if dist.get_rank() == 0 and (acer < max_accuracy):
            max_accuracy = acer
            save_max_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
            resname = os.path.join(config.OUTPUT, f'reslist.txt')
            f = open(resname,'w')
            f.write("%s"%reslist)
            f.close()
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        max_accuracy = min(max_accuracy, acer)
        logger.info(f"Minimal ACER of the network on the oulu test images: {max_accuracy:.4f}.")

    total_time = time.time() 
    start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        targets = targets.view(outputs.shape[0])
        #print(outputs)
        mask_pos = targets > 0
        mask_neg = targets == 0
        color_p = outputs[mask_pos, 1].cpu().data.numpy()
        color_n = outputs[mask_neg, 0].cpu().data.numpy()
        color_dis_p = 1 - color_p  # protocol 4 : 1; 
        color_dis_p[color_dis_p < 0] = 0
        color_dis_n = 1 - color_n  # protocol 4 : 1;
        color_dis_n[color_dis_n < 0] = 0
        color_dis_n = color_dis_n.mean()
        color_dis_p = color_dis_p.mean()

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets) + (color_dis_p + color_dis_n)*0.5
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets) + (color_dis_p + color_dis_n)*0.5
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


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

import torch.nn.functional as F
def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    correct = correct.data.cpu().numpy()
    return correct, prob

@torch.no_grad()
def validate(config, data_loader, model):
    losses   = []
    corrects = []
    probs = []
    labels = []
    valid_num = 0
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = softmax_cross_entropy_criterion
    
    model.eval()

    acer_meter = AverageMeter()
    apcer_meter = AverageMeter()
    bpcer_meter = AverageMeter()
    res_con = []
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        #print("image shape : ",images.shape)
        #images = images.view(96,9,3,48,48)
        images = images.view(120,9,3,48,48)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        with torch.no_grad():
            # src
            # res, dep_res = net(input, input_lr)
            res = model(images)
            # logit_sum = res + dep_res * 0.5
            
            logit_mean = res.mean(dim=0)
            target = target.view(res.shape[0])[0].data.cpu().numpy()
            #print("target = ",target)
            prob = F.softmax(logit_mean)
            #print("prob = ",prob)
            pred = prob[1].data.cpu().numpy()
            res_con.append([pred,target])
            thres = 0.9 # procotol 3&4 : 0.9; protocol 2: 0.5
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




#            logit_sum = res
#            #print(res.shape)
#            target = target.view(res.shape[0])
#            #loss = criterion(logit_sum, target, False)
#            correct, prob = metric(logit_sum, target)
#            #print("prob torch shape", prob.shape)
#            prob = prob.data.cpu().numpy()
#            #print("prob shape",prob.shape)
#            target = target.data.cpu().numpy()
#            
#            acer, apcer, bpcer, _, _, _, _ = ACER(0.5, prob[:,1], target)
#            print("ACER = ", acer)
            acer_meter.update(acer)
            apcer_meter.update(apcer)
            bpcer_meter.update(bpcer)
        #valid_num += len(images)
        #losses.append(loss.data.cpu().numpy())
        #corrects.append(np.asarray(correct))
        #probs.append(prob.data.cpu().numpy())
        #labels.append(target.data.cpu().numpy())

#    correct_cat = np.concatenate(corrects)
#    loss    = np.concatenate(losses)
#    loss    = loss.mean()
#    correct_ = correct_cat.mean()

#    probs = np.concatenate(probs)
#    labels = np.concatenate(labels)

#    acer,apcer,npcer,_,_,_,_ = ACER(0.9, probs[:, 1], labels)

    logger.info(f' * ACER {acer_meter.avg:.4f} APCER {apcer_meter.avg:.4f} BPCER {bpcer_meter.avg:.4f}')
    return acer_meter.avg, apcer_meter.avg, bpcer_meter.avg, res_con


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    #print("=============local rank ", config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    #print("----------------------get rank = ", dist.get_rank())
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 1:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
