import torch
import os
import cv2
import argparse
from process.data_fusion_s import *
from process.augmentation import *
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet

    net = FusionNet(num_class=num_class)
    return net

def reshape_transform(tensor):
    tensor = torch.mean(tensor, dim=2)
    # result = tensor[:, 1 :  , :].reshape(tensor.size(0),
    #     height, width, tensor.size(2))
    #
    # # Bring the channels to the first dimension,
    # # like in CNNs.
    # result = result.transpose(2, 3).transpose(1, 2)
    return tensor

def main(config):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
        # out_dir = './models_mixer'
    out_dir = './models_wmca'
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

    print(net.module.mixer.bottleneck_4)

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    target_layer = net.module.mixer.bottleneck_4

    cam = GradCAM(model=net,
                  target_layer=target_layer,
                  use_cuda=config.use_cuda,
                  reshape_transform=reshape_transform)

    color_src = cv2.imread(r"/home/wwh/E/CASIA-CeFA-Challenge/test/000033/profile/0006_rect.jpg", 1)
    depth_src = cv2.imread(r"/home/wwh/E/CASIA-CeFA-Challenge/test/000033/depth/0006_rect.jpg", 0)
    ir_src = cv2.imread(r"/home/wwh/E/CASIA-CeFA-Challenge/test/000033/ir/0006_rect.jpg", 0)

    color_src = cv2.resize(color_src, (RESIZE_SIZE, RESIZE_SIZE))
    depth_src = cv2.resize(depth_src, (RESIZE_SIZE, RESIZE_SIZE))
    ir_src = cv2.resize(ir_src, (RESIZE_SIZE, RESIZE_SIZE))

    color, depth, ir, fake = augumentor_1(color_src, depth_src, ir_src, target_shape=(48, 48, 3), is_infer=True)
    n = len(depth)
    depth = np.concatenate(depth, axis=0)
    ir = np.concatenate(ir, axis=0)
    color = np.concatenate(color, axis=0)
    depth_src = cv2.merge([depth_src] * 3)

    image = np.concatenate([
        depth.reshape([n, 48, 48, 1]),
        ir.reshape([n, 48, 48, 1]),
        color.reshape([n, 48, 48, 3])],
        axis=3)

    image = np.transpose(image, (0, 3, 1, 2))
    image = image.astype(np.float32)
    image = image.reshape([n, 5, 48, 48])
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.unsqueeze(0)
    print(image.size())

    depth_src = np.transpose(depth_src, (2, 0, 1))
    depth_src = depth_src.astype(np.float32)
    depth_src = depth_src.reshape([3, RESIZE_SIZE, RESIZE_SIZE])
    depth_src = depth_src / 255
    depth_src = torch.FloatTensor(depth_src)
    depth_src = depth_src.unsqueeze(0)
    print(depth_src.size())

    target_category = None
    input = image.clone()
    input = input.cuda()
    res = net(input)

    grayscale_cam = cam(input_tensor=image,
                        target_category=target_category,
                        eigen_smooth=config.eigen_smooth
                        )

    grayscale_cam = grayscale_cam[0, :]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--image_mode', type=str, default='fusion')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=1)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_val','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default='global_min_acer_model.pth')

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    config = parser.parse_args()
    print(config)
    main(config)