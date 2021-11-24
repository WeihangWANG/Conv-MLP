from imgaug import augmenters as iaa
import math
import cv2
from PIL import Image
from skimage import exposure
from torchvision import transforms as tfs
from .data_helper import *

def random_cropping(image, target_shape=(32, 32), is_random = True):
    # print("shape:",image.shape)
    image = cv2.resize(image,(RESIZE_SIZE,RESIZE_SIZE))
    target_h, target_w = target_shape
    height, width = image.shape

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w]
    return zeros

def TTA_9_cropps_color(image, target_shape=(48, 48, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, _ = image.shape
    target_w, target_h, _ = target_shape
    #
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    # start_x = 40
    # start_y = 40
    #
    starts = [[start_x - target_w, start_y - target_w],[start_x - target_w, start_y],[start_x - target_w, start_y + target_w],
              [start_x, start_y - target_w],[start_x, start_y],[start_x, start_y + target_w],
              [start_x + target_w, start_y - target_w],[start_x + target_w, start_y],[start_x + target_w, start_y + target_w],
              ]

    # ## no overlap
    # starts = [[0, 0], [0, 32], [0, 64], [32, 0], [32, 32], [32, 64], [64, 0], [64, 32], [64, 64], ]
    # target_w, target_h = 32, 32
    # starts = [[0, 0], [0, 24], [0, 48], [0, 72], [24, 0], [24, 24], [24, 48], [24, 72], [48, 0], [48, 24], [48, 48],
    #           [48, 72], [72, 0], [72, 24], [72, 48], [72, 72], ]
    # target_w, target_h = 24, 24

    ## no overlap
    # starts = [[0, 0], [0, 37], [0, 74], [37, 0], [37, 37], [37, 74], [74, 0], [74, 37], [74, 74], ]
    # target_w, target_h = 37, 37

    # # left-up index 0
    # starts = [[start_x - target_w, start_y - target_w],  [start_x, start_y - target_w], [start_x + target_w, start_y - target_w],
    #           [start_x - target_w, start_y], [start_x, start_y], [start_x + target_w, start_y],
    #           [start_x - target_w, start_y + target_w], [start_x, start_y + target_w], [start_x + target_w, start_y + target_w],
    #           ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index
        # x = random.randint(0, width - target_w)
        # y = random.randint(0, height - target_h)

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y + target_h]
        # zeros = cv2.resize(zeros, (48, 48))  ## overlap adjust

        image_ = zeros.copy()

        images.append(image_.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))

    return images

def TTA_9_cropps(image, target_shape=(48, 48, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height = image.shape
    target_w, target_h, _ = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    # start_x = 40
    # start_y = 40
    #
    starts = [[start_x - target_w, start_y - target_w], [start_x - target_w, start_y],
              [start_x - target_w, start_y + target_w],
              [start_x, start_y - target_w], [start_x, start_y], [start_x, start_y + target_w],
              [start_x + target_w, start_y - target_w], [start_x + target_w, start_y],
              [start_x + target_w, start_y + target_w],
              ]

    # ## no overlap
    # starts = [[0, 0],[0, 32],[0, 64],[32, 0],[32, 32],[32, 64],[64, 0],[64, 32],[64, 64],]
    # target_w, target_h = 32, 32
    # starts = [[0, 0], [0, 24], [0, 48], [0, 72], [24, 0], [24, 24], [24, 48], [24, 72], [48, 0], [48, 24], [48, 48],
    #           [48, 72], [72, 0], [72, 24], [72, 48], [72, 72],]
    # target_w, target_h = 24, 24

    ## no overlap
    # starts = [[0, 0],[0, 37],[0, 74],[37, 0],[37, 37],[37, 74],[74, 0],[74, 37],[74, 74],]
    # target_w, target_h = 37, 37
    # starts = [[0, 0], [0, 37], [0, 74], [37, 0], [37, 37], [37, 74], [74, 0], [74, 37], [74, 74], ]
    # target_w, target_h = 37, 37

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index
        # x = random.randint(0, width - target_w)
        # y = random.randint(0, height - target_h)

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y + target_h]
        # zeros = cv2.resize(zeros, (48, 48))  ## overlap adjust

        image_ = zeros.copy()

        images.append(image_.reshape([1, target_shape[0], target_shape[1], 1]))

    return images

def random_resize(img, probability = 0.5,  minRatio = 0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio_h = random.uniform(minRatio, 1.0)
    ratio_w = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h*ratio_h)
    new_w = int(w*ratio_w)

    img = cv2.resize(img, (new_w,new_h))
    img = cv2.resize(img, (w, h))
    return img

def augumentor_1(color, depth, ir, target_shape=(48, 48, 3), is_infer=False):
    fake = 0
    if is_infer:
        # augment_img = iaa.Sequential([
        #     iaa.Fliplr(0),
        # ])
        # color = augment_img.augment_image(color)

        color = TTA_9_cropps_color(color, target_shape)
        # ir = TTA_9_cropps_color(ir, target_shape)
        # depth = TTA_9_cropps_color(depth, target_shape)
        ir = TTA_9_cropps(ir, target_shape)
        depth = TTA_9_cropps(depth, target_shape)

        return color, depth, ir, fake

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)
        # augment_img = iaa.Sequential([
        #     iaa.Fliplr(0),
        # ])


        color = augment_img.augment_image(color)
        ir = augment_img.augment_image(ir)
        depth = augment_img.augment_image(depth)


        # exp_num = len(ir[ir > 200])
        # if exp_num >= 150:
        #     ir = cv2.merge([ir] * 3)
        #     ir = Image.fromarray(ir)
        #     ir = tfs.ColorJitter(brightness=(0.5,0.5), contrast=(2.0,2.0), saturation=(1.0,1.0))(ir)
        #     ir = cv2.cvtColor(np.array(ir), cv2.COLOR_BGR2GRAY)

        # if np.random.random() <= 0.001:
        #     x = random.randint(0, 63)
        #     y = random.randint(0, 63)
        #     color[y:y + 48, x:x + 48] = 0
        #     ir[y:y + 48, x:x + 48] = 0
        #     depth[y:y + 48, x:x + 48] = 0

        if np.random.random() <= 0.001:
            fake = 1
            value = 0
            color[:, :] = value     # full fill black
            ir[:, :] = value     # full fill black
            depth[:, :] = value     # full fill black
        #
        #     # if np.random.random() <= 0.3:
        #     #     color = cv2.GaussianBlur(color, (9, 9), 0)
        #     #     ir = cv2.GaussianBlur(ir, (9, 9), 0)
        #
        # if np.random.random() <= 0.2:
        #     color = Image.fromarray(color)
        #     color = tfs.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(color)
        #     color = cv2.cvtColor(np.array(color), cv2.COLOR_RGB2BGR)
        #     color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        #
        # # if np.random.random() <= 0.2:
        # #     # depth = exposure.adjust_gamma(depth, 0.5)
        # #     ir = Image.fromarray(ir)
        # #     ir = tfs.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(ir)
        # #     ir = cv2.cvtColor(np.array(ir), cv2.COLOR_RGB2BGR)
        # #     ir = cv2.cvtColor(ir, cv2.COLOR_RGB2BGR)
        # #     depth = Image.fromarray(depth)
        # #     depth = tfs.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(depth)
        # #     depth = cv2.cvtColor(np.array(depth), cv2.COLOR_RGB2BGR)
        # #     depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
        #
        # if np.random.random() <= 0.4:
        #     color[:,:10] = 0     # fill left column
        #     ir[:,:10] = 0     # fill left column
        #     depth[:,:10] = 0     # fill left column

        depth = random_resize(depth)
        ir = random_resize(ir)
        color = random_resize(color)

        depth = TTA_9_cropps(depth, target_shape)
        ir = TTA_9_cropps(ir, target_shape)
        color = TTA_9_cropps_color(color, target_shape)
        # ir = TTA_9_cropps_color(ir, target_shape)
        # depth = TTA_9_cropps_color(depth, target_shape)

        return color, depth, ir, fake

## augment for validation and test
def augumentor_2(color, depth, ir, target_shape=(32, 32, 3)):

        augment_img = iaa.Sequential([
            iaa.Fliplr(1),
        ])

        ## horizontal flip
        depth_lr = augment_img.augment_image(depth)
        color_lr = augment_img.augment_image(color)
        ir_lr = augment_img.augment_image(ir)

        # exp_num = len(ir[ir > 200])
        # if exp_num >= 150:
        #     ir = cv2.merge([ir] * 3)
        #     ir = Image.fromarray(ir)
        #     ir = tfs.ColorJitter(brightness=(0.5, 0.5), contrast=(2.0, 2.0), saturation=(1.0, 1.0))(ir)
        #     ir = cv2.cvtColor(np.array(ir), cv2.COLOR_BGR2GRAY)

        ## segmentation
        # src
        ir = TTA_9_cropps(ir, target_shape)
        depth = TTA_9_cropps(depth, target_shape)
        color = TTA_9_cropps_color(color, target_shape)
        # ir = TTA_9_cropps_color(ir, target_shape)
        # depth = TTA_9_cropps_color(depth, target_shape)
        # horizontal flip
        ir_lr= TTA_9_cropps(ir_lr, target_shape)
        depth_lr = TTA_9_cropps(depth_lr, target_shape)
        color_lr = TTA_9_cropps_color(color_lr, target_shape)
        # ir_lr = TTA_9_cropps_color(ir_lr, target_shape)
        # depth_lr = TTA_9_cropps_color(depth_lr, target_shape)

        return color, depth, ir, color_lr, depth_lr, ir_lr