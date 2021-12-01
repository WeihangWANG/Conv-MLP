from imgaug import augmenters as iaa
import math
import cv2
from PIL import Image
from skimage import exposure
from torchvision import transforms as tfs
from .prepare_data import *

def random_cropping(image, target_shape=(48, 48), is_random = True):
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

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y + target_h]

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

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y + target_h]

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

        color = TTA_9_cropps_color(color, target_shape)
        ir = TTA_9_cropps(ir, target_shape)
        depth = TTA_9_cropps(depth, target_shape)

        return color, depth, ir, fake

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        color = augment_img.augment_image(color)
        ir = augment_img.augment_image(ir)
        depth = augment_img.augment_image(depth)

        if np.random.random() <= 0.001:
            fake = 1
            value = 0
            color[:, :] = value     # full fill black
            ir[:, :] = value     # full fill black
            depth[:, :] = value     # full fill black
        
        depth = random_resize(depth)
        ir = random_resize(ir)
        color = random_resize(color)

        depth = TTA_9_cropps(depth, target_shape)
        ir = TTA_9_cropps(ir, target_shape)
        color = TTA_9_cropps_color(color, target_shape)

        return color, depth, ir, fake

## augment for validation and test
def augumentor_2(color, depth, ir, target_shape=(48, 48, 3)):

        augment_img = iaa.Sequential([
            iaa.Fliplr(1),
        ])

        ## horizontal flip
        depth_lr = augment_img.augment_image(depth)
        color_lr = augment_img.augment_image(color)
        ir_lr = augment_img.augment_image(ir)

        ## segmentation
        # src
        ir = TTA_9_cropps(ir, target_shape)
        depth = TTA_9_cropps(depth, target_shape)
        color = TTA_9_cropps_color(color, target_shape)
        # horizontal flip
        ir_lr= TTA_9_cropps(ir_lr, target_shape)
        depth_lr = TTA_9_cropps(depth_lr, target_shape)
        color_lr = TTA_9_cropps_color(color_lr, target_shape)

        return color, depth, ir, color_lr, depth_lr, ir_lr