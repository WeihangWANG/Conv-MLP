from imgaug import augmenters as iaa
import random
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

def CutOut(img, length=20):
    h, w = img.shape[0], img.shape[1]    # Tensor [1][2],  nparray [0][1]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    length_new = np.random.randint(1, length)

    y1 = np.clip(y - length_new // 2, 0, h)
    y2 = np.clip(y + length_new // 2, 0, h)
    x1 = np.clip(x - length_new // 2, 0, w)
    x2 = np.clip(x + length_new // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    img[mask == 0.]= 0.
    return img

def RandomErasing(img, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):

    if random.uniform(0, 1) < probability:
        attempts = np.random.randint(1, 3)
        for attempt in range(attempts):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)

                img[x1:x1+h, y1:y1+w, 0] = mean[0]
                img[x1:x1+h, y1:y1+w, 1] = mean[1]
                img[x1:x1+h, y1:y1+w, 2] = mean[2]
    return img

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

def augumentor_1(color, label, target_shape=(32, 32, 3)):

    augment_img_neg = iaa.Sequential([
        iaa.Fliplr(0.5),
        #iaa.Add(value=(0,20),per_channel=True),
        iaa.Add(value=(-10,10),per_channel=True),
        iaa.GammaContrast(gamma=(0.5, 1.5)),
        #iaa.Affine(rotate=(-30, 30)),
    ])

    augment_img_pos = iaa.Sequential([
        iaa.Fliplr(0.5),
        #iaa.Add(value=(-10,10),per_channel=True),
        #iaa.Add(value=(-10,10),per_channel=True),
        #iaa.GammaContrast(gamma=(0.5, 1.5)),
    ])

    if random.random() < 0.5:
        if int(label) > 0:
            color = augment_img_pos.augment_image(color)
        else:
            color = augment_img_neg.augment_image(color)
    #color = RandomErasing(color)
    if random.random() < 0.1:
        color = CutOut(color)
#    if int(label) == 0:
#        if random.random() < 0.2:
#            color = noise.augment_image(color)
    
    #color = random_resize(color)
    color = (color - 127.5) / 128
    #color = color / 255.0
    # color = image_into_patches(color, target_shape, 0.25)
    color = TTA_9_cropps_color(color, target_shape)

    return color

## augment for validation and test
def augumentor_2(color, target_shape=(32, 32, 3)):

    augment_img = iaa.Sequential([
         iaa.Fliplr(0.5),
    ])
    color = augment_img.augment_image(color)
    #color = random_resize(color)
    color = (color - 127.5) / 128
    #color = color / 255.0
    ## segmentation
    # src
    # color = image_into_patches(color, target_shape, 0.25)
    color = TTA_9_cropps_color(color, target_shape)
    return color

def image_into_patches(image, target_patch=32, overlap=0.25):
    # print("*****image shape*******")
    # print(image.shape)
    # print(target_patch)
    width, height, _ = image.shape
    overlap_size = int(target_patch[0] * (1 - overlap))

    patch_num = (width - target_patch[0]) // overlap_size + 1

    starts = []
    for i in range(patch_num):
        for j in range(patch_num):
            starts.append([overlap_size * i, overlap_size * j])

    images = []
    for start_index in starts:
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_patch[0] >= width:
            x = width - target_patch[0] - 1
        if y + target_patch[0] >= height:
            y = height - target_patch[0] - 1

        patch = image[x:x + target_patch[0], y: y + target_patch[0]]
        img_ = patch.copy()
        images.append(img_.reshape([1,target_patch[0],target_patch[1],target_patch[2]]))
        
    return images
