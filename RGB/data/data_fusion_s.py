from utils import *
# from .augmentation import *
from .augmentation_oulu import *
from .prepare_data import *

class FasDataset(Dataset):
    def __init__(self, mode, image_size=128, augment = None, balance = False):
        super(FasDataset, self).__init__()

        self.augment = augment
        self.mode       = mode
        self.balance = balance

        self.channels = 3
        self.frame_len = 120#96
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size

        self.pixel_mean = [103.52,116.28,123.675]
        self.Pixel_std = [57.375,57.12,58.395]
        self.scale = 1.0 

        self.set_mode(self.mode)

    def set_mode(self, mode):
        self.mode = mode
        print(mode)
        
        if self.mode == 'val':
            self.val_list = load_val_list()
            self.num_data = len(self.val_list)
            print('set dataset mode: val')

        elif self.mode == 'train':
            self.train_list = load_train_list()
            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)
            print('set dataset mode: train')

            if self.balance:
                self.train_list = transform_balance(self.train_list)

        print(self.num_data)


    def __getitem__(self, index):

        if self.mode == 'train':
            if self.balance:
                if random.randint(0,1)==0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0,len(tmp_list)-1)
                color, label= tmp_list[pos]
            else:
                color, label= self.train_list[index]

        elif self.mode == 'val':
            # single-frame
            #color, label= self.val_list[index]

            # video
            label, videoname = self.val_list[index]
            if int(label) > 0:
                label = 1
            else:
                label = 0
        
        if self.mode == 'train':
            color = cv2.imread(os.path.join(DATA_ROOT, color),1)
            color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
            color = augumentor_1(color, label, target_shape=(self.image_size, self.image_size, 3))
            n = len(color)
            
            image = np.concatenate(color, axis=0)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, 3, self.image_size, self.image_size])
            # print(image.shape)

            # for ch in range(3):
            #     image[:,ch,:,:] = (image[:,2-ch,:,:]/self.scale - self.pixel_mean[2-ch] ) / self.Pixel_std[2-ch]
            # image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        if self.mode == 'val':
            val_path = os.path.join(VAL_ROOT, videoname)
            image_x = np.zeros((self.frame_len, 9, 3, self.image_size, self.image_size))
            label_list = np.zeros(self.frame_len)
            frames_total = len([name for name in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, name))])
            for idx in range(self.frame_len):
                if idx >= frames_total:
                    img_idx = np.random.randint(0,frames_total)
                else:
                    img_idx = idx
                img_dir = val_path + '/%s.jpg'%img_idx
                color = cv2.imread(img_dir, 1)
                color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
                color = augumentor_2(color, target_shape=(self.image_size, self.image_size, 3))
                image = np.concatenate(color, axis=0)
                image = np.transpose(image, (0, 3, 1, 2))
                image = image.astype(np.float32)
                image = image.reshape([9, 3, self.image_size, self.image_size])
                image_x[idx, :, :, :, :] = image
                label_list[idx] = int(label)
                
            return torch.FloatTensor(image_x), torch.LongTensor(np.asarray(label_list).reshape([-1]))

    def __len__(self):
        return self.num_data
