from utils import *
from .augmentation import *
from .prepare_data import *

class FasDataset(Dataset):
    def __init__(self, mode, image_size=128, augment = None, balance = False):
        super(FasDataset, self).__init__()

        self.augment = augment
        self.mode       = mode
        self.balance = balance

        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size

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
                color, depth, ir, label = tmp_list[pos]
            else:
                color, depth, ir, label = self.train_list[index]

        elif self.mode == 'val':
            color, depth, ir, label = self.val_list[index]
            
        color = cv2.imread(os.path.join(DATA_ROOT, color),1)
        depth = cv2.imread(os.path.join(DATA_ROOT, depth),0)
        ir = cv2.imread(os.path.join(DATA_ROOT, ir),0)

        color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
        depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
        ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))

        if self.mode == 'train':
            color, depth, ir, fake = augumentor_1(color, depth, ir, target_shape=(self.image_size, self.image_size, 3), is_infer=False)
            n = len(depth)
            depth = np.concatenate(depth, axis=0)
            ir = np.concatenate(ir, axis=0)
            color = np.concatenate(color, axis=0)

            image = np.concatenate([
                depth.reshape([n, self.image_size, self.image_size, 1]),
                ir.reshape([n,self.image_size, self.image_size, 1]),
                color.reshape([n, self.image_size, self.image_size, 3])],

                axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, 5, self.image_size, self.image_size])
            image = image / 255.0

            if fake > 0:
                label = 0
            else:
                label = int(label)

            return torch.FloatTensor(image), torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        if self.mode == 'val':
            color, depth, ir, color_lr, depth_lr, ir_lr = augumentor_2(color, depth, ir, target_shape=(self.image_size, self.image_size, 3))
            n = len(depth)

            ## src
            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            ir = np.concatenate(ir, axis=0)
            image = np.concatenate([depth.reshape([n,self.image_size, self.image_size, 1]),
                                    ir.reshape([n,self.image_size, self.image_size, 1]),
                                    color.reshape([n, self.image_size, self.image_size, 3])],
                                    axis=3)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, 5, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

    def __len__(self):
        return self.num_data





