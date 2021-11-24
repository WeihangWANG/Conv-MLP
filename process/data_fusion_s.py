from utils import *
from .augmentation import *
from .data_helper import *

class FDDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index='<NIL>', image_size=128, augment = None, balance = False):
        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

        self.augment = augment
        self.mode       = mode
        self.modality = modality
        self.balance = balance

        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index

        self.set_mode(self.mode,self.fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print(mode)
        print('fold index set: ', fold_index)

        if self.mode == 'test':
            self.test_list = load_test_list()
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        elif self.mode == 'val':
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

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

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
            # color, depth, ir, label, fold = self.val_list[index]

        elif self.mode == 'test':
            color,depth,ir = self.test_list[index]
            test_id = color+' '+depth+' '+ir

        color = cv2.imread(os.path.join(DATA_ROOT, color),1)
        depth = cv2.imread(os.path.join(DATA_ROOT, depth),0)
        ir = cv2.imread(os.path.join(DATA_ROOT, ir),0)

        color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
        depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
        ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))

        if self.mode == 'train':
            depth_src = cv2.merge([depth] * 3)
            # depth_src = color.copy()
            # depth_src = cv2.resize(depth_src, (self.image_size, self.image_size))
            # n_d = len(depth_src)
            # print("nd=", n_d)
            # print("depth src shape=", depth_src.shape)
            color, depth, ir, fake = augumentor_1(color, depth, ir, target_shape=(self.image_size, self.image_size, 3), is_infer=False)
            # depth, ir = augumentor(depth, ir, target_shape=(self.image_size, self.image_size))
            n = len(depth)
            # print("n=", n)
            # print("fake is ", fake)

            depth = np.concatenate(depth, axis=0)
            # depth_src = np.concatenate(depth_src, axis=0)
            ir = np.concatenate(ir, axis=0)
            color = np.concatenate(color, axis=0)

            image = np.concatenate([
                # depth.reshape([n, self.image_size, self.image_size, 3]),
                depth.reshape([n, self.image_size, self.image_size, 1]),
                # ir.reshape([n,self.image_size, self.image_size, 3]),
                ir.reshape([n,self.image_size, self.image_size, 1]),
                # ir.reshape([n,self.image_size, self.image_size, 1]),
                # ir.reshape([n,self.image_size, self.image_size, 1]),
                color.reshape([n, self.image_size, self.image_size, 3])],

                # depth.reshape([n, self.image_size, self.image_size, 1]),
                # ir.reshape([n, self.image_size, self.image_size, 1])],
                # depth.reshape([n, self.image_size, self.image_size, 1])],
                axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            depth_src = np.transpose(depth_src, (2, 0, 1))
            image = image.astype(np.float32)
            depth_src = depth_src.astype(np.float32)
            image = image.reshape([n, 5, self.image_size, self.image_size])
            depth_src = depth_src.reshape([3, RESIZE_SIZE, RESIZE_SIZE])
            image = image / 255.0
            depth_src = depth_src / 255.0

            if fake > 0:
                label = 0
            else:
                label = int(label)

            # label = int(label)

            return torch.FloatTensor(image), torch.FloatTensor(depth_src), torch.LongTensor(np.asarray(label).reshape([-1]))

        if self.mode == 'val':
            depth_src = cv2.merge([depth] * 3)
            # depth_src = color.copy()
            # depth_src = cv2.resize(depth_src, (self.image_size, self.image_size))
            color, depth, ir, color_lr, depth_lr, ir_lr = augumentor_2(color, depth, ir, target_shape=(self.image_size, self.image_size, 3))
            # depth, ir = augumentor(depth,ir, target_shape=(self.image_size, self.image_size), is_infer=True)
            n = len(depth)


            ## src
            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            # depth_src = np.concatenate(depth_src, axis=0)
            ir = np.concatenate(ir, axis=0)
            image = np.concatenate([
                                    # depth.reshape([n,self.image_size, self.image_size, 3]),
                                    depth.reshape([n,self.image_size, self.image_size, 1]),
                                    # ir.reshape([n,self.image_size, self.image_size, 3]),
                                    ir.reshape([n,self.image_size, self.image_size, 1]),
                                    # ir.reshape([n,self.image_size, self.image_size, 1]),
                                    # ir.reshape([n,self.image_size, self.image_size, 1]),
                                    # ir.reshape([n,self.image_size, self.image_size, 1])],
                                    color.reshape([n, self.image_size, self.image_size, 3])],

                                    # depth.reshape([n, self.image_size, self.image_size, 1]),
                                    # depth.reshape([n, self.image_size, self.image_size, 1]),
                                    # ir.reshape([n, self.image_size, self.image_size, 1]),
                                    # depth.reshape([n, self.image_size, self.image_size, 1])],

                                   axis=3)
            image = np.transpose(image, (0, 3, 1, 2))
            depth_src = np.transpose(depth_src, (2, 0, 1))
            image = image.astype(np.float32)
            depth_src = depth_src.astype(np.float32)
            image = image.reshape([n, 5, self.image_size, self.image_size])
            image = image / 255.0
            depth_src = depth_src.reshape([3, RESIZE_SIZE, RESIZE_SIZE])
            depth_src = depth_src / 255.0

            # ## flip
            # color_lr = np.concatenate(color_lr, axis=0)
            # depth_lr = np.concatenate(depth_lr, axis=0)
            # ir_lr = np.concatenate(ir_lr, axis=0)
            # image_lr = np.concatenate([
            #     depth_lr.reshape([n, self.image_size, self.image_size, 1]),
            #     ir_lr.reshape([n, self.image_size, self.image_size, 1]),
            #     color_lr.reshape([n, self.image_size, self.image_size, 3])],
            #     axis=3)
            # image_lr = np.transpose(image_lr, (0, 3, 1, 2))
            # image_lr = image_lr.astype(np.float32)
            # image_lr = image_lr.reshape([n, 5, self.image_size, self.image_size])
            # image_lr = image_lr / 255.0

            label = int(label)
            # fold = int(fold)
            # return torch.FloatTensor(image), torch.FloatTensor(depth_src), torch.LongTensor(np.asarray(label).reshape([-1])), torch.LongTensor(np.asarray(fold).reshape([-1]))
            return torch.FloatTensor(image), torch.FloatTensor(depth_src), torch.LongTensor(np.asarray(label).reshape([-1]))


        elif self.mode == 'test':
            color, depth, ir, fake = augumentor_1(color, depth, ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            # depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            # ir = ir_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            ir = np.concatenate(ir, axis=0)

            image = np.concatenate([
                                    depth.reshape([n, self.image_size, self.image_size, 3]),
                                    ir.reshape([n, self.image_size, self.image_size, 3]),
                                    color.reshape([n, self.image_size, self.image_size, 3]),],
                                   axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0
            return torch.FloatTensor(image), test_id

    def __len__(self):
        return self.num_data


# check #################################################################
def run_check_train_data():
    dataset = FDDataset(mode = 'train')
    # print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label)

        if m > 100:
            break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


