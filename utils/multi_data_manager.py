import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR100_224, iImageNetR, iCUB200_224, iResisc45_224, iCARS196_224, iSketch345_224
from copy import deepcopy
import random
import os
class MultiDataManager(object):
    def __init__(self, dataset_names, dataset_order, shuffle, seed):
    
        self.dataset_names = [dataset_names[i] for i in dataset_order]
        self.shuffle = shuffle
        self.seed = seed
        self.current_task = 0  # 当前任务 ID，默认为 0
        self._setup_data()
        self.global_total_classes = len(self._class_order)  # 全局类别总数
        assert sum(self._increments) == self.global_total_classes, 'Task increments must match global total classes.'


    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        self.current_task = task
        return self._increments[task]
    
    
    def get_current_dataset_name(self):
      
        if 0 <= self.current_task < len(self.dataset_names):
            return self.dataset_names[self.current_task]
        return "Unknown"


    def load_images(self, image_paths):
      
        print("进入 load_images 方法")
        print(f"os 模块: {os}")
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"图像路径不存在: {path}")
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)  # 统一尺寸
            img = np.array(img)  # [H, W, C]
            img = img.transpose(2, 0, 1)  # [C, H, W]
            images.append(img)
        return np.array(images)  # [N, 3, 224, 224]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, with_raw=False, with_noise=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf_list[self.current_task], *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf_list[self.current_task], transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf_list[self.current_task], *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path, with_raw, with_noise)
        else:
            return DummyDataset(data, targets, trsf, self.use_path, with_raw, with_noise)
    def resize_data(self, data, size=(224, 224)):
      
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")

        if data.ndim == 3:  # 单张图像 [H, W, C]
            data = np.expand_dims(data, axis=0)  # [1, H, W, C]
        elif data.ndim != 4:
            raise ValueError(f"Expected 3D or 4D image data, got {data.ndim}D")

        # 确保通道在正确位置
        if data.shape[-1] in [1, 3]:  # [N, H, W, C]
            data = data.transpose(0, 3, 1, 2)  # [N, C, H, W]

        resized_data = []
        for img in data:
            img = Image.fromarray(img.transpose(1, 2, 0))  # [C, H, W] -> [H, W, C]
            img = img.resize(size, Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
            resized_data.append(img_array)
        return np.array(resized_data)
    def preprocess_data(self, idata, save_dir="data/preprocessed"):
        os.makedirs(save_dir, exist_ok=True)
        npy_path = os.path.join(save_dir, f"{idata.__class__.__name__}_train.npy")
        
        if os.path.exists(npy_path):
            print(f"从 {npy_path} 加载预处理数据")
            return np.load(npy_path)
        else:
            print(f"预处理 {idata.__class__.__name__} 的数据")
            if idata.use_path:
                train_data = self.load_images(idata.train_data)
            else:
                train_data = self.resize_data(idata.train_data, size=(224, 224))
            np.save(npy_path, train_data)
            return train_data



    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf_list[self.current_task], *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf_list[self.current_task], *self._common_trsf])

        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)
    
    def _setup_data(self):
        self._train_data = []
        self._train_targets = []
        self._test_data = []
        self._test_targets = []
        self._class_order = []
        self._increments = []
        current_class_offset = 0
        self._train_trsf_list = []
        self._test_trsf_list = []

        
        for dataset_name in self.dataset_names:
            idata = _get_idata(dataset_name)
            idata.download_data()
            train_data, train_targets = idata.train_data, idata.train_targets
            test_data, test_targets = idata.test_data, idata.test_targets
            
    
             
            if idata.use_path:
               
                train_data = self.load_images(train_data)
                test_data = self.load_images(test_data)
           
            if not isinstance(train_data, np.ndarray):
               raise ValueError(f"Train data for {dataset_name} must be a numpy array")
            if not isinstance(train_targets, np.ndarray):
               raise ValueError(f"Train targets for {dataset_name} must be a numpy array")
                
            # 统一图像尺寸为 (224, 224)
            train_data = self.resize_data(train_data, size=(224, 224))
            test_data = self.resize_data(test_data, size=(224, 224))

            # 验证调整后的维度
            if train_data.ndim != 4:
                raise ValueError(f"Train data for {dataset_name} must be 4D after resize, got {train_data.ndim}D")
            if train_targets.ndim > 2:
                raise ValueError(f"Train targets for {dataset_name} must be 1D or 2D, got {train_targets.ndim}D")


            train_targets = train_targets + current_class_offset
            test_targets = test_targets + current_class_offset

            self._train_data.append(train_data)
            self._train_targets.append(train_targets)
            self._test_data.append(test_data)
            self._test_targets.append(test_targets)

            num_classes = len(np.unique(train_targets))
            self._class_order.extend(list(range(current_class_offset, current_class_offset + num_classes)))
            self._increments.append(num_classes)
            current_class_offset += num_classes

            self._train_trsf_list.append(idata.train_trsf) 
            self._test_trsf_list.append(idata.test_trsf)    
            self._common_trsf = idata.common_trsf
            self.use_path = idata.use_path


         # 合并前检查维度一致性
        for i, data in enumerate(self._train_data):
             if data.ndim != 4:
                raise ValueError(f"Train data at index {i} must be 4D, got {data.ndim}D")

        self._train_data = np.concatenate(self._train_data, axis=0)
        self._train_targets = np.concatenate(self._train_targets, axis=0)
        self._test_data = np.concatenate(self._test_data, axis=0)
        self._test_targets = np.concatenate(self._test_targets, axis=0)

     
        self.use_path = False
        
        if self.shuffle:
            np.random.seed(self.seed)
            self._class_order = np.random.permutation(self._class_order).tolist()

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, with_raw=False, with_noise=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.with_raw = with_raw
        if use_path and with_raw:
            self.raw_trsf = transforms.Compose([transforms.Resize((500, 500)), transforms.ToTensor()])
        else:
            self.raw_trsf = transforms.Compose([transforms.ToTensor()])
        if with_noise:
            class_list = np.unique(self.labels)
            self.ori_labels = deepcopy(labels)
            for cls in class_list:
                random_target = class_list.tolist()
                random_target.remove(cls)
                tindx = [i for i, x in enumerate(self.ori_labels) if x == cls]
                for i in tindx[:round(len(tindx)*0.2)]:
                    self.labels[i] = random.choice(random_target)
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            load_image = pil_loader(self.images[idx])
            image = self.trsf(load_image)
        else:
            #load_image = Image.fromarray(self.images[idx])
             # 假设 images 是 [N, C, H, W]，转换为 [H, W, C]
            load_image = Image.fromarray(self.images[idx].transpose(1, 2, 0))
            image = self.trsf(load_image)
        label = self.labels[idx]
        if self.with_raw:
            return idx, image, label, self.raw_trsf(load_image) 
        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar100':
        return iCIFAR100()
    elif name == 'cifar100_224':
        return iCIFAR100_224()
    elif name == 'imagenet1000':
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "imagenet-r":
        return iImageNetR()
    elif name == 'cub200_224':
        return iCUB200_224()
    elif name == 'resisc45':
        return iResisc45_224()
    elif name == 'cars196_224':
        return iCARS196_224()
    elif name == 'sketch345_224':
        return iSketch345_224()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
