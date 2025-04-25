import argparse

from datasets.base import Dataset
from datasets.generation import get_dataset


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        augment_flip = False,
        convert_image_to = None,
        condition = 0,
        equalizeHist = False,
        crop_patch = False,
        sample = False,
        generation = False
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.augment_flip = augment_flip
        self.convert_image_to = convert_image_to
        self.condition = condition
        self.equalizeHist = equalizeHist
        self.crop_patch = crop_patch
        self.sample = sample
        self.generation = generation
        
        # 获取图像文件
        self.paths = sorted([p for p in Path(f'{folder}').glob('**/*.PNG')])
        
        # 如果是生成模式，加载标签
        if self.generation:
            label_file = os.path.join(os.path.dirname(folder), 'labels.txt')
            with open(label_file, 'r') as f:
                self.labels = [int(line.strip()) for line in f.readlines()]
            
            # 确保图像和标签数量匹配
            assert len(self.paths) >= len(self.labels), \
                f"Number of images ({len(self.paths)}) must be >= number of labels ({len(self.labels)})"
            self.paths = self.paths[:len(self.labels)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)

        if self.convert_image_to is not None:
            img = img.convert(self.convert_image_to)

        if self.equalizeHist:
            img = ImageOps.equalize(img)

        if self.crop_patch:
            img, pad_size = self.random_crop_with_pad(img)
            self.pad_size_list.append(pad_size)

        if self.augment_flip and random() < 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self.condition:
            img = img.resize((self.image_size, self.image_size))
            img = resize_with_pad(img, self.image_size, self.image_size)
        else:
            img = img.resize((self.image_size, self.image_size))

        img = T.ToTensor()(img)
        
        # 如果是生成模式，返回图像和标签
        if self.generation:
            return img, self.labels[index]
        return img

# def dataset(folder,
#             image_size,
#             exts=['jpg', 'jpeg', 'png', 'tiff'],
#             augment_flip=False,
#             convert_image_to=None,
#             condition=0,
#             equalizeHist=False,
#             crop_patch=True,
#             sample=False, 
#             generation=False):
#     if generation:
#         # dataset_import = "generation"
#         # dataset = "CELEBA"
#         # args = {"exp": "xxx/dataset/diffusion_dataset"}

#         dataset_import = "base"
#     else:
#         dataset_import = "base"

#     if dataset_import == "base":
#         return Dataset(folder,
#                        image_size,
#                        exts=exts,
#                        augment_flip=augment_flip,
#                        convert_image_to=convert_image_to,
#                        condition=condition,
#                        equalizeHist=equalizeHist,
#                        crop_patch=crop_patch,
#                        sample=sample)
#     elif dataset_import == "generation":
#         if dataset == "CELEBA":
#             config = {
#                 "data": {
#                     "dataset": "CELEBA",
#                     "image_size": 64,  # 64
#                     "channels": 3,
#                     "logit_transform": False,
#                     "uniform_dequantization": False,
#                     "gaussian_dequantization": False,
#                     "random_flip": True,
#                     "rescaled": True,
#                 }}
#         elif dataset == "CIFAR10":
#             config = {
#                 "data": {
#                     "dataset": "CIFAR10",
#                     "image_size": 32,  # 32
#                     "channels": 3,
#                     "logit_transform": False,
#                     "uniform_dequantization": False,
#                     "gaussian_dequantization": False,
#                     "random_flip": True,
#                     "rescaled": True,
#                 }}
#         elif dataset == "bedroom":
#             config = {
#                 "data": {
#                     "dataset": "LSUN",
#                     "category": "bedroom",
#                     "image_size": 256,  # 256
#                     "channels": 3,
#                     "logit_transform": False,
#                     "uniform_dequantization": False,
#                     "gaussian_dequantization": False,
#                     "random_flip": True,
#                     "rescaled": True,
#                 }}
#         elif dataset == "church_outdoor":
#             config = {
#                 "data": {
#                     "dataset": "LSUN",
#                     "category": "church_outdoor",
#                     "image_size": 256,  # 256
#                     "channels": 3,
#                     "logit_transform": False,
#                     "uniform_dequantization": False,
#                     "gaussian_dequantization": False,
#                     "random_flip": True,
#                     "rescaled": True
#                 }}
#         args = dict2namespace(args)
#         config = dict2namespace(config)
#         return get_dataset(args, config)[0]
