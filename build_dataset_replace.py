import numpy as np
import torch
from skimage.feature import local_binary_pattern
import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List
from torch.utils.data.distributed import DistributedSampler 

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def _check_config_type(config):
    if not isinstance(config, dict):
        raise TypeError("dataset config should be dict type, but get {}".format(type(config)))

def _check_pretrain_dataset_config(args):
    """调整配置检查函数，直接使用args的属性"""
    # 确保必要的参数存在或设置默认值
    if not hasattr(args, 'arch'):
        args.arch = 'ringmo'  # 默认架构
    args.mask_patch_size = getattr(args, 'patch_dim', 32)  # 映射参数名
    args.batch_size = getattr(args, 'micro_batch_size', 16)
    assert args.img_h == args.img_w ,"假设图像为正方形"
    args.image_size = args.img_h  
    args.mask_ratio = getattr(args, 'mask_factor', 0.6)  # 根据实际需求调整
    args.inside_ratio = getattr(args, 'inside_ratio', 0.6)
    args.use_lbp = getattr(args, 'use_lbp', False)


def _check_finetune_dataset_config(config: dict):
    """check finetune dataset config"""
    _check_config_type(config)
    config.finetune_dataset.arch = config.arch

    if config.arch == "simmim" or config.arch == "ringmo" or config.arch == "ringmo_mm":
        config.finetune_dataset.mask_patch_size = config.model.mask_patch_size

    if config.train_config.batch_size:
        config.finetune_dataset.batch_size = config.train_config.batch_size

    if config.train_config.image_size:
        config.finetune_dataset.image_size = config.train_config.image_size

    if config.model.patch_size:
        config.finetune_dataset.patch_size = config.model.patch_size

    if config.device_num:
        config.finetune_dataset.device_num = config.device_num

    if config.local_rank is not None:
        config.finetune_dataset.local_rank = config.local_rank

    if config.train_config.num_classes:
        config.finetune_dataset.num_classes = config.train_config.num_classes




class MaskPolicyForPIMask:
    """Mask policy for PI mask in PyTorch."""
    def __init__(self, input_size=224, mask_patch_size=32, mask_ratio=0.6, inside_ratio=0.6, use_lbp=False):
        self.use_lbp = use_lbp
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.inside_count = mask_patch_size ** 2
        self.mask_pixel = int(np.ceil(self.inside_count * inside_ratio))

        assert self.input_size % self.mask_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2)
        return: (N, 1, H, W)
        """
        h = w = int(x.shape[1] ** 0.5)  # Calculate grid size
        p = self.mask_patch_size       # Patch size
        x = x.reshape(-1, h, w, p, p)  # Reshape to patches
        x = x.transpose(0, 1, 3, 2, 4)  # Rearrange dimensions
        return x.reshape(-1, h * p, w * p)  # Merge patches into full image

    def __call__(self, img):
        # Create a random mask for patches
        generator = torch.Generator().manual_seed(torch.initial_seed())
        mask_idx = torch.randperm(self.token_count, generator=generator)[:self.mask_count]
        mask_patch = torch.zeros(self.token_count, dtype=int)
        mask_patch[mask_idx] = 1

        # Generate pixel-wise mask inside patches
        mask_collect = np.ones((self.token_count, self.inside_count), dtype=int)
        for value, i in zip(mask_patch, range(self.token_count)):
            mask = np.zeros(self.inside_count, dtype=int)
            if value == 1:
                inside_id = np.random.permutation(self.inside_count)[:self.mask_pixel]
                mask[inside_id] = 1
            mask_collect[i] = mask

        mask_collect = mask_collect.reshape(1, self.token_count, self.inside_count)
        mask = self.unpatchify(mask_collect).astype(np.float32)

        out = (img,)
        if self.use_lbp:
            lbp_img = self.lbp(img)
            out = out + (lbp_img,)
        out = out + (torch.from_numpy(mask).float(),)
        return out

    def lbp(self, img):
        """Local Binary Pattern (LBP) for an image."""
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for processing
        lbps = []
        for channel in range(img.shape[2]):
            lbp_channel = local_binary_pattern(img[..., channel], P=8, R=2, method='uniform')
            lbps.append(np.expand_dims(lbp_channel, axis=2))
        lbp_img = np.concatenate(lbps, axis=2).transpose(2, 0, 1)  # Back to CHW format
        return torch.tensor(lbp_img, dtype=torch.float32)
    
class ImageLoader(Dataset):
    """ImageLoader for loading custom datasets."""
    def __init__(self, json_path, data_dir=None, transform=None):
        self.transform = transform
        # 加载图像路径列表
        #import pdb;pdb.set_trace()
        if len(json_path)> 1 and isinstance(json_path,List):
            train_json_path = json_path[0]
        else:
            train_json_path =  json_path
        with open(train_json_path, 'r') as f:
            
            self.image_paths = json.load(f)
        if data_dir is not None:
            self.image_paths = [os.path.join(data_dir, path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 加载图像并进行转换
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
def worker_init_fn(worker_id):
    # 设置每个 worker 的随机种子（需全局种子已固定）
    base_seed = torch.initial_seed() % (2**32)  # 强制 base_seed 在有效范围内
    # 生成 worker 的种子：base_seed + worker_id，再取模
    seed = (base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
'''
def build_mask(args, dataset):
    """Apply masking to the dataset and return batches as dicts."""
    mask_policy = MaskPolicyForPIMask(
        input_size=args.image_size,
        mask_patch_size=args.mask_patch_size,
        mask_ratio=args.mask_ratio,
        inside_ratio=args.inside_ratio,
        use_lbp=args.use_lbp
    )

    def apply_mask(data):
        image = data
        img, mask = mask_policy(image)  # mask 是 numpy 数组
        # 直接转换 mask 为 Tensor
        return {
            'image': image,  # image 已经是 Tensor（来自 ImageLoader 的 transform）
            'mask': torch.from_numpy(mask).float(),  # 强制转换 numpy 数组为 Tensor
        }
    
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False, #must be false during PP !
        collate_fn=lambda batch: {
            'image': torch.stack([apply_mask(torch.tensor(data))['image'] for data in batch]),
            'mask': torch.stack([
            apply_mask(torch.tensor(data) if isinstance(data, np.ndarray) else data)['mask']  for data in batch
   
            ])
        },
        num_workers=args.num_workers
        #pin_memory_device= 'cuda0'
    )


def build_transforms(args):
    """Build data augmentation transforms."""
    #import pdb;pdb.set_trace()
    transform_list = [
        transforms.RandomResizedCrop(
            args.image_size,
            scale=(args.crop_min, 1.0),
            ratio=(3. / 4., 4. / 3.)
        ),
        transforms.RandomHorizontalFlip(p=args.prop),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    return transforms.Compose(transform_list)
'''

class MockVisionDataset(Dataset):
    def __init__(self, num_samples, image_shape=(3, 192, 192), mask_shape=(192, 192)):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.mask_shape = mask_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'image': torch.randn(*self.image_shape, dtype=torch.float32),
            'mask': torch.randint(0, 2, self.mask_shape, dtype=torch.long)
        }


def mock_provider(train_valid_test_num_samples):
    train_ds = DataLoader( MockVisionDataset) 
    return train_ds ,None ,None


def create_pretrain_dataset(args):
    """Create dataset for self-supervised training."""
    # 创建数据增强
    
    _check_pretrain_dataset_config(args)
    #dataset_config = args.pretrain_dataset
    #
    # transforms = build_transforms(dataset_config)
    seed = getattr(args, 'seed', 114514)

    torch.manual_seed(seed )
    np.random.seed(seed )
    random.seed(seed )

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.image_size,
            scale=(getattr(args, 'crop_min', 0.08), 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=getattr(args, 'flip_prob',0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    #import pdb;pdb.set_trace()
    # 加载自定义数据集
    dataset = ImageLoader(
        json_path=args.data_path,  # 假设路径参数名
        data_dir=None,
        transform=transform
    )

    sampler = DistributedSampler(
        dataset,
        shuffle=False,  # 保持顺序采样
        seed=seed # 确保所有进程使用相同种子
    )

    # 应用 Mask 策略
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        #sampler = sampler,
        shuffle=False, ## must be False in PP
        num_workers=args.num_workers,
        collate_fn=lambda batch: {
            'image': torch.stack([item for item in batch]),
            'mask': torch.stack([
                MaskPolicyForPIMask(
                    input_size=args.image_size,
                    mask_patch_size=args.mask_patch_size,
                    mask_ratio=args.mask_ratio,
                    inside_ratio=args.inside_ratio
                )(item)[1] for item in batch
            ])
        },
        worker_init_fn=worker_init_fn
    )

    return dataloader
