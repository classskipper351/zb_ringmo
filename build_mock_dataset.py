import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

# 常量：图像归一化的均值和标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def _check_pretrain_dataset_config(args):
    """检查并设置预训练数据集配置的默认值"""
    if not hasattr(args, 'arch'):
        args.arch = 'ringmo'  # 默认架构
    args.mask_patch_size = getattr(args, 'patch_dim', 32)  # 映射参数名
    args.batch_size = getattr(args, 'micro_batch_size', 16)
    assert args.img_h == args.img_w, "假设图像为正方形"
    args.image_size = args.img_h
    args.mask_ratio = getattr(args, 'mask_factor', 0.6)
    args.inside_ratio = getattr(args, 'inside_ratio', 0.6)
    args.use_lbp = getattr(args, 'use_lbp', False)

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
            raise NotImplementedError("LBP is not supported in mock dataset")
        out = out + (torch.from_numpy(mask).float(),)
        return out

class MockImageLoader(Dataset):
    """Mock dataset for generating random images."""
    def __init__(self, num_samples=1000, image_size=224, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        # 模拟图像路径列表（仅用于占位）
        self.image_paths = [f"mock_image_{i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 生成随机图像张量，形状为 (3, image_size, image_size)
        image = torch.randn(3, self.image_size, self.image_size)
        # 为了兼容 transform，将张量转换为 PIL 图像
        if self.transform:
            image = image.mul(0.5).add(0.5).clamp(0, 1)  # 缩放到 [0, 1]
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        return image

def worker_init_fn(worker_id):
    """设置每个 worker 的随机种子"""
    base_seed = torch.initial_seed() % (2**32)  # 强制 base_seed 在有效范围内
    seed = (base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)

def create_pretrain_dataset(args):
    """Create a mock dataset for self-supervised training."""
    # 检查配置
    _check_pretrain_dataset_config(args)
    seed = getattr(args, 'seed', 114514)

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建数据增强
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.image_size,
            scale=(getattr(args, 'crop_min', 0.08), 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=getattr(args, 'flip_prob', 0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # 创建虚假数据集
    dataset = MockImageLoader(
        num_samples=getattr(args, 'num_samples', 1000),  # 可通过 args 控制数据集大小
        image_size=args.image_size,
        transform=transform
    )

    # 支持分布式训练
    sampler = DistributedSampler(
        dataset,
        shuffle=False,  # 保持顺序采样
        seed=seed  # 确保所有进程使用相同种子
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 必须为 False 以支持管道并行
        num_workers=args.num_workers,
        collate_fn=lambda batch: {
            'image': torch.stack([item for item in batch]),
            'mask': torch.stack([
                MaskPolicyForPIMask(
                    input_size=args.image_size,
                    mask_patch_size=args.mask_patch_size,
                    mask_ratio=args.mask_ratio,
                    inside_ratio=args.inside_ratio,
                    use_lbp=False
                )(item)[1] for item in batch
            ])
        },
        worker_init_fn=worker_init_fn
    )

    return dataloader