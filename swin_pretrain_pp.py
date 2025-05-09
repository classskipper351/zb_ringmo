from functools import partial
import os
import torch
from torch import nn
from megatron.core import parallel_state
from megatron.training import pretrain, get_args, get_timers, print_rank_0
from megatron.core.enums import ModelType
from pytorch_for_ringmo import SwinTransformerForRingMo, build_ringmo
#from build_dataset_replace import create_pretrain_dataset
from build_dataset_replace import create_pretrain_dataset
from megatron.core import mpu
import numpy as np
import random
import collections

USE_DTR = True if os.environ.get('DTR_ENABLE') == '1' else False
MEM_BUDGET = float(os.environ.get('MEM_BUDGET')) if os.environ.get('MEM_BUDGET') else 0
RECORD_MEM_SNAPSHOT = True if os.environ.get('RECORD_MEM_SNAPSHOT') == '1' else False
snapshot_filename = os.environ.get('SNAP_FILE_NAME')

class SwinDataLoaderStore:
    cache = collections.deque()

    @classmethod
    def push(cls, data_iterator, h2d_stream=False):
        timers = get_timers()
        timers('batch-generator', log_level=2).start()

        if h2d_stream:
            from megatron.core.pipeline_parallel.offload import get_offload_h2d_stream
            load_event = torch.cuda.Event()
            original_stream = torch.cuda.current_stream()
            with torch.cuda.stream(get_offload_h2d_stream()):
                data = cls.get_batch_on_this_tp_rank(data_iterator)
                for key in data:
                    if data[key] is not None:
                        data[key].record_stream(original_stream)
                load_event.record()
                cls.cache.append((data, load_event))
        else:
            data = cls.get_batch_on_this_tp_rank(data_iterator)
            cls.cache.append((data, None))

        timers('batch-generator').stop()

    @classmethod
    def pop(cls):
        data, load_event = cls.cache.popleft()
        if load_event:
            load_event.wait()
        return data

    @classmethod
    def get_batch_on_this_tp_rank(cls, data_iterator):
        args = get_args()

        def _broadcast(item):
            if item is not None:
                torch.distributed.broadcast(
                    item,
                    mpu.get_tensor_model_parallel_src_rank(),
                    group=mpu.get_tensor_model_parallel_group()
                )

        if mpu.get_tensor_model_parallel_rank() == 0:
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None

        return data

def _masked_mean(loss, mask):
    """Masked weighted mean"""
    weighted_loss = torch.mul(loss, mask)
    sum_loss = torch.sum(weighted_loss)
    sum_mask = torch.sum(mask) + 1e-5  # Prevent division by zero
    return sum_loss / sum_mask / 3  # in_chans

def ringmo_loss(x_rec, x_ori, mask, lbp=None, lbp_rec=None):
    """RingMo loss function"""
    loss_ori_recon = torch.abs(torch.sub(x_ori, x_rec))
    loss_ori_mask = _masked_mean(loss_ori_recon, mask)
    loss_lbp_mask = 0.
    return loss_ori_mask + loss_lbp_mask, {'recovery_loss': loss_ori_mask}

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    assert parallel_state.get_tensor_model_parallel_world_size() == 1 
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()

    pipeline_args = {
        'depths': [2, 2, 6, 2],  # Total 4 stages
        'num_heads': [4, 8, 16, 32],
    }

    total_layers = len(pipeline_args['depths'])
    assert total_layers % pipeline_size == 0, \
        f"Depth stages ({total_layers}) must be divisible by pipeline size ({pipeline_size})"

    per_layer = total_layers // pipeline_size
    start_idx = pp_rank * per_layer
    end_idx = start_idx + per_layer

    full_model_args = {
        'img_size': 192,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 0,
        'embed_dim': 128,
        'window_size': 6,
        'mlp_ratio': 4,
        'ape': False,
        'patch_norm': True,
        'pre_process': pre_process,
        'post_process': post_process,
        'depths': pipeline_args['depths'][start_idx:end_idx],
        'num_heads': pipeline_args['num_heads'][start_idx:end_idx],
        'encoder_stride': 32,
        'total_layers': total_layers,
        'pipeline_model_parallel_size': pipeline_size,
        'pipeline_dtype': torch.float16,
    }

    model = SwinTransformerForRingMo(**full_model_args)
    
    if not mpu.is_pipeline_first_stage():
        model.patch_embed = nn.Identity()
        model.pos_drop = nn.Identity()
    
    if mpu.is_pipeline_last_stage():
        model.decoder = nn.Conv2d(
            model.num_features, 
            (32**2)*3,
            kernel_size=1
        )
        model.pixelshuffle = nn.PixelShuffle(32)
    
    return model.cuda()

def forward_step(data_iterator, model):
    """Forward step function adapted for ZeroBubble with SwinDataLoaderStore"""
    timers = get_timers()
    timers('batch-generator', log_level=2).start()

    # Fetch data from SwinDataLoaderStore
    from collections.abc import Iterable
    if not isinstance(data_iterator, Iterable) and not data_iterator is None:  # isinstance(data_iterator, DataLoaderStore):
        data = data_iterator.pop()
    else:
        SwinDataLoaderStore.push(data_iterator, h2d_stream=False)
        data =  SwinDataLoaderStore.pop()
    timers('batch-generator').stop()

    # Extract data
    image = data["image"].cuda()
    mask = data["mask"].cuda()

    # Process input based on pipeline stage
    if parallel_state.is_pipeline_first_stage():
        x_output = model((image, mask))
    elif not parallel_state.is_pipeline_last_stage():
        x_output = model(None)
    else:
        x_output = model(None)

    # Return output and loss function
    if parallel_state.is_pipeline_last_stage():
        return x_output, partial(ringmo_loss, x_ori=image, mask=mask)
    else:
        return x_output, None

def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Dataset provider function"""
    print_rank_0("> building train, validation, and test datasets for ringmo ...")
    args = get_args()
    train_ds = create_pretrain_dataset(args)  # Returns a data loader
    print_rank_0("> finished creating ringmo datasets ...")
    return train_ds, None, None

if __name__ == "__main__":
    if USE_DTR:
        torch.init_dtb_manager()
        print('FlashDTR initialization succeed.')
        if MEM_BUDGET > 0:
            torch.set_memory_budget(int(MEM_BUDGET * 1e10))

    from torch.distributed.elastic.multiprocessing.errors.handlers import get_error_handler
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
    error_handler = get_error_handler()
    error_handler.initialize()

    try:
        if RECORD_MEM_SNAPSHOT:
            torch.cuda.memory._record_memory_history()
            
        pretrain(
            train_valid_test_dataset_provider=train_valid_test_dataset_provider,
            model_provider=model_provider,
            forward_step_func=forward_step,
            model_type=ModelType.encoder_or_decoder,
        )
        
        if RECORD_MEM_SNAPSHOT:
            local_rank = torch.distributed.get_rank()
            torch.cuda.memory._dump_snapshot(snapshot_filename + '_' + str(local_rank) + ".pickle")

    except Exception as e:
        print('[Exception]', str(e))
        if RECORD_MEM_SNAPSHOT:
            local_rank = torch.distributed.get_rank()
            torch.cuda.memory._dump_snapshot(snapshot_filename + '_' + str(local_rank) + ".pickle")
        raise