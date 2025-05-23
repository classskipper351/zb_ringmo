# This file is from https://github.com/NVIDIA/gpu_affinity/tree/main/gpu_affinity
# We copy the file here to work around the version conflict of pynvml.
#
# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import enum
import functools
import itertools
import operator
import os
import pathlib
import re
from typing import Optional

import pynvml


class GPUAffinityError(Exception):
    pass


class StrEnum(enum.Enum):
    def __str__(self):
        return self.value


class Mode(StrEnum):
    ALL = 'all'
    SINGLE = 'single'
    SINGLE_UNIQUE = 'single_unique'
    UNIQUE_INTERLEAVED = 'unique_interleaved'
    UNIQUE_CONTIGUOUS = 'unique_contiguous'


class Scope(StrEnum):
    NODE = 'node'
    SOCKET = 'socket'


class Multithreading(StrEnum):
    ALL_LOGICAL = 'all_logical'
    SINGLE_LOGICAL = 'single_logical'


class Device:
    # assume nvml returns list of 64 bit ints
    _nvml_bit_affinity = 64

    _nvml_affinity_elements = (
        os.cpu_count() + _nvml_bit_affinity - 1
    ) // _nvml_bit_affinity

    def __init__(self, device_idx):
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(os.environ['CUDA_VISIBLE_DEVICES'])
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            if device_idx >= len(visible_devices):
                msg = (
                    f'Requested device_idx={device_idx} is out of bounds for '
                    f'the specified CUDA_VISIBLE_DEVICES={visible_devices}'
                )
                raise GPUAffinityError(msg)
            try:
                # interpret elements of CUDA_VISIBLE_DEVICES as integer indices
                remapped_device_ids = [int(id) for id in visible_devices]
                device_idx = remapped_device_ids[device_idx]
            except ValueError:
                # interpret elements of CUDA_VISIBLE_DEVICES as UUID strings
                num_devices = pynvml.nvmlDeviceGetCount()
                devices_uuids = [
                    pynvml.nvmlDeviceGetUUID(handle)
                    for handle in [
                        pynvml.nvmlDeviceGetHandleByIndex(idx)
                        for idx in range(num_devices)
                    ]
                ]
                indices_with_matching_prefixes = [
                    i
                    for i in range(num_devices)
                    if devices_uuids[i].startswith(visible_devices[device_idx])
                ]
                if len(indices_with_matching_prefixes) == 0:
                    msg = (
                        f'None of the devices matched the specified UUID '
                        f'prefix, prefix={visible_devices[device_idx]}, '
                        f'devices={devices_uuids}'
                    )
                    raise GPUAffinityError(msg)
                elif len(indices_with_matching_prefixes) > 1:
                    msg = (
                        f'Multiple devices matched the specified UUID '
                        f'prefix, prefix={visible_devices[device_idx]}, '
                        f'devices={devices_uuids}'
                    )
                    raise GPUAffinityError(msg)
                device_idx = indices_with_matching_prefixes[0]

        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        except Exception as ex:
            msg = f'Unable to get NVML handle for device {device_idx}'
            raise GPUAffinityError(msg) from ex

    def get_name(self):
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_uuid(self):
        return pynvml.nvmlDeviceGetUUID(self.handle)

    def get_cpu_affinity(self, scope):
        if scope == Scope.SOCKET:
            nvml_scope = pynvml.NVML_AFFINITY_SCOPE_SOCKET
        elif scope == Scope.NODE:
            nvml_scope = pynvml.NVML_AFFINITY_SCOPE_NODE
        else:
            raise GPUAffinityError('Unknown affinity scope')

        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinityWithinScope(
            self.handle, Device._nvml_affinity_elements, nvml_scope
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = '{:064b}'.format(j) + affinity_string

        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]
        return ret


def get_thread_siblings_list():
    r'''
    Returns a list of 2-element integer tuples representing pairs of
    hyperthreading cores.
    '''
    path = '/sys/devices/system/cpu/cpu*/topology/thread_siblings_list'
    thread_siblings_list = []
    pattern = re.compile(r'(\d+)\D(\d+)')
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(sorted(map(int, res[0])))
                thread_siblings_list.append(pair)
    thread_siblings_list = list(set(thread_siblings_list))
    return thread_siblings_list


def build_thread_siblings_dict(siblings_list):
    siblings_dict = {}
    for siblings_tuple in siblings_list:
        for core in siblings_tuple:
            siblings_dict[core] = siblings_tuple

    return siblings_dict


def group_list_by_key(the_list, key):
    sorted_list = sorted(the_list, key=key)
    grouped = [
        tuple(group) for key, group in itertools.groupby(sorted_list, key=key)
    ]
    return grouped


def group_by_siblings(affinities):
    siblings_list = get_thread_siblings_list()
    siblings_dict = build_thread_siblings_dict(siblings_list)
    siblings_key = lambda x: siblings_dict.get(x, (x,))
    affinities = [
        tuple(group_list_by_key(affinity, key=siblings_key))
        for affinity in affinities
    ]
    return affinities


def group_by_node(socket_affinities, node_affinities):
    socket_node_assigned_cores = collections.defaultdict(list)
    for socket, node_cores in zip(socket_affinities, node_affinities):
        socket_node_assigned_cores[socket].extend(node_cores)

    socket_node_assigned_cores = {
        key: tuple(sorted(set(value)))
        for key, value in socket_node_assigned_cores.items()
    }

    node_grouping = collections.defaultdict(list)

    for socket_cores, assigned_cores in socket_node_assigned_cores.items():
        unassigned_cores = sorted(
            list(set(socket_cores) - set(assigned_cores))
        )

        for assigned_core in assigned_cores:
            node_grouping[assigned_core].append(assigned_core)

        for assigned, unassigned in zip(
            itertools.cycle(assigned_cores), unassigned_cores
        ):
            node_grouping[assigned].append(unassigned)

    node_grouping = {key: tuple(value) for key, value in node_grouping.items()}

    grouped_affinities = [
        tuple(node_grouping[item] for item in node_affinity)
        for node_affinity in node_affinities
    ]
    return grouped_affinities


def ungroup_by_nodes(affinities, scope):
    if scope == Scope.SOCKET:
        affinities = [
            list(itertools.chain(*zip(*affinity))) for affinity in affinities
        ]
    elif scope == Scope.NODE:
        affinities = [
            [group[0] for group in affinity] for affinity in affinities
        ]
    return affinities


def ungroup_by_siblings(affinities, multithreading):
    if multithreading == Multithreading.ALL_LOGICAL:
        affinities = [
            list(itertools.chain(*affinity)) for affinity in affinities
        ]
    elif multithreading == Multithreading.SINGLE_LOGICAL:
        affinities = [
            [group[0] for group in affinity] for affinity in affinities
        ]
    else:
        raise GPUAffinityError('Unknown multithreading mode')
    return affinities


def check_core_count(
    affinities, min_physical_cores=1, max_physical_cores=None
):
    for gpu_id, affinity in enumerate(affinities):
        if len(affinity) < min_physical_cores:
            raise GPUAffinityError(
                f'Number of available physical cores for GPU {gpu_id} is less '
                f'the predefinied minimum, '
                f'min_physical_cores={min_physical_cores}, '
                f'available physical cores: {affinity} (count={len(affinity)})'
            )

    if max_physical_cores is not None:
        affinities = [affinity[:max_physical_cores] for affinity in affinities]

    return affinities


def ungroup_all_and_check_count(
    affinities,
    scope,
    multithreading,
    min_physical_cores=1,
    max_physical_cores=None,
):
    affinities = ungroup_by_nodes(affinities, scope)
    affinities = check_core_count(
        affinities, min_physical_cores, max_physical_cores
    )
    affinities = ungroup_by_siblings(affinities, multithreading)
    return affinities


def check_affinities(affinities):
    if not len(affinities):
        raise GPUAffinityError(
            f'List of all affinities is empty: {affinities}'
        )

    for idx, affinity in enumerate(affinities):
        if not len(affinities):
            raise GPUAffinityError(f'Affinity {idx} is empty: {affinity}')

    # sets of cores should be either identical or disjoint
    for i, j in itertools.product(affinities, affinities):
        if not set(i) == set(j) and not set(i).isdisjoint(set(j)):
            raise GPUAffinityError(
                f'Sets of cores should be either identical or disjoint, '
                f'but got {i} and {j}.'
            )


def get_affinities(nproc_per_node, scope, exclude_unavailable_cores=True):
    devices = [Device(i) for i in range(nproc_per_node)]
    affinities = [dev.get_cpu_affinity(scope) for dev in devices]

    if exclude_unavailable_cores:
        available_cores = os.sched_getaffinity(0)
        affinities = [
            sorted(list(set(affinity) & available_cores))
            for affinity in affinities
        ]

    check_affinities(affinities)

    return affinities


def get_grouped_affinities(nproc_per_node, exclude_unavailable_cores=True):
    socket_affinities = get_affinities(
        nproc_per_node, Scope.SOCKET, exclude_unavailable_cores
    )
    node_affinities = get_affinities(
        nproc_per_node, Scope.NODE, exclude_unavailable_cores
    )

    sibling_socket_affinities = group_by_siblings(socket_affinities)
    sibling_node_affinities = group_by_siblings(node_affinities)

    grouped_affinities = group_by_node(
        sibling_socket_affinities, sibling_node_affinities
    )

    return grouped_affinities


def get_all(
    nproc_per_node,
    scope,
    multithreading,
    min_physical_cores,
    max_physical_cores,
):
    r'''
    The process is assigned with all available physical CPU cores recommended
    by pynvml for the GPU with a given id.

    Assignment automatically includes available hyperthreading siblings if
    multithreading=Multithreading.ALL_LOGICAL.

    Args:
        nproc_per_node (int): number of processes per node
        scope (Scope): scope for retrieving affinity from pynvml
        multithreading (Multithreading): multithreading mode
    '''
    affinities = get_affinities(nproc_per_node, scope)

    affinities = group_by_siblings(affinities)

    node_affinities = group_by_siblings(
        get_affinities(nproc_per_node, Scope.NODE)
    )
    all_node_affinities = functools.reduce(operator.add, node_affinities)

    affinities = [
        tuple(
            sorted(
                affinity,
                key=lambda x: (
                    0 if x in all_node_affinities else 1,
                    x,
                ),
            )
        )
        for affinity in affinities
    ]

    affinities = check_core_count(
        affinities, min_physical_cores, max_physical_cores
    )
    affinities = ungroup_by_siblings(affinities, multithreading)
    return affinities


def get_single(
    nproc_per_node,
    scope,
    multithreading,
    min_physical_cores=1,
    max_physical_cores=1,
):
    r'''
    The process is assigned with the first available physical CPU core from the
    list of all physical CPU cores recommended by pynvml for the GPU with a
    given id.

    Assignment automatically includes available hyperthreading siblings if
    multithreading=Multithreading.ALL_LOGICAL.

    Args:
        nproc_per_node: number of processes per node
        scope (Scope): scope for retrieving affinity from pynvml
        multithreading (Multithreading): multithreading mode
    '''
    grouped_affinities = get_grouped_affinities(nproc_per_node)
    ungrouped_affinities = ungroup_all_and_check_count(
        grouped_affinities,
        scope,
        multithreading,
        min_physical_cores,
        max_physical_cores,
    )
    return ungrouped_affinities


def get_single_unique(
    nproc_per_node,
    scope,
    multithreading,
    min_physical_cores=1,
    max_physical_cores=1,
):
    r'''
    The process is assigned with a single unique available physical CPU core
    from the list of all physical CPU cores recommended by pynvml for the GPU
    with a given id.

    Assignment automatically includes available hyperthreading siblings if
    multithreading=Multithreading.ALL_LOGICAL.

    Args:
        nproc_per_node: number of processes per node
        scope (Scope): scope for retrieving affinity from pynvml
        multithreading (Multithreading): multithreading mode
    '''
    grouped_affinities = get_grouped_affinities(nproc_per_node)

    affinities = []
    assigned_groups = set()

    for grouped_affinity in grouped_affinities:
        for group in grouped_affinity:
            if group not in assigned_groups:
                affinities.append([group])
                assigned_groups.add(group)
                break

    ungrouped_affinities = ungroup_all_and_check_count(
        affinities,
        scope,
        multithreading,
        min_physical_cores,
        max_physical_cores,
    )

    return ungrouped_affinities


def get_unique(
    nproc_per_node: int,
    scope: Scope,
    multithreading: Multithreading,
    mode: Mode,
    min_physical_cores: int,
    max_physical_cores: int,
    balanced: bool = True,
):
    r'''
    The process is assigned with a unique subset of available physical CPU
    cores from the list of all CPU cores recommended by pynvml for the GPU with
    a given id.

    Assignment automatically includes available hyperthreading siblings if
    multithreading=Multithreading.ALL_LOGICAL.

    Args:
        nproc_per_node (int): number of processes per node
        scope (Scope): scope for retrieving affinity from pynvml
        multithreading (Multithreading): multithreading mode
        mode (Mode): Mode.UNIQUE_CONTIGUOUS or Mode.UNIQUE_INTERLEAVED
        balanced (bool): assign an equal number of physical cores to each
            process
    '''
    grouped_affinities = get_grouped_affinities(nproc_per_node)

    grouped_affinities_to_device_ids = collections.defaultdict(list)

    for idx, grouped_affinity in enumerate(grouped_affinities):
        grouped_affinities_to_device_ids[tuple(grouped_affinity)].append(idx)

    # compute minimal number of physical cores per GPU across all GPUs and
    # sockets, code assigns this number of cores per GPU if balanced == True
    min_physical_cores_per_gpu = min(
        [
            len(cores) // len(gpus)
            for cores, gpus in grouped_affinities_to_device_ids.items()
        ]
    )

    grouped_unique_affinities = [None] * nproc_per_node

    for (
        grouped_affinity,
        device_ids,
    ) in grouped_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        if balanced:
            cores_per_device = min_physical_cores_per_gpu
            grouped_affinity = grouped_affinity[
                : devices_per_group * min_physical_cores_per_gpu
            ]
        else:
            cores_per_device = len(grouped_affinity) // devices_per_group

        for subgroup_id, device_id in enumerate(device_ids):
            if mode == Mode.UNIQUE_INTERLEAVED:
                unique_grouped_affinity = list(
                    grouped_affinity[subgroup_id::devices_per_group]
                )
            elif mode == Mode.UNIQUE_CONTIGUOUS:
                unique_grouped_affinity = list(
                    grouped_affinity[
                        subgroup_id
                        * cores_per_device : (subgroup_id + 1)
                        * cores_per_device
                    ]
                )
            else:
                raise GPUAffinityError('Unknown set_unique mode')

            grouped_unique_affinities[device_id] = unique_grouped_affinity

    ungrouped_affinities = ungroup_all_and_check_count(
        grouped_unique_affinities,
        scope,
        multithreading,
        min_physical_cores,
        max_physical_cores,
    )
    return ungrouped_affinities


def set_affinity(
    gpu_id: int,
    nproc_per_node: int,
    *,
    mode: Mode = Mode.UNIQUE_CONTIGUOUS,
    scope: Scope = Scope.NODE,
    multithreading: Multithreading = Multithreading.ALL_LOGICAL,
    balanced: bool = True,
    min_physical_cores: int = 1,
    max_physical_cores: Optional[int] = None,
):
    r'''
    The process is assigned with a proper CPU affinity that matches CPU-GPU
    hardware architecture on a given platform. Usually, setting proper affinity
    improves and stabilizes the performance of deep learning training
    workloads.

    This function assumes that the workload runs in multi-process single-device
    mode (there are multiple training processes, and each process is running on
    a single GPU). This is typical for multi-GPU data-parallel training
    workloads (e.g., using `torch.nn.parallel.DistributedDataParallel`).

    Available affinity modes:
    * `Mode.ALL` - the process is assigned with all available physical CPU
        cores recommended by pynvml for the GPU with a given id.
    * `Mode.SINGLE` - the process is assigned with the first available physical
        CPU core from the list of all physical CPU cores recommended by pynvml
        for the GPU with a given id (multiple GPUs could be assigned with the
        same CPU core).
    * `Mode.SINGLE_UNIQUE` - the process is assigned with a single unique
        available physical CPU core from the list of all CPU cores recommended
        by pynvml for the GPU with a given id.
    * `Mode.UNIQUE_INTERLEAVED` - the process is assigned with a unique subset
        of available physical CPU cores from the list of all physical CPU cores
        recommended by pynvml for the GPU with a given id, cores are assigned
        with interleaved indexing pattern
    * `Mode.UNIQUE_CONTIGUOUS` - (the default mode) the process is assigned
        with a unique subset of available physical CPU cores from the list of
        all physical CPU cores recommended by pynvml for the GPU with a given
        id, cores are assigned with contiguous indexing pattern

    `Mode.UNIQUE_CONTIGUOUS` is the recommended affinity mode for deep learning
    training workloads on NVIDIA DGX servers.

    Available affinity scope modes:
    * `Scope.NODE` - sets the scope for pynvml affinity queries to NUMA node
    * `Scope.SOCKET` - sets the scope for pynvml affinity queries to processor
        socket

    Available multithreading modes:
    * `Multithreading.ALL_LOGICAL` - assigns the process with all logical cores
        associated with a given corresponding physical core (i.e., it
        automatically includes all available hyperthreading siblings)
    * `Multithreading.SINGLE_LOGICAL` - assigns the process with only one
        logical core associated with a given corresponding physical core (i.e.,
        it excludes hyperthreading siblings)

    Args:
        gpu_id (int): index of a GPU, value from 0 to `nproc_per_node` - 1
        nproc_per_node (int): number of processes per node
        mode (gpu_affinity.Mode): affinity mode (default:
            `gpu_affinity.Mode.UNIQUE_CONTIGUOUS`)
        scope (gpu_affinity.Scope): scope for retrieving affinity from pynvml
            (default: `gpu_affinity.Scope.NODE`)
        multithreading (gpu_affinity.Multithreading): multithreading mode
            (default: `gpu_affinity.Multithreading.ALL_LOGICAL`)
        balanced (bool): assigns equal number of physical cores to each
            process, it affects only `gpu_affinity.Mode.UNIQUE_INTERLEAVED` and
            `gpu_affinity.Mode.UNIQUE_CONTIGUOUS` affinity modes (default:
            `True`)
        min_physical_cores (int): the intended minimum number of physical cores
            per process, code raises GPUAffinityError if the number of
            available cores is less than `min_physical_cores` (default: 1)
        max_physical_cores: the intended maxmimum number of physical cores per
            process, the list of assigned cores is trimmed to the first
            `max_physical_cores` cores if `max_physical_cores` is not None
            (default: `None`)

    Returns a set of logical CPU cores on which the process is eligible to run.

    WARNING: On NVIDIA DGX A100, only half of the CPU cores have direct access
    to GPUs. set_affinity with `scope=Scope.NODE` restricts execution only to
    the CPU cores directly connected to GPUs. On DGX A100, it will limit the
    execution to half of the CPU cores and half of CPU memory bandwidth (which
    may be fine for many DL models). Use `scope=Scope.SOCKET` to use all
    available DGX A100 CPU cores.

    WARNING: Intel's OpenMP implementation resets affinity on the first call to
    an OpenMP function after a fork. It's recommended to run with the following
    env variable: `KMP_AFFINITY=disabled` if the affinity set by
    `gpu_affinity.set_affinity` should be preserved after a fork (e.g. in
    PyTorch DataLoader workers).

    Example:

    import argparse
    import os

    import gpu_affinity
    import torch


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--local_rank',
            type=int,
            default=os.getenv('LOCAL_RANK', 0),
        )
        args = parser.parse_args()

        nproc_per_node = torch.cuda.device_count()

        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
        print(f'{args.local_rank}: core affinity: {affinity}')


    if __name__ == '__main__':
        main()

    Launch the example with:
    python -m torch.distributed.run --nproc_per_node <#GPUs> example.py
    '''
    if gpu_id >= nproc_per_node:
        msg = (
            f'gpu_id={gpu_id} should be smaller than '
            f'nproc_per_node={nproc_per_node}'
        )
        raise GPUAffinityError(msg)

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as ex:
        msg = 'Error while initializing pynvml'
        raise GPUAffinityError(msg) from ex

    if mode == Mode.ALL:
        affinity = get_all(
            nproc_per_node,
            scope,
            multithreading,
            min_physical_cores,
            max_physical_cores,
        )
    elif mode == Mode.SINGLE:
        affinity = get_single(nproc_per_node, scope, multithreading)
    elif mode == Mode.SINGLE_UNIQUE:
        affinity = get_single_unique(nproc_per_node, scope, multithreading)
    elif mode in {Mode.UNIQUE_CONTIGUOUS, Mode.UNIQUE_INTERLEAVED}:
        affinity = get_unique(
            nproc_per_node,
            scope,
            multithreading,
            mode,
            min_physical_cores,
            max_physical_cores,
            balanced,
        )
    else:
        raise GPUAffinityError('Unknown affinity mode')

    try:
        os.sched_setaffinity(0, affinity[gpu_id])
        new_affinity = os.sched_getaffinity(0)
    except OSError as ex:
        msg = f'Error while calling os affinity functions'
        raise GPUAffinityError(msg) from ex
    except AttributeError as ex:
        msg = f'os affinity functions are not available'
        raise GPUAffinityError(msg) from ex

    return new_affinity
