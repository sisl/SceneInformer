import numpy as np
import torch
import importlib
import random
import os
from torch.utils.data import Dataset

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True #False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def my_collate(batch):
    processed_batch = {}
    for b in batch:
        if b is None: #If sample is broken return none
            #print(f'Skip')
            continue
        # else:
        #     print(f'Not skip')
        for key in b.keys():
            if key not in processed_batch.keys():
                processed_batch[key] = []
            processed_batch[key].append(b[key])

    for key in processed_batch.keys():
        if key == 'id':
            continue
        try:
            sizes = [elem.shape[0] for elem in processed_batch[key]]
            # print(f'Key:{key}. Sizes:{sizes}')
        except:
            # print(f'Broken key:{key}')
            raise KeyError
        max_size = max(sizes)

        if key == 'polylines':
            max_size_n_points = max([elem.shape[1] for elem in processed_batch[key]])

        for idx, elem in enumerate(processed_batch[key]):
            if len(elem.shape) == 2:
                if elem.shape[0] < max_size:
                    adjust_size = max_size - elem.shape[0]
                    elem = np.pad(elem, ((0,adjust_size), (0,0)), constant_values=((0,np.nan), (0,0)))
            elif key != 'polylines':
                if elem.shape[0] < max_size:
                    adjust_size = max_size - elem.shape[0]
                    elem = np.pad(elem, ((0,adjust_size), (0,0), (0,0)), constant_values=((0,np.nan), (0,0), (0,0)))
            else:
                adjust_size_n_lines = max_size - elem.shape[0]
                adjust_size_n_points = max_size_n_points - elem.shape[1]
                elem = np.pad(elem, ((0,adjust_size_n_lines), (0,adjust_size_n_points), (0,0)), constant_values=((0,np.nan), (0,np.nan), (0,0)))
 
            processed_batch[key][idx] = torch.from_numpy(elem)
            # print(f'Key:{key}. Elem:{elem.shape}')

        processed_batch[key] = torch.stack(processed_batch[key], dim=0)
    return processed_batch

def my_collate_multi_occlusion(batch):
    batch = batch[0]
    if batch is None:
        return None
    processed_batch = {}
    for b in batch:
        if b is None: #If sample is broken return none
            #print(f'Skip')
            continue
        # else:
        #     print(f'Not skip')
        for key in b.keys():
            if key not in processed_batch.keys():
                processed_batch[key] = []
            processed_batch[key].append(b[key])

    for key in processed_batch.keys():
        try:
            sizes = [elem.shape[0] for elem in processed_batch[key]]
            # print(f'Key:{key}. Sizes:{sizes}')
        except:
            # print(f'Broken key:{key}')
            raise KeyError
        max_size = max(sizes)

        if key == 'polylines':
            max_size_n_points = max([elem.shape[1] for elem in processed_batch[key]])

        for idx, elem in enumerate(processed_batch[key]):
            if len(elem.shape) == 2:
                if elem.shape[0] < max_size:
                    adjust_size = max_size - elem.shape[0]
                    elem = np.pad(elem, ((0,adjust_size), (0,0)), constant_values=((0,np.nan), (0,0)))
            elif key != 'polylines':
                if elem.shape[0] < max_size:
                    adjust_size = max_size - elem.shape[0]
                    elem = np.pad(elem, ((0,adjust_size), (0,0), (0,0)), constant_values=((0,np.nan), (0,0), (0,0)))
            else:
                adjust_size_n_lines = max_size - elem.shape[0]
                adjust_size_n_points = max_size_n_points - elem.shape[1]
                elem = np.pad(elem, ((0,adjust_size_n_lines), (0,adjust_size_n_points), (0,0)), constant_values=((0,np.nan), (0,np.nan), (0,0)))
 
            processed_batch[key][idx] = torch.from_numpy(elem)
            # print(f'Key:{key}. Elem:{elem.shape}')

        processed_batch[key] = torch.stack(processed_batch[key], dim=0)
    return processed_batch

def my_collate_with_cuda_mem_check(batch):
    processed_batch = {}
    b_actual = 0
    for b in batch:
        if b is None: #If sample is broken return none
            # print(f'Broken sample:{b}') #TODO: FIX
            continue
        b_actual += 1
        for key in b.keys():
            if key not in processed_batch.keys():
                processed_batch[key] = []
            processed_batch[key].append(b[key])
    
    estimated_cuda_mem = 0
    max_size_list = []

    for key in processed_batch.keys():
        sizes = [elem.shape[0] for elem in processed_batch[key]]
        max_size = max(sizes)
        max_size_list.append({key: max_size})
    
    estimated_cuda_mem = 0


    for key in processed_batch.keys():
        sizes = [elem.shape[0] for elem in processed_batch[key]]
        max_size = max(sizes)
        max_size_list.append({key: max_size})

        for idx, elem in enumerate(processed_batch[key]):
            if len(elem.shape) == 2:
                if elem.shape[0] < max_size:
                    adjust_size = max_size - elem.shape[0]
                    elem = np.pad(elem, ((0,adjust_size), (0,0)), constant_values=((0,np.nan), (0,0)))
            else:
                if elem.shape[0] < max_size:
                    adjust_size = max_size - elem.shape[0]
                    elem = np.pad(elem, ((0,adjust_size), (0,0), (0,0)), constant_values=((0,np.nan), (0,0), (0,0)))

            processed_batch[key][idx] = torch.from_numpy(elem)

        processed_batch[key] = torch.stack(processed_batch[key], dim=0)
        estimated_cuda_mem += processed_batch[key].element_size() * processed_batch[key].nelement()
        # print(f'Key:{key}. Size:{processed_batch[key].element_size()} N: {processed_batch[key].nelement()}')
        # Convert to GB 
    estimated_cuda_mem = estimated_cuda_mem / 1.e9
    # print(f'Len:{len(batch)}/{b_actual}. Estimated cuda memory usage: {estimated_cuda_mem:.2f} GB')

    if estimated_cuda_mem >= 0.09:
        b_new = int(0.085 / estimated_cuda_mem * b_actual)
        # print(f'Old:{b_actual}. New:{b_new}')
        indices = torch.randperm(b_actual)[:b_new]
        for key in processed_batch.keys():
            processed_batch[key] = processed_batch[key][indices]

    return processed_batch

from torch.utils.data.sampler import Sampler
class CustomSampler(Sampler):
    def __init__(self, data_source, data_ratio, replacement=False, num_samples=None, batch_size=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.data_ratio = data_ratio
        self.batch_size = batch_size


    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        self.positive_num_samples = int(self.data_ratio * self.num_samples)
        self.negative_num_samples = self.num_samples - self.positive_num_samples


        if self.replacement:
            raise NotImplementedError
            # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        sampling_list = torch.zeros(self.num_samples).int().tolist()
        list_postive_sample = torch.randint(high=self.data_source.num_positive_samples, 
                                            size=(self.positive_num_samples,), 
                                            dtype=torch.int64).tolist()

        list_negative_sample = torch.randint(high=self.data_source.num_positive_samples + self.data_source.num_negative_samples,
                                            low=self.data_source.num_positive_samples, 
                                            size=(self.negative_num_samples,), 
                                            dtype=torch.int64).tolist()
        sampling_list = list_postive_sample + list_negative_sample
        # random.shuffle(sampling_list) #TODO: FIX

        return iter(sampling_list)

    def __len__(self):
        return self.num_samples
    
def is_memory_exceeded(device: int):
    current_device = device if device is not None else torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    current_memory = torch.cuda.memory_allocated(current_device)
    
    return current_memory / total_memory > 0.6

def print_memory_usage(device: int):
    current_device = device if device is not None else torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    current_memory = torch.cuda.memory_allocated(current_device)
    print(f"Current memory usage: {current_memory / total_memory * 100:.2f}%")

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]