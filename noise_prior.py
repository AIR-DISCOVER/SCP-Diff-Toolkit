### Noise Prior for Diffusion-based Semantic Synthesis

### Imports
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import argparse
from ldm.util import default
from cldm.model import create_model, load_state_dict
from torch.nn.functional import interpolate
import multiprocessing
from multiprocessing import Pool
from p_tqdm import p_map
from functools import partial

### constants
DATASETS = ["cityscapes", "ade20k", "coco-stuff"]

### functions for calculating statistics
def init_diffusion(config_path, ckpt_path):
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
    model = model.cuda()
    return model


def init_dataset(args):
    if args.dataset == "cityscapes":
        from datasets.cityscapes import CityscapesDataset
        dataset = CityscapesDataset()
    elif args.dataset == "ade20k":
        from datasets.ade20k import ADE20KDataset
        dataset = ADE20KDataset()
    elif args.dataset == "coco-stuff":
        from datasets.cocostuff import CocostuffDataset
        dataset = CocostuffDataset()
    else:
        raise NotImplementedError
    return dataset

def calculate_categorical_statistics(args, model, dataset):
    # Indexing stage. Iterate over the dataset. For each category, find the indices for corresponding images.
    # Return a dict with keys as category indices and values as lists of indices.
    # Each category has a list of indices, length of which is args.sample_size.

    # set seed
    random.seed(args.seed)
    final_dict = {} # category_index: 4x1 list of class_values
    length_dict = {} # category_index: length
    
    # random sample images from dataset
    choices = random.choices(range(len(dataset)), k=args.sample_size)
    
    for idx in tqdm(choices):
        term = dataset[idx]
        term['jpg'] = torch.tensor(term['jpg']).unsqueeze(0).cuda()
        term['hint'] = torch.tensor(term['hint']).unsqueeze(0).cuda()
        term['txt'] = [term['txt']]
        label = term['label']

        downsampled_size = (args.resolution[0] // 8, args.resolution[1] // 8)
        downsampled_label = interpolate(torch.tensor(label)[None, None, ...], size=downsampled_size, mode='nearest')[0][0]
        
        # extract all class indices
        class_indices = torch.unique(downsampled_label).tolist()
        
        for class_idx in class_indices:
            need_insertion = False
            
            # Check if the class index is already in the length_dict
            if class_idx in length_dict:
                # If the sample size for this class is not reached, add the index
                if length_dict[class_idx] < args.sample_size:
                    need_insertion = True
                    length_dict[class_idx] += 1
            else:
                # If the class index is not in the length_dict, add it and initialize in final_dict
                need_insertion = True
                length_dict[class_idx] = 1
                final_dict[class_idx] = []
        
            # insert indices into final_dict
            if need_insertion:
                x, c = model.get_input(term, 'jpg')
                # x is 4*H/8*W/8. sample corresponding pixels and add to final_dict[class_idx]
                sample_positions = torch.nonzero(downsampled_label == class_idx)
                sample_position = sample_positions[random.randint(0, len(sample_positions) - 1)]
                class_value = x[0, :, sample_position[0], sample_position[1]]
                final_dict[class_idx].append(class_value.cpu())
        
    # save statistics to file `args.save_dir/args.dataset/args.save_name`
    os.makedirs(os.path.join(args.save_dir, args.dataset), exist_ok=True)
    
    # concatenate class_values for each class
    final_dict_for_save = {}
    for class_idx in final_dict:
        final_dict_for_save[class_idx] = torch.stack(final_dict[class_idx], dim=0).numpy()

    # save raw statistics to file `args.save_dir/args.dataset/args.save_name`
    np.save(os.path.join(args.save_dir, args.dataset, args.save_name + "_categorical_raw.npy"), final_dict_for_save)
    
    # calculate mean and std for each class
    final_dict_for_save = {}
    for class_idx in final_dict:
        final_dict_for_save[class_idx] = {
            "mean": torch.stack(final_dict[class_idx], dim=0).mean(dim=0).numpy(),
            "std": torch.stack(final_dict[class_idx], dim=0).std(dim=0).numpy()
        }
    np.save(os.path.join(args.save_dir, args.dataset, args.save_name + "_categorical_mean_std.npy"), final_dict_for_save)


def calculate_spatial_statistics(args, model, dataset):
    # set seed
    random.seed(args.seed)
    sample_indices = random.choices(range(len(dataset)), k=args.sample_size)
    
    x_list = []
    # calculate statistics in blocks
    for idx in tqdm(sample_indices):
        term = dataset[idx]
        term['jpg'] = torch.tensor(term['jpg']).unsqueeze(0).cuda()
        term['hint'] = torch.tensor(term['hint']).unsqueeze(0).cuda()
        term['txt'] = [term['txt']]
    
        x, c = model.get_input(term, 'jpg')
        
        x_list.append(x.cpu())
        
    x = torch.cat(x_list, dim=0)
    x = x.cpu().numpy()
    # save statistics to file `args.save_dir/args.dataset/args.save_name`
    os.makedirs(os.path.join(args.save_dir, args.dataset), exist_ok=True)
    
    raw_path = os.path.join(args.save_dir, args.dataset, args.save_name + "_spatial_raw.npy")
    np.save(raw_path, x)
    
    stat = {
        "mean": x.mean(axis=0),
        "std": x.std(axis=0)
    }
    mean_std_path = os.path.join(args.save_dir, args.dataset, args.save_name + "_spatial_mean_std.npy")
    np.save(mean_std_path, stat)
    

def calculate_spatial_class_statistics_init_dict(args):
    return {
        "mean": np.zeros((256, 4, args.resolution[0] // 8, args.resolution[1] // 8)).astype(np.float32),  # 256 means 256 classes
        "std": np.zeros((256, 4, args.resolution[0] // 8, args.resolution[1] // 8)).astype(np.float32),
        "count": np.zeros((256, 1, args.resolution[0] // 8, args.resolution[1] // 8)).astype(np.float32)
    }

def calculate_spatial_class_statistics_batch_calculate_dict(indices_iterable, args, model, dataset, epsilon=1e-6):
    final_dict = calculate_spatial_class_statistics_init_dict(args)
    for idx in tqdm(indices_iterable):
        term = dataset[idx]
        term['jpg'] = torch.tensor(term['jpg']).unsqueeze(0).cuda()
        term['hint'] = torch.tensor(term['hint']).unsqueeze(0).cuda()
        term['txt'] = [term['txt']]
        label = term['label']

        downsample_size = (args.resolution[0] // 8, args.resolution[1] // 8)
        downsampled_label = interpolate(torch.tensor(label)[None, None, ...], size=downsample_size, mode='nearest')[0][0]
        
        # extract all class indices
        class_indices = torch.unique(downsampled_label).tolist()
        x, c = model.get_input(term, 'jpg')
        
        for class_idx in class_indices:
            class_map = (downsampled_label == class_idx).cpu().numpy().astype(np.float32)
            current_x = x[0].cpu().numpy()
            # add class_map regions to final_dict
            new_mean = (1 * current_x * class_map[None, ...] + final_dict["count"][class_idx] * final_dict["mean"][class_idx] * class_map[None, ...]) / (final_dict["count"][class_idx] + class_map[None, ...] + epsilon) \
                + (1 - class_map)[None, ...] * final_dict["mean"][class_idx]
            final_dict["std"][class_idx] = np.sqrt(
                (final_dict["count"][class_idx] * (final_dict["std"][class_idx]**2) * class_map[None, ...] + \
                final_dict["count"][class_idx] * (final_dict["mean"][class_idx] - new_mean)**2 * class_map[None, ...] + \
                1 * (current_x - new_mean)**2 * class_map[None, ...]) / (final_dict["count"][class_idx] + class_map[None, ...] + epsilon)) + (1 - class_map)[None, ...] * final_dict["std"][class_idx]
            final_dict["mean"][class_idx] = new_mean
            final_dict["count"][class_idx] += class_map[None, ...]
        
    return final_dict

def calculate_spatial_class_statistics(args, model, dataset):
    # set seed
    random.seed(args.seed)
    epsilon = 1e-6
    batch_size = 64
    num_cpus = 16

    def merging_dicts(all_dicts):
        final_dict = calculate_spatial_class_statistics_init_dict(args)
        for d in all_dicts:
            new_mean = (d["mean"] * d["count"] + final_dict["mean"] * final_dict["count"]) / (d["count"] + final_dict["count"] + epsilon)
            final_dict["std"] = np.sqrt(
                (d["count"] * (d["std"]**2) + final_dict["count"] * (final_dict["std"]**2) + d["count"] * (d["mean"] - new_mean)**2 + final_dict["count"] * (final_dict["mean"] - new_mean)**2) / (d["count"] + final_dict["count"] + epsilon)
            )
            final_dict["mean"] = new_mean
            final_dict["count"] = d["count"] + final_dict["count"]
        return final_dict

    def parallel_processing(dataset, batch_size, num_cpus):
        # Create a list of all indices
        all_indices = list(range(len(dataset)))

        # Divide indices for parallel processing
        divided_indices = [all_indices[i::batch_size] for i in range(num_cpus)]

        # Set 'spawn' as the default method for starting a process
        multiprocessing.set_start_method('spawn', force=True)

        # Use a Pool to parallelize the calculation
        func = partial(calculate_spatial_class_statistics_batch_calculate_dict, args=args, model=model, dataset=dataset, epsilon=epsilon)
        with Pool(processes=num_cpus) as pool:
            all_dicts = pool.map(func, divided_indices)

        return all_dicts

    all_dicts = parallel_processing(dataset, batch_size, num_cpus)
    final_dict = merging_dicts(all_dicts)

    mean_std_path = os.path.join(args.save_dir, args.dataset, args.save_name + "_spatial_class_mean_std.npy")
    np.save(mean_std_path, final_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cityscapes", choices=DATASETS)
    parser.add_argument("--sample_size", type=int, default=10000, help="number of samples to use for statistics")
    parser.add_argument("--ddpm_steps", type=int, default=1000, help="number of steps for DDPM")
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generator")
    parser.add_argument("--save_name", type=str, help="name of the file to save the statistics")
    parser.add_argument("--config", type=str, default="./models/cldm_v21.yaml", help="path to the config file")
    parser.add_argument("--ckpt", type=str, help="path to the checkpoint file")
    parser.add_argument("--resolution", type=int, nargs=2, help="resolution of the images")
    parser.add_argument("--save_dir", type=str, default="./statistics", help="directory to save the statistics")

    args = parser.parse_args()
    
    assert args.sample_size <= 10000, "sample size must be less than 10000"
    
    model = init_diffusion(args.config, args.ckpt)

    dataset = init_dataset(args)

    calculate_categorical_statistics(args, model, dataset)
    calculate_spatial_statistics(args, model, dataset)
    calculate_spatial_class_statistics(args, model, dataset)
