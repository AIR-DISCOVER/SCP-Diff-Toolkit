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
from cldm.ddim_hacked import DDIMSampler
import einops
from PIL import Image

### constants
DATASETS = ["cityscapes", "ade20k", "coco-stuff"]

def get_statistics_result(args):
    return {
            "spatial": os.path.join(args.stat_dir, args.dataset, args.stat_name + "_spatial_mean_std.npy"),
            "categorical": os.path.join(args.stat_dir, args.dataset, args.stat_name + "_categorical_mean_std.npy"),
            "spatial_raw": os.path.join(args.stat_dir, args.dataset, args.stat_name + "_spatial_raw.npy"),
            "categorical_raw": os.path.join(args.stat_dir, args.dataset, args.stat_name + "_categorical_raw.npy"),
            "spatial_class_joint": os.path.join(args.stat_dir, args.dataset, args.stat_name + "_spatial_class_mean_std.npy"),
        }

PROMPT = {
    "cityscapes": "Realistic city road scenes",
    "ade20k": "diverse images depicting various scenes",
    "coco-stuff": "Realistic image",
}


### inference function
def sample_spatial_prior(args, model, dataset, term, prior_dict, return_mean_std_map=False, noise=None):
    # prior sampling
    spatial_prior = prior_dict['spatial'].item()
    spatial_prior_mean = torch.tensor(spatial_prior['mean']).to(model.device)
    spatial_prior_std = torch.tensor(spatial_prior['std']).to(model.device)
    
    if return_mean_std_map:
        return spatial_prior_mean, spatial_prior_std
    
    if noise is None:
        noise = torch.randn((args.batch_size, 4, args.resolution[0] // 8, args.resolution[1] // 8)).to(model.device)
    final_prior_map = noise * spatial_prior_std + spatial_prior_mean
    return final_prior_map

def sample_class_prior(args, model, dataset, term, prior_dict, return_mean_std_map=False, noise=None):
    class_prior = prior_dict['class'].item()
    # get label map for the image
    label_map = term["label"]
    downsampled_label = interpolate(torch.tensor(label_map)[None, None, ...], size=(args.resolution[0] // 8, args.resolution[1] // 8), mode='nearest')[0][0]
    
    # set all keys that not in class_prior.keys() to 255
    downsampled_label[~torch.isin(downsampled_label.long(), torch.tensor(list(class_prior.keys())))] = 255
    
    # assemble class map
    if noise is None:
        noise = torch.randn((args.batch_size, 4, args.resolution[0] // 8, args.resolution[1] // 8)).to(model.device)
    label_indices = torch.unique(downsampled_label).tolist()
    label_means = {k: class_prior[k]['mean'][None, :, None, None] for k in label_indices}
    label_stds = {k: class_prior[k]['std'][None, :, None, None] for k in label_indices}
    label_mask_maps = {k: (downsampled_label == k).float()[None, None, ...].to(model.device) for k in label_indices}
    
    mean_map = sum([torch.tensor(label_means[k]).to(model.device) * label_mask_maps[k] for k in label_indices])
    std_map = sum([torch.tensor(label_stds[k]).to(model.device) * label_mask_maps[k] for k in label_indices])
    
    if return_mean_std_map:
        return mean_map, std_map
    
    gaussian_map = noise * std_map + mean_map
    return gaussian_map

def sample_spatial_class_prior(args, model, dataset, term, prior_dict, return_mean_std_map=False, noise=None, fallback="spatial"):
    joint_prior = prior_dict['spatial_class_joint'].item()
    
    # get label map for the image
    label_map = term["label"]
    downsampled_label = interpolate(torch.tensor(label_map)[None, None, ...], size=(args.resolution[0] // 8, args.resolution[1] // 8), mode='nearest')[0][0]
    
    # assemble map
    if noise is None:
        noise = torch.randn((args.batch_size, 4, args.resolution[0] // 8, args.resolution[1] // 8)).to(model.device)
    
    label_indices = torch.unique(downsampled_label).tolist()
    label_spatial_mean_maps = {class_id: joint_prior['mean'][class_id] for class_id in label_indices}
    label_spatial_std_maps = {class_id: joint_prior['std'][class_id] for class_id in label_indices}
    # label_spatial_maps = {class_id: noise * torch.tensor(label_spatial_std_maps[class_id]).to(model.device) + torch.tensor(label_spatial_mean_maps[class_id]).to(model.device) for class_id in label_indices}
    label_count_maps = {class_id: joint_prior['count'][class_id] for class_id in label_indices}
    label_mask_maps = {k: (downsampled_label == k).float()[None, None, ...].to(model.device) for k in label_indices}
    
    mean_map = sum([torch.tensor(label_spatial_mean_maps[k]).to(model.device) * label_mask_maps[k] for k in label_indices])
    std_map = sum([torch.tensor(label_spatial_std_maps[k]).to(model.device) * label_mask_maps[k] for k in label_indices])
    count_map = sum([torch.tensor(label_count_maps[k]).to(model.device) * label_mask_maps[k] for k in label_indices])
    
    # if count map reaches 0, use fallback
    replace_region = (count_map == 0).float()
    if replace_region.sum() == 0:
        new_mean_map = mean_map
        new_std_map = std_map
        
    else:
        if fallback == "spatial":
            replace_prior = prior_dict['spatial'].item()
            replace_prior_mean = torch.tensor(replace_prior['mean']).to(model.device)
            replace_prior_std = torch.tensor(replace_prior['std']).to(model.device)
        elif fallback == "class":
            replace_prior_mean, replace_prior_std = sample_class_prior(args, model, dataset, term, prior_dict, return_mean_std_map=True)
        else:
            raise NotImplementedError
    
        new_mean_map = replace_region * replace_prior_mean + (1 - replace_region) * mean_map
        new_std_map = replace_region * replace_prior_std + (1 - replace_region) * std_map
    
    if return_mean_std_map:
        return new_mean_map, new_std_map

    gaussian_map = noise * new_std_map + new_mean_map
    return gaussian_map
    

def sample_prior(args, model, dataset, term, prior_dict):
    if args.prior_type == "normal":
        assert args.diffusion_steps == 999, "diffusion_steps must be 999 for normal prior"
        return torch.randn((args.batch_size, 4, args.resolution[0] // 8, args.resolution[1] // 8)).to(model.device)
    
    elif args.prior_type == "spatial":
        
        final_prior_map = sample_spatial_prior(args, model, dataset, term, prior_dict)

        # add noise
        noise = torch.randn_like(final_prior_map)
        diffusion_step = torch.tensor([args.diffusion_steps]).long().to(model.device)
        x_noisy = model.q_sample(x_start=final_prior_map, t=diffusion_step, noise=noise)
        return x_noisy
    
    elif args.prior_type == "class":

        final_reduced_map = sample_class_prior(args, model, dataset, term, prior_dict)
        
        # add noise
        noise = torch.randn_like(final_reduced_map)
        diffusion_step = torch.tensor([args.diffusion_steps]).long().to(model.device)
        x_noisy = model.q_sample(x_start=final_reduced_map, t=diffusion_step, noise=noise)
        return x_noisy
    
    elif args.prior_type == "spatial-class":
        # prior construction
        spatial_mean, spatial_std = sample_spatial_prior(args, model, dataset, term, prior_dict, return_mean_std_map=True)
        class_mean, class_std = sample_class_prior(args, model, dataset, term, prior_dict, return_mean_std_map=True)

        alpha = args.spatial_scale
        beta = args.class_scale
        new_mean = alpha * spatial_mean + beta * class_mean
        new_std = torch.sqrt(1 / (alpha + beta) * (alpha * spatial_std ** 2 + beta * class_std ** 2 + alpha * (spatial_mean - new_mean) ** 2 + beta * (class_mean - new_mean) ** 2))
        final_reduced_map = torch.randn_like(new_mean) * new_std + new_mean
        
        # add noise
        noise = torch.randn((args.batch_size, 4, args.resolution[0] // 8, args.resolution[1] // 8)).to(model.device)
        diffusion_step = torch.tensor([args.diffusion_steps]).long().to(model.device)
        x_noisy = model.q_sample(x_start=final_reduced_map, t=diffusion_step, noise=noise)
        return x_noisy

    elif args.prior_type == "spatial-class-joint":
        
        final_reduced_map = sample_spatial_class_prior(args, model, dataset, term, prior_dict, fallback="spatial")
        
        # add noise
        noise = torch.randn_like(final_reduced_map)
        diffusion_step = torch.tensor([args.diffusion_steps]).long().to(model.device)
        x_noisy = model.q_sample(x_start=final_reduced_map, t=diffusion_step, noise=noise)
        return x_noisy
    
    elif args.prior_type == "class-spatial-joint":
        
        final_reduced_map = sample_spatial_class_prior(args, model, dataset, term, prior_dict, fallback="class")
        
        # add noise
        noise = torch.randn_like(final_reduced_map)
        diffusion_step = torch.tensor([args.diffusion_steps]).long().to(model.device)
        x_noisy = model.q_sample(x_start=final_reduced_map, t=diffusion_step, noise=noise)
        return x_noisy

    elif args.prior_type == "x0":
        
        term['jpg'] = torch.tensor(term['jpg']).unsqueeze(0).cuda()
        term['hint'] = torch.tensor(term['hint']).unsqueeze(0).cuda()
        term['txt'] = [term['txt']]
        
        x, c = model.get_input(term, 'jpg')
        noise = torch.randn((args.batch_size, *x.shape[1:])).to(x.device)
        diffusion_step = torch.tensor([args.diffusion_steps]).long().to(x.device)
        x_noisy = model.q_sample(x_start=x, t=diffusion_step, noise=noise)
        return x_noisy



def inference(args, model, dataset):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # inference hyperparameters
    ddim_sampler = DDIMSampler(model)
    device = ddim_sampler.model.betas.device
    num_samples = args.batch_size
    ddim_steps = 20
    scale = args.scale
    eta = 0.0
    strength = 1.0
    prompt = PROMPT[args.dataset]
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    
    # load prior
    statistic_result = get_statistics_result(args)
    prior_dict = {}
    prior_dict['spatial'] = np.load(statistic_result['spatial'], allow_pickle=True)
    prior_dict['class'] = np.load(statistic_result['categorical'], allow_pickle=True)
    prior_dict['spatial_raw'] = np.load(statistic_result['spatial_raw'], allow_pickle=True)
    prior_dict['class_raw'] = np.load(statistic_result['categorical_raw'], allow_pickle=True)
    prior_dict['spatial_class_joint'] = np.load(statistic_result['spatial_class_joint'], allow_pickle=True)
    
    # prepare output path `SAVE_DIR/args.dataset/args.save_name/{real/sample/gt_label}/{000000,000001,...}.png`
    save_path = os.path.join(args.save_dir, args.dataset, f"{args.dataset}_{args.sample_size}_{args.diffusion_steps}_{args.prior_type}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "real"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "sample"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "gt_label"), exist_ok=True)
    
    ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta)
    
    with torch.no_grad():

        current_samples = 0
        sample_indices = random.choices(range(len(dataset)), k=args.sample_size // num_samples)
        
        for idx in tqdm(sample_indices):
            term = dataset[idx]
    
            control = torch.stack([torch.tensor(term['hint']).cuda() for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            
            H, W = control.shape[2:]
            shape = (4, H // 8, W // 8)
            C, H, W = shape
            size = (num_samples, C, H, W)
            
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            
            model.control_scales = ([strength] * 13)

            x_start = sample_prior(args, model, dataset, term, prior_dict)
            
            # find the indices where ddim_sampler.ddim_timesteps > args.diffusion_steps
            if args.diffusion_steps == 999:
                sample_index = len(ddim_sampler.ddim_timesteps)
            else:
                sample_index = torch.where(torch.tensor(ddim_sampler.ddim_timesteps) >= args.diffusion_steps)[0][0] + 2
            
            samples, intermediates = ddim_sampler.ddim_sampling(cond, size, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond, x_T=x_start, timesteps=sample_index)            

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            
            for b in x_samples:
                # save sampled
                Image.fromarray(b).save(f"{save_path}/sample/{current_samples:06d}.png")
                # save real (soft link)
                os.symlink(term['orig_img_path'], f"{save_path}/real/{current_samples:06d}.png")
                # save gt_label
                os.symlink(term['orig_label_path'], f"{save_path}/gt_label/{current_samples:06d}.png")
                current_samples += 1


### functions for calculating statistics
def init_diffusion(args):
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location='cpu'))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cityscapes", choices=DATASETS)
    parser.add_argument("--sample_size", type=int, default=50000, help="number of samples to use for statistics")
    parser.add_argument("--diffusion_steps", type=int, default=999, help="number of steps for DDPM")
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generator")
    parser.add_argument("--save_dir", type=str, default="./temp", help="directory to save results")
    parser.add_argument("--resolution", type=int, nargs=2, help="resolution of the images")
    parser.add_argument("--ckpt", type=str, help="checkpoint to use for model")
    parser.add_argument("--config", type=str, help="config file to use for model", default="./models/cldm_v21.yaml")
    parser.add_argument("--stat_dir", type=str, default="./statistics", help="directory to save statistics")
    parser.add_argument("--stat_name", type=str, help="name of statistics")
    parser.add_argument("--scale", type=float, default=7.5, help="scale for unconditional guidance")
    parser.add_argument("--prior_type", type=str, default="normal", choices=["normal", "spatial", "class", "spatial-class", "spatial-class-joint", "class-spatial-joint", "x0"], help="type of prior to use")
    parser.add_argument("--spatial_scale", type=float, default=1.00, help="scale for spatial prior")
    parser.add_argument("--class_scale", type=float, default=1.00, help="scale for class prior")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size for inference")

    args = parser.parse_args()

    assert args.sample_size <= 50000, "sample size must be less than 50000"
    
    model = init_diffusion(args)
    dataset = init_dataset(args)
    
    inference(args, model, dataset)