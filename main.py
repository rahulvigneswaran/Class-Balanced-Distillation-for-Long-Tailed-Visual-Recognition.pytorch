import os
import argparse
import pprint
import warnings
import yaml
import libs.utils.globals as g
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4048, rlimit[1]))

data_root = {
    "ImageNet": "/DATA/datasets/ImageNet",
}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--gpu", default="0,1,2,3", type=str)

parser.add_argument("--experiment", default=0.1, type=float)
parser.add_argument("--alpha", type=str, default="1")     # always make sure to convert to the desired type in experiment_maker
parser.add_argument("--beta", type=str, default="1")     # always make sure to convert to the desired type in experiment_maker
parser.add_argument("--normal_teachers", default=None, type=str)
parser.add_argument("--aug_teachers", default=None, type=str)

parser.add_argument("--wandb_logger", default=False, action="store_true")
parser.add_argument("--log_offline", default=True, action="store_true")
parser.add_argument("--resume", default=False, action="store_true", help="Will resume from the 'latest_model_checkpoint.pth'")

args = parser.parse_args()

# global configs 
g.wandb_log = args.wandb_logger
g.epoch_global = 0
g.log_offline = args.log_offline

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# custom
from libs.utils.experiments_maker import experiment_maker
from libs.data import dataloader
from libs.utils.utils import *

# Random Seed
import torch
import random
import numpy as np

print(f"=======> Using seed: {args.seed} <========")
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g.seed = args.seed
    
config = experiment_maker(args.experiment, data_root, normal_teacher=[int(s) for s in args.normal_teachers.split(',')] if args.normal_teachers else [], aug_teacher=[int(s) for s in args.aug_teachers.split(',')] if args.aug_teachers else [], seed=args.seed, custom_var1=args.alpha, custom_var2=args.beta)

if g.wandb_log:
    import wandb
    config_dictionary = config
    if args.resume: 
        id = torch.load(config["training_opt"]["log_dir"]+"/latest_model_checkpoint.pth")['wandb_id']
        print(f"\nResuming wandb id: {id}!\n")
    else:
        id = wandb.util.generate_id()
        print(f"\nStarting wandb id: {id}!\n")
    wandb.init(
        project="long-tail", 
        entity="long-tail",
        reinit=True,
        name=f"{config['training_opt']['stage']}",
        allow_val_change=True,
        save_code=True,
        config=config_dictionary,
        tags=config["wandb_tags"],
        id=id,
        resume="allow",
    )  
    wandb.config.update(args, allow_val_change=True)
    config["wandb_id"] = id
else: 
    config["wandb_id"] = None

if not os.path.isdir(config["training_opt"]["log_dir"]):
    os.makedirs(config["training_opt"]["log_dir"])
# else: 
#     raise Exception("Directory already exists!!")

g.log_dir = config["training_opt"]["log_dir"]
if g.log_offline:
    if not os.path.isdir(f"{g.log_dir}/metrics"):
        os.makedirs(f"{g.log_dir}/metrics")
    

splits = ["train", "val"]

data = {
    x: dataloader.load_data(
        data_root=data_root[config["training_opt"]["dataset"].rstrip("_LT")],
        dataset=config["training_opt"]["dataset"],
        phase=x,
        batch_size=config["training_opt"]["batch_size"],
        sampler_dic=get_sampler_dict(config["training_opt"]["sampler"]),
        num_workers=config["training_opt"]["num_workers"],
        special_aug=config["training_opt"]["special_aug"] if "special_aug" in config["training_opt"] else False,
    )
    for x in splits
}
# Number of samples in each class
config["training_opt"]["data_count"] = data["train"].dataset.img_num_list

# import appropriate core
if "core" in config:
    training_model = source_import(config["core"]).get_core(config, data)
else:
    from libs.core.stage_2 import model
    training_model = model(config, data, test=False)        

# training sequence
print("\nInitiating training sequence!")
if args.resume:
    training_model.resume_run(config["training_opt"]["log_dir"]+"/latest_model_checkpoint.pth")
training_model.train()

print("=" * 25, " ALL COMPLETED ", "=" * 25)
