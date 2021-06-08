import yaml
import pprint
import libs.utils.globals as g
import torch


experiments = {
    0.1 : "BaselineCosineCE",     
    0.2 : "BaselineCosineCE_DifferentAug",
    0.3 : "ECBD",
}

def experiment_maker(experiment, data_root, normal_teacher=1, aug_teacher=None, seed=1, custom_var1="0", custom_var2="0"):
    """Creates an experiment and outputs an appropriate yaml file

    Args:
        experiment (float): Experiment of choice
        dataset (float): Dataset name
        data_root (dict): Dict of the root directories of all the datasets
        seed (int, optional): Which seed is being used ? Defaults to 1.
        custom_var1 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment
        custom_var2 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment

    Returns:
        [dictionary]: list of modified config files (length = number of experiments)
    """
    assert experiment in experiments.keys(), "Wrong Experiment!"

    # Load Default configuration
    with open("libs/utils/default_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    num_of_classes = 1000
    dataset_name = f"ImageNet_LT"
    exp_name_template = f'ImageNet'
        
    # Have a separate root folders and experiments names for all seeds except for seed 1
    if seed == 1:
        init_dir = "logs" 
    else:
        init_dir = f"logs/other_seeds/seed_{seed}"
        exp_name_template = f"seed_{seed}_{exp_name_template}"
        
    config["training_opt"]["num_classes"] = num_of_classes
    config["training_opt"]["dataset"] = dataset_name
    
    if experiment == 0.1: #BaselineCosineCE
        config["core"] = "./libs/core/BaselineCosineCE.py"
        
        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0
                
        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet50Feature.py"
        config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2 
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = False
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = None
        
        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        config["networks"]["classifier"]["optim_params"]["lr"] = 0.2 
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["classifier"]["scheduler_params"]["coslr"] = True
        config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None
        
        #delete
        del(config["criterions"]["PerformanceLoss"])
        del(config["criterions"]["EmbeddingLoss"])
        del(config["networks"]["embedding"])

        # force shuffle dataset
        config["shuffle"] = False   
        
        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]
        
        # other training configs
        config["training_opt"]["backbone"] = "resnet50"
        
        #------Effective batch size after considering GPU count and 
        #------gradient accumulation for GPU memory bottlenech is 512.
        #------64 samples per batch, accumulated over 8 iters for GPU memory bottleneck.
        #------Since DataParallel is used, to achieve the effective batchsize of 512, the 64 samples per batch
        #------is divided by the GPU count.
        config["training_opt"]["batch_size"] = int(64/int(torch.cuda.device_count()))   
        config["training_opt"]["accumulation_step"] = int(512/config["training_opt"]["batch_size"])
        
        config["training_opt"]["feature_dim"] = 2048
        config["training_opt"]["num_workers"] = 20
        config["training_opt"]["num_epochs"] = 90
        config["training_opt"]["sampler"] = False   
        
        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        config["training_opt"]["stage"] = exp_name
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{exp_name}'  
    
    elif experiment == 0.2: #BaselineCosineCE_DifferentAug
        config["core"] = "./libs/core/BaselineCosineCE.py"
        
        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0
                
        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet50Feature.py"
        config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2 
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = False
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = None
        
        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        config["networks"]["classifier"]["optim_params"]["lr"] = 0.2 
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["classifier"]["scheduler_params"]["coslr"] = True
        config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None
        
        #delete
        del(config["criterions"]["PerformanceLoss"])
        del(config["criterions"]["EmbeddingLoss"])
        del(config["networks"]["embedding"])

        # force shuffle dataset
        config["shuffle"] = False   
        
        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]
        
        # other training configs
        config["training_opt"]["backbone"] = "resnet50"
        
        #------Effective batch size after considering GPU count and 
        #------gradient accumulation for GPU memory bottlenech is 512.
        #------64 samples per batch, accumulated over 8 iters for GPU memory bottleneck.
        #------Since DataParallel is used, to achieve the effective batchsize of 512, the 64 samples per batch
        #------is divided by the GPU count.
        config["training_opt"]["batch_size"] = int(64/int(torch.cuda.device_count()))   
        config["training_opt"]["accumulation_step"] = int(512/config["training_opt"]["batch_size"])
        
        config["training_opt"]["feature_dim"] = 2048
        config["training_opt"]["num_workers"] = 20
        config["training_opt"]["num_epochs"] = 90
        config["training_opt"]["sampler"] = False  
        config["training_opt"]["special_aug"] = True
        
        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'
        config["training_opt"]["stage"] = exp_name
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{exp_name}'   
        
    
    elif experiment == 0.3: #ECBD_BaselineCosineCE
        config["core"] = "./libs/core/CBDistillCE.py"
        
        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0 - float(custom_var1)

        # Distill loss (Just doing cosine distance between teacher and student features)
        config["criterions"]["DistillLoss"] = {}
        config["criterions"]["DistillLoss"]["def_file"] = "./libs/loss/CosineDistill.py"
        config["criterions"]["DistillLoss"]["loss_params"] = {}
        config["criterions"]["DistillLoss"]["loss_params"]["beta"] = float(custom_var2)
        config["criterions"]["DistillLoss"]["optim_params"] = False
        config["criterions"]["DistillLoss"]["weight"] = float(custom_var1)
                
        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet50Feature.py"
        config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2 
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = False
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = None
        
        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        config["networks"]["classifier"]["optim_params"]["lr"] = 0.2 
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["classifier"]["scheduler_params"]["coslr"] = True
        config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None
        
        config["training_opt"]["backbone"] = "resnet50"
        for i,j in zip(range(len(normal_teacher)), normal_teacher):
            
            exp_name_template_t = f'ImageNet'
            seed_t = j 
            # Have a separate root folders and experiments names for all seeds except for seed 1
            if seed_t == 1:
                init_dir_t = "logs" 
            else:
                init_dir_t = f"logs/other_seeds/seed_{seed_t}"
                exp_name_template_t = f"seed_{seed_t}_{exp_name_template_t}"
                
            config["networks"][f"normal_t{i}_model"] = {}
            config["networks"][f"normal_t{i}_model"]["trainable"] = True
            config["networks"][f"normal_t{i}_model"]["def_file"] = "./libs/models/ResNet50Feature.py"
            config["networks"][f"normal_t{i}_model"]["optim_params"] = {}
            config["networks"][f"normal_t{i}_model"]["optim_params"]["lr"] = 0.2 
            config["networks"][f"normal_t{i}_model"]["optim_params"]["momentum"] = 0.9
            config["networks"][f"normal_t{i}_model"]["optim_params"]["weight_decay"] = 0.0005 
            config["networks"][f"normal_t{i}_model"]["scheduler_params"] = {}
            config["networks"][f"normal_t{i}_model"]["scheduler_params"]["coslr"] = True
            config["networks"][f"normal_t{i}_model"]["scheduler_params"]["endlr"] = 0.0
            config["networks"][f"normal_t{i}_model"]["scheduler_params"]["step_size"] = 30
            config["networks"][f"normal_t{i}_model"]["params"] = {}
            config["networks"][f"normal_t{i}_model"]["params"]["pretrain"] = True
            config["networks"][f"normal_t{i}_model"]["params"]["pretrain_dir"] = f'./{init_dir_t}/{dataset_name}/{experiments[0.1]}_{exp_name_template_t}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
            config["networks"][f"normal_t{i}_model"]["fix"] = True
        
        for i,j in zip(range(len(aug_teacher)), aug_teacher):
            
            exp_name_template_t = f'ImageNet'
            seed_t = j 
            # Have a separate root folders and experiments names for all seeds except for seed 1
            if seed_t == 1:
                init_dir_t = "logs" 
            else:
                init_dir_t = f"logs/other_seeds/seed_{seed_t}"
                exp_name_template_t = f"seed_{seed_t}_{exp_name_template_t}"
                
            config["networks"][f"aug_t{i}_model"] = {}
            config["networks"][f"aug_t{i}_model"]["trainable"] = True
            config["networks"][f"aug_t{i}_model"]["def_file"] = "./libs/models/ResNet50Feature.py"
            config["networks"][f"aug_t{i}_model"]["optim_params"] = {}
            config["networks"][f"aug_t{i}_model"]["optim_params"]["lr"] = 0.2 
            config["networks"][f"aug_t{i}_model"]["optim_params"]["momentum"] = 0.9
            config["networks"][f"aug_t{i}_model"]["optim_params"]["weight_decay"] = 0.0005 
            config["networks"][f"aug_t{i}_model"]["scheduler_params"] = {}
            config["networks"][f"aug_t{i}_model"]["scheduler_params"]["coslr"] = True
            config["networks"][f"aug_t{i}_model"]["scheduler_params"]["endlr"] = 0.0
            config["networks"][f"aug_t{i}_model"]["scheduler_params"]["step_size"] = 30
            config["networks"][f"aug_t{i}_model"]["params"] = {}
            config["networks"][f"aug_t{i}_model"]["params"]["pretrain"] = True
            config["networks"][f"aug_t{i}_model"]["params"]["pretrain_dir"] = f'./{init_dir_t}/{dataset_name}/{experiments[0.2]}_{exp_name_template_t}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
            config["networks"][f"aug_t{i}_model"]["fix"] = True
        
        if (len(normal_teacher) + len(aug_teacher)) > 1 :
            config["networks"]["ecbd_converter"] = {}
            config["networks"]["ecbd_converter"]["trainable"] = True
            config["networks"]["ecbd_converter"]["def_file"] = "./libs/models/ecbd_converter.py"
            config["networks"]["ecbd_converter"]["optim_params"] = {}
            config["networks"]["ecbd_converter"]["optim_params"]["lr"] = 0.2 
            config["networks"]["ecbd_converter"]["optim_params"]["momentum"] = 0.9
            config["networks"]["ecbd_converter"]["optim_params"]["weight_decay"] = 0.0005 
            config["networks"]["ecbd_converter"]["scheduler_params"] = {}
            config["networks"]["ecbd_converter"]["scheduler_params"]["coslr"] = True
            config["networks"]["ecbd_converter"]["scheduler_params"]["endlr"] = 0.0
            config["networks"]["ecbd_converter"]["scheduler_params"]["step_size"] = 30
            config["networks"]["ecbd_converter"]["params"] = {}
            config["networks"]["ecbd_converter"]["params"]["feat_in"] = config["networks"]["classifier"]["params"]["feat_dim"]
            config["networks"]["ecbd_converter"]["params"]["feat_out"] = config["networks"]["classifier"]["params"]["feat_dim"]*(len(normal_teacher) + len(aug_teacher))
        
        #delete
        del(config["criterions"]["PerformanceLoss"])
        del(config["criterions"]["EmbeddingLoss"])
        del(config["networks"]["embedding"])

        # force shuffle dataset
        config["shuffle"] = False   
        
        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]
        
        # other training configs
        config["training_opt"]["backbone"] = "resnet50"
        
        #------Effective batch size after considering GPU count and 
        #------gradient accumulation for GPU memory bottlenech is 512.
        #------64 samples per batch, accumulated over 8 iters for GPU memory bottleneck.
        #------Since DataParallel is used, to achieve the effective batchsize of 512, the 64 samples per batch
        #------is divided by the GPU count.
        config["training_opt"]["batch_size"] = int(64/int(torch.cuda.device_count()))   
        config["training_opt"]["accumulation_step"] = int(512/config["training_opt"]["batch_size"])
        
        config["training_opt"]["feature_dim"] = 2048
        config["training_opt"]["num_workers"] = 20
        config["training_opt"]["num_epochs"] = 90
        
        config["training_opt"]["sampler"] = {"def_file": "./libs/samplers/ClassAwareSampler.py", "num_samples_cls": 4, "type": "ClassAwareSampler"}

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'
        config["training_opt"]["stage"] = exp_name
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{exp_name}/alpha_{float(custom_var1)},beta_{float(custom_var2)}_normal_k_{len(normal_teacher)}_aug_k_{len(aug_teacher)}'   
    
    else:
        print(f"Wrong experiments setup!-{experiment}")
    
    config["training_opt"]["num_epochs"] = 2
    config["training_opt"]["accumulation_step"] = int(128/config["training_opt"]["batch_size"])
    
    if g.log_offline:
        g.log_dir = config["training_opt"]["log_dir"]
    return config
 
