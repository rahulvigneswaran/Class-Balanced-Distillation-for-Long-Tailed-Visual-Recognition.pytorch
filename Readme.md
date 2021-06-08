# Unoffical Pytorch Implementation of [Class-Balanced Distillation for Long-Tailed Visual Recognition](https://arxiv.org/abs/2104.05279) by Ahmet Iscen, André Araujo, Boqing Gong, Cordelia Schmid
---
### Note:
    - Implemented only for ImageNetLT

## Things to do before you run :
- Change the `data_root` for your dataset in `main.py`.
- If you are using wandb logging (Weights & Biases), make sure to change the `wandb.init` in `main.py` accordingly.

## How to use?
- Easy to use : Check this script - `multi_runs.sh`
- Train the normal teachers :
```
python main.py --experiment=0.1 --seed=1 --gpu="0,1" --train --log_offline
```
- Train the augmentation teachers :
```
python main.py --experiment=0.2 --seed=1 --gpu="0,1" --train --log_offline
```
- Train the Class Balanced Distilled Student :
```
python main.py --experiment=0.3 --alpha=0.4 --beta=100 --seed=$seeds --gpu="0,1" --train --log_offline --normal_teacher="10,20" --aug_teacher="20,30"
```

### Arguments :
(General)
- `--seed`: Seed of your current run
- `--gpu`: GPUs to be used
- `--experiment`: Experiment number (Check `libs/utils/experiment_maker.py` for more details)
- `--wandb_logger`: Does wandb Logging
- `--log_offline`: Does offline Logging
- `--resume`: Resumes the training if the run crashes

(Specific to Distillation and Student's training)
- `--alpha`: Weightage between Classifier loss and distillation loss
- `--beta`: weightage for the Cosine Similarity between teachers and student
- `--normal_teachers`: What all seed of norma teachers do you want to use?
- `--aug_teachers`:  What all seed of augmented teachers do you want to use?

## Raise an issue :
If something is not clear or you found a bug, raise an issue!!
