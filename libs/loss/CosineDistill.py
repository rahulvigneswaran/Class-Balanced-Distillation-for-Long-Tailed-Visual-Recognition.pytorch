import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import libs.utils.globals as g



class CosineDistill(nn.Module):
    def __init__(self, beta=100):
        super(CosineDistill, self).__init__()
        self.beta = beta
        
    def forward(self, student, teacher):
        cos = nn.CosineSimilarity(dim=1)
        return self.beta*(1-cos(student, teacher)).mean(dim=0)

def create_loss(*args):
    print("Loading Cosine Distance Loss.")
    return CosineDistill(*args)
