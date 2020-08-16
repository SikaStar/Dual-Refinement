import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math

# Spread loss
class SpreadLoss(nn.Module):
    def __init__(self, num_features, num_classes, init_em, m=0.35,knn=6):
        super(SpreadLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.m = m  # margin for spread loss
        self.knn = knn  # Knn for neighborhood invariance
        #  instant memory bank
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))
        self.em.data = torch.tensor(init_em).float().to(self.device)

    def forward(self, inputs, targets, epoch=None):
        norm_em = F.normalize(self.em)
        exemplars_feats = norm_em[targets]
        examplars = torch.sum(inputs*exemplars_feats,dim=1,keepdim=True)
        sim_mat = inputs.mm(norm_em.t())
        _, index_sorted = torch.sort(sim_mat, dim=1, descending=True)
        neighbors = torch.gather(sim_mat, 1, index_sorted)[:, 0:self.knn]
        negatives = torch.gather(sim_mat, 1, index_sorted)[:, self.knn:-1]
        postives = torch.cat((examplars, neighbors), dim=1)

        triplets = torch.exp(negatives.unsqueeze(2)-postives.unsqueeze(1)+self.m)
        batch_size = inputs.shape[0]
        sum_triplets = torch.sum(triplets.view(batch_size, -1), dim=1, keepdim=True)
        spread_loss = torch.mean(torch.log(1.0 + sum_triplets))

        return spread_loss
