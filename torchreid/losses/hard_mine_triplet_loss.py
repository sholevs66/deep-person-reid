from __future__ import division, absolute_import
import torch
import torch.nn as nn
from ..metrics.distance import cosine_distance


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, distance='l2'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        N = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'l2':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(N, N)
            dist = dist + dist.t()
            dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
        else:
            dist = cosine_distance(inputs, inputs)
            dist = dist.clamp(min=1e-12)


        # For each anchor, find the hardest positive and negative
        mask = targets.expand(N, N).eq(targets.expand(N, N).t())
        dist_ap, dist_an = [], []
        for i in range(N):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
