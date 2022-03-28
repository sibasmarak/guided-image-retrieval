# Import required libraries
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_metric_learning import losses

class ContrastiveLoss(nn.Module):
	
	def __init__(self, temperature = 0.07):
		super(ContrastiveLoss, self).__init__()
		self.temperature = temperature

	def forward(self, image_features, text_features, labels):

		neg_mask = ~(labels.view(1, labels.shape[0]) == labels.view(labels.shape[0], 1))  # [B, B]
		pos_mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device=labels.device)  # [B, B]

		image_features = F.normalize(image_features, p = 2, dim = 1)
		text_features = F.normalize(text_features, p = 2, dim = 1)

		sim_scores = torch.exp(torch.matmul(image_features, text_features.T))

		neg_similarity = torch.sum(sim_scores * neg_mask, dim = 0)
		pos_similarity = torch.sum(sim_scores * pos_mask, dim = 0)

		return -torch.mean(torch.log(
			torch.div(pos_similarity, pos_similarity + neg_similarity)
		))