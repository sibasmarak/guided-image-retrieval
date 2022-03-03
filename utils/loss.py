# Import required libraries
import torchmetrics
import pytorch_lightning as pl
from pytorch_metric_learning import losses
import torch.nn.functional as F
import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
	def __init__(self, temperature = 0.07):
		super(ContrastiveLoss, self).__init__()
		self.temperature = temperature

	def forward(self, image_features, text_features, labels):
		image_features_normalized = F.normalize(image_features, p=2, dim=1)
		text_features_normalized = F.normalize(text_features, p=2, dim=1)
		logits = torch.div(
			torch.matmul(
				image_features_normalized, torch.transpose(text_features_normalized, 0, 1)
			),
			self.temperature,
		)
		
		return losses.NTXentLoss(self.temperature)(logits, torch.squeeze(labels))