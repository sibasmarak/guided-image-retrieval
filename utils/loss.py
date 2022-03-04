# Import required libraries
import torchmetrics
import pytorch_lightning as pl
from pytorch_metric_learning import losses
import torch.nn.functional as F
import torch.nn as nn
import torch

# class ContrastiveLoss(nn.Module):
# 	def __init__(self, temperature = 0.07):
# 		super(ContrastiveLoss, self).__init__()
# 		self.temperature = temperature

# 	def forward(self, image_features, text_features, labels):
# 		print(labels)
# 		image_features_normalized = F.normalize(image_features, p=2, dim=1)
# 		text_features_normalized = F.normalize(text_features, p=2, dim=1)
# 		logits = torch.div(
# 			torch.matmul(
# 				image_features_normalized, torch.transpose(text_features_normalized, 0, 1)
# 			),
# 			self.temperature,
# 		)
		
# 		return losses.NTXentLoss(self.temperature)(logits, torch.squeeze(labels))


class ContrastiveLoss(nn.Module):
	
	def __init__(self, temperature = 0.07):
		super(ContrastiveLoss, self).__init__()
		self.temperature = temperature

	def create_mask(self, labels):
		mask = torch.zeros((labels.shape[0], labels.shape[0])).to(labels.device)
		for i in range(labels.shape[0]):
			for j in range(i+1, labels.shape[0]):
				if labels[i] != labels[j]:
					mask[i][j] = 1
					mask[j][i] = 1
		return mask

	def forward(self, image_features, text_features, labels):

		mask = self.create_mask(labels)
		image_features = F.normalize(image_features, p=2, dim=1)
		text_features = F.normalize(text_features, p=2, dim=1)
		# print(mask)
		# print(labels)
		# neg_similarity = torch.sum(torch.exp(torch.matmul(torch.matmul(mask, image_features), text_features.T)))
		neg_similarity = torch.sum(torch.exp(torch.matmul(torch.matmul(text_features, image_features.T), mask)))
		# print("Neg similarity: ", neg_similarity)
		pos_similarity = torch.zeros(image_features.shape[0]).to(labels.device)
		for i in range(image_features.shape[0]):
			pos_sim = torch.exp(torch.matmul(image_features[i].T, text_features[i]))
			# print("Pos similarity: ", pos_sim, end = " ")
			pos_similarity[i] = torch.log(torch.div(pos_sim, torch.add(pos_sim, neg_similarity)))
			# print(pos_similarity[i])
			# exit(0)
		# print("\n")
		# print("Pos similarity: ", pos_similarity)
		return -torch.sum(pos_similarity)