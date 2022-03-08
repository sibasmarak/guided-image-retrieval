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
# 		image_features_normalized = F.normalize(image_features, p=2, dim=1)
# 		text_features_normalized = F.normalize(text_features, p=2, dim=1)
# 		logits = torch.div(
# 			torch.matmul(
# 				image_features_normalized, torch.transpose(text_features_normalized, 0, 1)
# 			),
# 			self.temperature,
# 		)
		
# 		return losses.NTXentLoss(self.temperature)(logits, torch.squeeze(labels))


# class ContrastiveLoss(nn.Module):
	
# 	def __init__(self, temperature = 0.07):
# 		super(ContrastiveLoss, self).__init__()
# 		self.temperature = temperature

# 	def create_mask(self, labels):
# 		mask = torch.zeros((labels.shape[0], labels.shape[0])).to(labels.device)
# 		for i in range(labels.shape[0]):
# 			for j in range(i+1, labels.shape[0]):
# 				if labels[i] != labels[j]:
# 					mask[i][j] = 1
# 					mask[j][i] = 1
# 		return mask

# 	def forward(self, image_features, text_features, labels):
# 		"""
# 			x_1, x_1, x_1, x_2, x_2, x_3, x_3
# 			y_1, y_2, y_3, y_4, y_5, y_6, y_7
# 			1, 1, 1, 2, 2, 3, 3
# 			0, 1, 2, 3, 4, 5, 6

# 			2, 5
# 			7 x 7
# 			[0, 0, 0, 1, 1, 1, 1]
# 			[0, 0, 0, 1, 1, 1, 1]
# 			[0, 0, 0, 1, 1, 1, 1]

# 		"""
# 		mask = self.create_mask(labels) # bs x bs
# 		image_features = F.normalize(image_features, p=2, dim=1)
# 		text_features = F.normalize(text_features, p=2, dim=1)
# 		# bs x feat_s ===== bs x bs ==== bs x bs
# 		neg_similarity = torch.sum(torch.exp(torch.matmul(image_features, text_features.T) * mask), dim=0)
# 		pos_similarity = torch.zeros(image_features.shape[0]).to(labels.device)

# 		for i in range(image_features.shape[0]):
# 			pos_sim = torch.exp(torch.matmul(image_features[i].T, text_features[i]))
# 			neg_sim = neg_similarity[i]
# 			pos_similarity[i] = torch.log(torch.div(pos_sim, torch.add(pos_sim, neg_sim)))
# 			# print(pos_sim, neg_sim)

# 		return -torch.mean(pos_similarity, dim=0)


class ContrastiveLoss(nn.Module):
	
	def __init__(self, temperature = 0.07):
		super(ContrastiveLoss, self).__init__()
		self.temperature = temperature

	def create_negative_mask(self, labels):
		B = labels.size(0)
		labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
		# indices_equal = torch.eye(B, dtype=torch.bool).to(device = labels.device)  # [B, B]
		# return ~labels_equal & ~indices_equal  # [B, B]
		# return ~labels_equal

	def get_positive_mask(self, labels):
		B = labels.size(0)
		labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
		indices_equal = torch.eye(B, dtype=torch.bool).to(device=labels.device)  # [B, B]
		return labels_equal & ~indices_equal  # [B, B]

	def create_pos_neg_masks(self, labels):
		B = labels.size(0)
		labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
		indices_equal = torch.eye(B, dtype=torch.bool).to(device=labels.device)  # [B, B]
		# return (labels_equal & ~indices_equal), (~labels_equal & ~indices_equal)  # [B, B]
		return (indices_equal), (~labels_equal)  # [B, B]

	def create_mask(self, labels):
		mask = torch.zeros((labels.shape[0], labels.shape[0])).to(labels.device)
		for i in range(labels.shape[0]):
			for j in range(i+1, labels.shape[0]):
				if labels[i] != labels[j]:
					mask[i][j] = 1
					mask[j][i] = 1
		return mask

	def forward(self, image_features, text_features, labels):

		neg_mask = ~(labels.view(1, labels.shape[0]) == labels.view(labels.shape[0], 1))  # [B, B]
		pos_mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device=labels.device)  # [B, B]

		# neg_mask = torch.rand(labels.shape[0], labels.shape[0], device=labels.device, requires_grad=True)
		# pos_mask = torch.rand(labels.shape[0], labels.shape[0], device=labels.device, requires_grad=True)

		# pos_mask, neg_mask = self.create_pos_neg_masks(labels) # bs x bs
		image_features = F.normalize(image_features, p=2, dim=1)
		text_features = F.normalize(text_features, p=2, dim=1)
		# bs x feat_s ===== bs x bs ==== bs x bs
		# neg_similarity ===== bs x 1
		sim_scores = torch.exp(torch.matmul(image_features, text_features.T))
		# sim_scores = torch.rand(labels.shape[0], labels.shape[0], device=labels.device, requires_grad=True)

		neg_similarity = torch.sum(sim_scores * neg_mask, dim=0)
		# neg_similarity = torch.sum(torch.exp(torch.matmul(image_features, text_features.T) * neg_mask), dim=0)
		# pos_similarity = torch.zeros(image_features.shape[0]).to(labels.device)
		# bs x bs
		pos_similarity = torch.sum(sim_scores * pos_mask, dim = 0)
		# pos_similarity = torch.sum(torch.exp(torch.matmul(image_features, text_features.T)) * pos_mask, dim = 0)

		return -torch.mean(torch.log(
			torch.div(pos_similarity, pos_similarity + neg_similarity)
		))

		# Idea to use cross entropy to find noise contrastive loss
		# print(pos_similarity.shape)
		# print(neg_similarity.shape)
		# print(torch.zeros(pos_similarity.shape, dtype=torch.long, device=labels.device).shape)
		# return torch.nn.CrossEntropyLoss()(torch.div(pos_similarity, pos_similarity + neg_similarity), 
		# 									torch.zeros(pos_similarity.shape, dtype=torch.long, device=labels.device))

		# for i in range(image_features.shape[0]):
		# 	pos_sim = torch.exp(torch.matmul(image_features[i].T, text_features[i]))
		# 	neg_sim = neg_similarity[i]
		# 	pos_similarity[i] = torch.log(torch.div(pos_sim, torch.add(pos_sim, neg_sim)))
		# 	# print(pos_sim, neg_sim)



		# return -torch.mean(pos_similarity, dim=0)