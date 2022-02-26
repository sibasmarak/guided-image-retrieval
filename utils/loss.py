# Import required libraries
import torchmetrics
import pytorch_lightning as pl
from model import LanguageModel, VisionModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import losses
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
	def __init__(self, temperature = 0.07):
		super(SupervisedContrastiveLoss, self).__init__()
		self.temperature = temperature

	def forward(self, image_features, text_features):
        labels = torch.eye(image_features.shape[0])
		image_features_normalized = F.normalize(image_features, p=2, dim=1)
		text_features_normalized = F.normalize(text_features, p=2, dim=1)
		logits = torch.div(
			torch.matmul(
				image_features_normalized, torch.transpose(text_features_normalized, 0, 1)
			),
			self.temperature,
		)
		return losses.NTXentLoss(self.temperature)(logits, torch.squeeze(labels))