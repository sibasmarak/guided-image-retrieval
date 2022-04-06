# Import the required libraries
import os
import json
from sre_parse import Tokenizer
from data.preprocess_data import preprocess
import torch
import warnings
import regex as re
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.env import model_modelpath_mapping, model_with_no_token_types
import numpy as np

def preprocess_text(caption, language_model_name = "bert", max_length_caption=64, local_files_only=True):
		language_model_path = model_modelpath_mapping[language_model_name]
		tokenizer = AutoTokenizer.from_pretrained(language_model_path, local_files_only = local_files_only, TOKENIZERS_PARALLELISM = True)
		caption = preprocess_caption(caption)
		encoded_caption = tokenizer(caption, max_length=max_length_caption, padding='max_length', truncation=True)
		caption_input_ids = encoded_caption['input_ids']
		caption_attention_masks = encoded_caption['attention_mask']
		caption_input_ids = torch.tensor([caption_input_ids]).squeeze()
		caption_attention_masks = torch.tensor([caption_attention_masks]).squeeze()
		caption_token_type_ids = None
		if language_model_name not in model_with_no_token_types:
			caption_token_type_ids = encoded_caption['token_type_ids']
			caption_token_type_ids = torch.tensor([caption_token_type_ids]).squeeze()

		caption_dict = {
			'caption_input_ids': caption_input_ids,
			'caption_attention_masks': caption_attention_masks,
			'caption_token_type_ids': caption_token_type_ids,
		}
		return caption_dict


def preprocess_caption(caption):

		caption = re.sub('[^a-zA-Z]', ' ', caption)
			
		url = re.compile(r'https?://\S+|www\.\S+')
		caption = url.sub(r'', caption)
		
		html = re.compile(r'<.*?>')
		caption = html.sub(r'', caption)

		emoji_pattern = re.compile("["
								u"\U0001F600-\U0001F64F"  # emoticons
								u"\U0001F300-\U0001F5FF"  # symbols & pictographs
								u"\U0001F680-\U0001F6FF"  # transport & map symbols
								u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
								u"\U00002702-\U000027B0"
								u"\U000024C2-\U0001F251"
								"]+", flags=re.UNICODE)
		caption = emoji_pattern.sub(r'', caption)

		return caption

# Image Caption Dataset for coco, flickr30k, conceptual captions
class ImageCaptionDataset(Dataset):

	def __init__(self, dataset, language_model_name = "bert", preprocess_text = True,
				split='train', max_length_caption = 64, local_files_only = True, 
				image_resize = (224, 224), warn_grayscale = False, eval = False):

		self.dataset = dataset
		self.language_model_name = language_model_name
		self.preprocess_text = preprocess_text
		self.split = split
		self.max_length_caption = max_length_caption
		self.local_files_only = local_files_only
		self.image_resize = image_resize
		self.warn_grayscale = warn_grayscale

		self.preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/{self.split}_image_captions.json', 'r'))

		self.language_model_path = model_modelpath_mapping[self.language_model_name]
		self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_path, local_files_only = self.local_files_only,
														TOKENIZERS_PARALLELISM = True)
														
		self.image_ids = []
		self.image_paths = []
		self.captions = []

		i = 0
		len_prepro_dict = len(list(self.preprocessed_dict.items()))
		for image_ids, path_caption_dict in list(self.preprocessed_dict.items()):
			if eval:
				num_captions = 1
			else:				
				num_captions = len(path_caption_dict['captions'])

			image_path = path_caption_dict['image_path']

			# create the lists
			self.image_ids.extend([image_ids] * num_captions)   
			self.image_paths.extend([image_path] * num_captions)
			captions = path_caption_dict['captions'][:num_captions]

			for idx in range(num_captions):
				if self.preprocess_text:
					captions[idx] = self.preprocess_caption(captions[idx])

				captions[idx] = self.tokenizer(captions[idx], max_length = self.max_length_caption, padding = 'max_length', truncation=True)

			self.captions.extend(captions)
			if self.split == 'train' and i >= int(0.20 * len_prepro_dict):
				break
			i += 1

		self.transform = transforms.Compose([
											transforms.Resize(self.image_resize),
											transforms.ToTensor()
											]) 
						
	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, idx):
		# obtain image id
		image_id = torch.tensor(int(self.image_ids[idx]))

		# obtain the image
		image_path = self.image_paths[idx]
		image = Image.open(image_path)
		# Convert gray scale image to RGB
		if image.mode == 'L':
			if self.warn_grayscale:
				warnings.warn('image %s is grayscale..' % image_path,
							RuntimeWarning)
			image = image.convert('RGB')
		
		image = self.transform(image)

		# obtain the text
		# caption = self.captions[idx]
		encoded_caption = self.captions[idx]
		# if self.preprocess_text:
		# 	encoded_caption = self.preprocess_caption(caption)

		if encoded_caption is None:
			# tokenize the caption
			encoded_caption = self.tokenizer(caption, max_length = self.max_length_caption, padding = 'max_length')

		caption_input_ids = encoded_caption['input_ids']
		caption_attention_masks = encoded_caption['attention_mask']
		caption_input_ids = torch.tensor(caption_input_ids).squeeze()
		caption_attention_masks = torch.tensor(caption_attention_masks).squeeze()

		caption_token_type_ids = None
		if self.language_model_name not in model_with_no_token_types:
			caption_token_type_ids = encoded_caption['token_type_ids']
			caption_token_type_ids = torch.tensor(caption_token_type_ids).squeeze()


		return image_id, image, caption_input_ids, caption_attention_masks, caption_token_type_ids, image_path

	def collater(self, items):
		
		if self.language_model_name not in model_with_no_token_types:
			batch = {
				'image_ids': torch.stack([x[0] for x in items], dim=0),
				'images': torch.stack([x[1] for x in items], dim=0),
				'caption_input_ids': torch.stack([x[2] for x in items], dim=0),
				'caption_attention_masks': torch.stack([x[3] for x in items], dim=0),
				'caption_token_type_ids': torch.stack([x[4] for x in items], dim=0),
				'image_path': np.array([x[5] for x in items]),
			}
		else:
			batch = {
				'image_ids': torch.stack([x[0] for x in items], dim=0),
				'images': torch.stack([x[1] for x in items], dim=0),
				'caption_input_ids': torch.stack([x[2] for x in items], dim=0),
				'caption_attention_masks': torch.stack([x[3] for x in items], dim=0),
				'image_path': np.array([x[5] for x in items]),
			}

		return batch
	
	def preprocess_caption(self, caption):

		return preprocess_caption(caption)
