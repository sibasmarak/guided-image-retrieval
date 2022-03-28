# Import the required libraries
import os
import json
import torch
import warnings
import regex as re
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.env import model_modelpath_mapping, model_with_no_token_types

class OldImageCaptionDataset():

	def __init__(self, dataset, language_model_name = "bert", batch_size = 64, preprocess_text = True,
				train = True, validation = True, test = True, max_length_caption = 64, 
				local_files_only = True, image_resize = (224, 224), warn_grayscale = False, load_cache = False):

		self.dataset = dataset
		self.language_model_name = language_model_name
		self.load_cache = load_cache
		self.batch_size = batch_size
		self.preprocess_text = preprocess_text
		self.train = train
		self.validation = validation
		self.test = test
		self.max_length_caption = max_length_caption
		self.local_files_only = local_files_only
		self.image_resize = image_resize
		self.warn_grayscale = warn_grayscale


		self.transform = transforms.Compose([
											transforms.Resize(self.image_resize),
											transforms.ToTensor()
											]) 
						

		self.language_model_path = model_modelpath_mapping[self.language_model_name]
		self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_path, local_files_only = self.local_files_only,
														TOKENIZERS_PARALLELISM = True)

		self.train_dataloader = None
		self.vaidation_dataloader = None
		self.test_dataloader = None

		if self.train:
			self.train_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/all_image_captions.json', 'r'))
			self.train_dataloader = self.create_dataset(self.train_preprocessed_dict)

		# 	self.train_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/train_image_captions.json', 'r'))
		# 	self.train_dataloader = self.create_dataset(self.train_preprocessed_dict)
		
		# if self.validation:
		# 	self.validation_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/val_image_captions.json', 'r'))
		# 	self.validation_dataloader = self.create_dataset(self.validation_preprocessed_dict)

		# if self.test:
		# 	self.test_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/test_image_captions.json', 'r'))
		# 	self.test_dataloader = self.create_dataset(self.test_preprocessed_dict)

	def preprocess_caption(self, caption):

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

	def create_dataset(self, preprocessed_dict):
		
		if self.load_cache:
			cache_dir = os.path.join('datasets', self.dataset, 'cache')
			if not os.path.exists(cache_dir):
				print("Caching not done!")

			print(f"Loading tensors from {cache_dir}")

			caption_input_ids = torch.load(os.path.join(cache_dir, 'caption_input_ids.pt'))
			caption_attention_masks = torch.load(os.path.join(cache_dir, 'caption_attention_masks.pt'))
			image_tensors = torch.load(os.path.join(cache_dir, 'image_tensors.pt'))
			image_ids = torch.load(os.path.join(cache_dir, 'image_ids.pt'))
			if self.language_model_name not in model_with_no_token_types:
				caption_token_type_ids = torch.load(os.path.join(cache_dir, 'caption_token_type_ids.pt'))
		else:
			image_tensors = []
			image_ids = []
			caption_input_ids = []
			caption_token_type_ids = []
			caption_attention_masks = []

			i = 0
			for image_id, values in tqdm(preprocessed_dict.items(), position = 0, leave = True):
				
				image_path = values["image_path"]
				# if not os.path.exists(image_path):
				# 	continue
				captions = values["captions"]
				image = Image.open(image_path)

				# Convert gray scale image to RGB
				if image.mode == 'L':
					if self.warn_grayscale:
						warnings.warn('image %s is grayscale..' % image_path,
									RuntimeWarning)
					image = image.convert('RGB')
				
				image = self.transform(image)
				
				for caption in captions:
					
					if self.preprocess_text:
						caption = self.preprocess_caption(caption)

					encoded_caption = self.tokenizer(caption, max_length = self.max_length_caption, padding = 'max_length')
					caption_input_ids.append(encoded_caption['input_ids'])
					caption_attention_masks.append(encoded_caption['attention_mask'])

					if self.language_model_name not in model_with_no_token_types:
						caption_token_type_ids.append(encoded_caption['token_type_ids'])

					image_tensors.append(image)
					image_ids.append(torch.tensor(int(image_id)))

				if i == 1000:
					break

				i+=1

			caption_input_ids = torch.tensor(caption_input_ids).squeeze()
			caption_attention_masks = torch.tensor(caption_attention_masks).squeeze()
			image_tensors = torch.stack(image_tensors, axis = 0)
			image_ids = torch.stack(image_ids, axis = 0)
			
			cache_dir = os.path.join('datasets', self.dataset, 'cache')
			if not os.path.exists(cache_dir):
				os.makedirs(cache_dir)

			print(f"Caching tensors to {cache_dir}")
			torch.save(caption_input_ids, os.path.join(cache_dir, 'caption_input_ids.pt'))
			torch.save(caption_attention_masks, os.path.join(cache_dir, 'caption_attention_masks.pt'))
			torch.save(image_tensors, os.path.join(cache_dir, 'image_tensors.pt'))
			torch.save(image_ids, os.path.join(cache_dir, 'image_ids.pt'))
			if self.language_model_name not in model_with_no_token_types:
				caption_token_type_ids = torch.tensor(caption_token_type_ids).squeeze()
				torch.save(caption_token_type_ids, os.path.join(cache_dir, 'caption_token_type_ids.pt'))


		if self.language_model_name not in model_with_no_token_types:
			dataset = TensorDataset(image_tensors, caption_input_ids, caption_attention_masks, caption_token_type_ids, image_ids)
		else:
			dataset = TensorDataset(image_tensors, caption_input_ids, caption_attention_masks, image_ids)

		dataloader = DataLoader(dataset, batch_size = self.batch_size, num_workers = 2)
		return dataloader













###############################################################################


class ImageCaptionDataset(Dataset):

	def __init__(self, dataset, language_model_name = "bert", preprocess_text = True,
				split='train', max_length_caption = 64, local_files_only = True, 
				image_resize = (224, 224), warn_grayscale = False, eval = False):
<<<<<<< HEAD

=======
>>>>>>> 893b5fe451dc1a51784a3147ede045954661a0f4

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
		# 	caption = self.preprocess_caption(caption)


		# tokenize the caption
		# encoded_caption = self.tokenizer(caption, max_length = self.max_length_caption, padding = 'max_length')


		caption_input_ids = encoded_caption['input_ids']
		caption_attention_masks = encoded_caption['attention_mask']
		caption_input_ids = torch.tensor(caption_input_ids).squeeze()
		caption_attention_masks = torch.tensor(caption_attention_masks).squeeze()

		caption_token_type_ids = None
		if self.language_model_name not in model_with_no_token_types:
			caption_token_type_ids = encoded_caption['token_type_ids']
			caption_token_type_ids = torch.tensor(caption_token_type_ids).squeeze()


		return image_id, image, caption_input_ids, caption_attention_masks, caption_token_type_ids

	def collater(self, items):
		
		if self.language_model_name not in model_with_no_token_types:
			batch = {
				'image_ids': torch.stack([x[0] for x in items], dim=0),
				'images': torch.stack([x[1] for x in items], dim=0),
				'caption_input_ids': torch.stack([x[2] for x in items], dim=0),
				'caption_attention_masks': torch.stack([x[3] for x in items], dim=0),
				'caption_token_type_ids': torch.stack([x[4] for x in items], dim=0)
			}
		else:
			batch = {
				'image_ids': torch.stack([x[0] for x in items], dim=0),
				'images': torch.stack([x[1] for x in items], dim=0),
				'caption_input_ids': torch.stack([x[2] for x in items], dim=0),
				'caption_attention_masks': torch.stack([x[3] for x in items], dim=0)
			}


		return batch
	
	def preprocess_caption(self, caption):

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
