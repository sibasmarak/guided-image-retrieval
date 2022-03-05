import argparse
from src.model import LanguageModel, VisionModel, DualEncoder
from utils.loss import ContrastiveLoss
from utils.env import set_seed
from data.dataloader import ImageCaptionDataset, OldImageCaptionDataset

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default = 128, type = int)
	parser.add_argument('--dataset', default = 'coco', type = str, choices=['cc', 'coco', 'flickr30k'])
	parser.add_argument('--deterministic', default = True, type = bool)
	parser.add_argument('--dropout', default = 0.4, type = float)
	parser.add_argument('--gradient_clip_val', default = 30.0, type = float)
	parser.add_argument('--gpus', default = 2, type = int)
	parser.add_argument('--image_resize', default = (224, 224), type = tuple)
	parser.add_argument('--language_input_size', default = 768, type = int)
	parser.add_argument('--language_model_name', default = 'bert', type = str)
	parser.add_argument('--vision_learning_rate', default = 1e-4, type = float)
	parser.add_argument('--language_learning_rate', default = 1e-5, type = float)
	parser.add_argument('--local_files_only', default = True, type = bool)
	parser.add_argument('--load_cache', default = True, type = bool)
	parser.add_argument('--max_epochs', default = 10, type = int)
	parser.add_argument('--max_length_caption', default = 64, type = int)
	parser.add_argument('--num_workers', default = 2, type = int)
	parser.add_argument('--output_size', default = 512, type = int)
	parser.add_argument('--preprocess_text', default = True, type = bool)
	parser.add_argument('--pretrained', default = False, type = bool)
	parser.add_argument('--progress_bar_refresh_rate', default = 5, type = int)
	parser.add_argument('--seed', default = 0, type = int)
	parser.add_argument('--test', default = False, type = bool)
	parser.add_argument('--train', default = True, type = bool)
	parser.add_argument('--validation', default = True, type = bool)
	parser.add_argument('--vision_hidden_size', default = 2048, type = int)
	parser.add_argument('--vision_model_name', default = 'resnet50', type = str)
	parser.add_argument('--warn_grayscale', default = False, type = bool)
	parser.add_argument('--weight_decay', default = 1e-4, type = float)


	args = parser.parse_args()

	set_seed(args.seed)
	device = "cpu"
	if torch.cuda.is_available(): device = torch.cuda.device("cuda:0")

	# dataloaders
	train_dataloader, validation_dataloader, test_dataloader = None, None, None
	if args.train:
		train_data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text, 
								split='train', max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
								image_resize = args.image_resize, warn_grayscale = args.warn_grayscale)
		train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collater)

	if args.validation:
		validation_data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text, 
								split='val', max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
								image_resize = args.image_resize, warn_grayscale = args.warn_grayscale)
		validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=validation_data.collater)
	
	if args.test:
		test_data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text, 
								split='test', max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
								image_resize = args.image_resize, warn_grayscale = args.warn_grayscale)
		test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=test_data.collater)

	# dataset = OldImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text,
	# 			batch_size = args.batch_size, train = True, validation = True, test = False, max_length_caption = 64, 
	# 			local_files_only = True, image_resize = (224, 224), warn_grayscale = False, load_cache = False)

	# model    
	model = DualEncoder(args.vision_model_name, args.language_model_name, language_input_size = args.language_input_size, 
				vision_hidden_size = args.vision_hidden_size, output_size = args.output_size, vision_learning_rate = args.vision_learning_rate,
				language_learning_rate = args.language_learning_rate, dropout = args.dropout, pretrained = args.pretrained, weight_decay=args.weight_decay)

	# trainer        
	trainer = pl.Trainer(max_epochs = args.max_epochs, strategy="ddp_spawn",
						progress_bar_refresh_rate = args.progress_bar_refresh_rate, gpus = args.gpus, gradient_clip_val=args.gradient_clip_val)
						# add deterministic in Trainer if cannot reproduce results

	if args.train and args.validation: trainer.fit(model, train_dataloader, validation_dataloader)
	if args.train: trainer.fit(model, train_dataloader)