import argparse
from src.model import LanguageModel, VisionModel, DualEncoder
from utils.loss import ContrastiveLoss
from utils.env import set_seed
from data.dataloader import ImageCaptionDataset

import pytorch_lightning as pl
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 64, type = int)
    parser.add_argument('--dataset', default = 'coco', type = str, choices=['cc', 'coco', 'flickr30k'])
    parser.add_argument('--deterministic', default = True, type = bool)
    parser.add_argument('--dropout', default = 0.4, type = float)
    parser.add_argument('--gpus', default = 2, type = int)
    parser.add_argument('--image_resize', default = (224, 224), type = tuple)
    parser.add_argument('--language_input_size', default = 768, type = int)
    parser.add_argument('--language_model_name', default = 'bert', type = str)
    parser.add_argument('--learning_rate', default = 1e-3, type = float)
    parser.add_argument('--local_files_only', default = True, type = bool)
    parser.add_argument('--max_epochs', default = 10, type = int)
    parser.add_argument('--max_length_caption', default = 64, type = int)
    parser.add_argument('--output_size', default = 512, type = int)
    parser.add_argument('--preprocess_text', default = True, type = bool)
    parser.add_argument('--pretrained', default = True, type = bool)
    parser.add_argument('--progress_bar_refresh_rate', default = 20, type = int)
    parser.add_argument('--seed', default = 21, type = int)
    parser.add_argument('--test', default = False, type = bool)
    parser.add_argument('--train', default = True, type = bool)
    parser.add_argument('--validation', default = True, type = bool)
    parser.add_argument('--vision_hidden_size', default = 2048, type = int)
    parser.add_argument('--vision_model_name', default = 'resnet50', type = str)
    parser.add_argument('--warn_grayscale', default = False, type = bool)


    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu"
    if torch.cuda.is_available(): device = torch.cuda.device("cuda:0")

    data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, batch_size = args.batch_size, 
                                preprocess_text = args.preprocess_text, train = args.train, validation = args.validation, 
                                test = args.test, max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
                                image_resize = args.image_resize, warn_grayscale = args.warn_grayscale)
    
    model = DualEncoder(args.vision_model_name, args.language_model_name, language_input_size = args.language_input_size, 
                vision_hidden_size = args.vision_hidden_size, output_size = args.output_size, learning_rate = args.learning_rate,
                dropout = args.dropout, pretrained = args.pretrained)
              
    trainer = pl.Trainer(max_epochs = args.max_epochs, 
                        progress_bar_refresh_rate = args.progress_bar_refresh_rate, 
                        gpus = args.gpus, deterministic=args.deterministic)

    if args.train and args.validation: trainer.fit(model, data.train_dataloader, data.validation_dataloader)
    if args.train: trainer.fit(model, data.train_dataloader)