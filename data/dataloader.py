# Import the required libraries
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

class ImageCaptionDataset():

    def __init__(self, dataset, language_model_name = "bert", batch_size = 64, preprocess_text = True,
                train = True, validation = True, test = True, max_length_caption = 64, 
                local_files_only = True, image_resize = (224, 224), warn_grayscale = False):

        self.dataset = dataset
        self.language_model_name = language_model_name
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
            self.train_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/train_image_captions.json', 'r'))
            self.train_dataloader = self.create_dataset(self.train_preprocessed_dict)
        
        if self.validation:
            self.validation_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/val_image_captions.json', 'r'))
            self.validation_dataloader = self.create_dataset(self.validation_preprocessed_dict)

        if self.test:
            self.test_preprocessed_dict = json.load(open(f'./datasets/{self.dataset}/test_image_captions.json', 'r'))
            self.test_dataloader = self.create_dataset(self.test_preprocessed_dict)

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
        
        image_tensors = []
        image_ids = []
        caption_input_ids = []
        caption_token_type_ids = []
        caption_attention_masks = []

        # i = 0

        for image_id, values in tqdm(preprocessed_dict.items(), position = 0, leave = True):
            
            image_path = values["image_path"]
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

            # i+=1
            # if i == 1000:
            #     break

        caption_input_ids = torch.tensor(caption_input_ids).squeeze()
        caption_attention_masks = torch.tensor(caption_attention_masks).squeeze()
        image_tensors = torch.stack(image_tensors, axis = 0)
        image_ids = torch.stack(image_ids, axis = 0)

        if self.language_model_name not in model_with_no_token_types:
            caption_token_type_ids = torch.tensor(caption_token_type_ids).squeeze()
            dataset = TensorDataset(image_tensors, caption_input_ids, caption_attention_masks, caption_token_type_ids, image_ids)
        else:
            dataset = TensorDataset(image_tensors, caption_input_ids, caption_attention_masks, image_ids)

        dataloader = DataLoader(dataset, batch_size = self.batch_size, num_workers = 2)
        return dataloader