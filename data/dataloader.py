# Import the required libraries
import json
import torch
import regex as re
from PIL import Image
from utils.env import model_modelpath_mapping, model_with_no_token_types
from torch.utils.data import Dataset, DataLoader, TensorDataset

class ImageCaptionDataset():

    def __init__(self, dataset, langauge_model_name = "bert", batch_size = 64, preprocess_text = True,
                train = True, validation = True, test = True, max_length_caption = 64):

        self.data_path = data_path
        self.langauge_model_name = langauge_model_name
        self.batch_size = batch_size
        self.preprocess_text = preprocess_text
        self.train = train
        self.validation = validation
        self.test = test
        self.max_length_caption = max_length_caption

        self.language_model_path = model_modelpath_mapping[self.langauge_model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_path)

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
        caption_input_ids = []
        caption_token_type_ids = []
        caption_attention_masks = []

        for image_id, values in preprocessed_dict:
            
            image_path = values["image_path"]
            captions = values["captions"]
            image = Image.open(image_path)
            image = torch.tensor(image)
            
            for caption in captions:
                
                if self.preprocess_text:
                    caption = self.preprocess_caption(caption)

                encoded_caption = self.tokenizer(caption, max_length = self.max_len_caption, padding = 'max_length')
                caption_input_ids.append(encoded_caption['input_ids'])
                caption_attention_masks.append(encoded_caption['attention_mask'])

                if self.langauge_model_name not in model_with_no_token_types:
                    caption_token_type_ids.append(encoded_caption['token_type_ids'])

                image_tensors.append(image)

        caption_input_ids = torch.tensor(caption_input_ids).squeeze()
        caption_attention_masks = torch.tensor(caption_attention_masks).squeeze()
        image_tensors = torch.stack(image_tensors)

        if self.langauge_model_name not in model_with_no_token_types:
            caption_token_type_ids = torch.tensor(caption_token_type_ids).squeeze()
            dataset = TensorDataset(image_tensors, caption_input_ids, caption_token_type_ids, caption_attention_masks)
        else:
            dataset = TensorDataset(image_tensors, caption_input_ids, caption_attention_masks)

        dataloader = DataLoader(dataset, batch_size = self.batch_size, num_workers = 2)
        return dataloader