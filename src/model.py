# Import the required libraries
import timm
import torch
import torch.nn as nn
from transformers import AutoModel
from utils.env import *
from utils.loss import ContrastiveLoss
from collections import OrderedDict

import torchmetrics
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import losses
import torch.nn.functional as F
from torch import optim

# Language model
class LanguageModel(nn.Module):

    def __init__(self, model_name, input_size = 768, output_size = 512, dropout=0.4):
        super(LanguageModel, self).__init__()
        
        self.model_name = model_name
        self.model_path = model_modelpath_mapping[self.model_name]
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        # Instantiating Pre trained model object 
        self.model = AutoModel.from_pretrained(self.model_path)
        
        # Layers
        # the first dense layer will have 768 neurons if base model is used and 
        # 1024 neurons if large model is used

        self.dense = nn.Linear(self.input_size, self.output_size)

    def forward(self, input_ids, attention_masks = None, token_type_ids = None):

        x = self.model(input_ids = input_ids, attention_mask = attention_masks,
                            token_type_ids = token_type_ids).pooler_output
        x = self.dense(x)
        return x

# TODO: Try later
# Use for non-BERT models
class NonPoolerTransformer(nn.Module):

    def __init__(self):
        super(NonPoolerTransformer, self).__init__()
        
        # Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)

        # Layers
        # the first dense layer will have 768 if base model is used and 
        # 1024 if large model is used

        self.dense_layer_1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.2)
        self.cls_layer = nn.Linear(128, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids, attention_masks=None):

        hidden_state = self.model_layer(input_ids=input_ids, attention_mask=attention_masks)[0]
        pooled_output = hidden_state[:, 0]

        x = self.dense_layer_1(pooled_output)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)

        logits = self.cls_layer(x_2)
        output = self.sigmoid(logits)

        return output

class VisionModel(nn.Module):

    def __init__(self, model_name, hidden_size = 2048, output_size = 512, pretrained = True):
        super(VisionModel, self).__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained = pretrained

        model = timm.create_model(self.model_name, pretrained = self.pretrained)
        dense = nn.Linear(model.fc.in_features, self.output_size)
        model.reset_classifier(0)
        self.model = nn.Sequential(OrderedDict([
            ('backbone', model),
            ('dense', dense)
        ]))

    def forward(self, x):
        x = self.model(x)
        return x

class DualEncoder(pl.LightningModule):

    def __init__(self, vision_model_name, language_model_name, language_input_size = 768, 
                vision_hidden_size = 2048, output_size = 512, vision_learning_rate=1e-2, 
                language_learning_rate = 1e-5, dropout = 0.4, pretrained = True, weight_decay=1e-4):
        super().__init__()

        # 'save_hyperparameters' saves the values of anything in the __init__ for us to the checkpoint.
        # This is a useful feature.

        self.save_hyperparameters()

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.language_input_size = language_input_size
        self.vision_hidden_size = vision_hidden_size
        self.output_size = output_size
        self.vision_learning_rate = vision_learning_rate
        self.language_learning_rate = language_learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.pretrained = pretrained

        self.loss_cls = ContrastiveLoss()

        self.vision_model = VisionModel(self.vision_model_name, hidden_size = self.vision_hidden_size, 
                                        output_size = self.output_size, pretrained = self.pretrained)
        self.language_model = LanguageModel(self.language_model_name, input_size = language_input_size, 
                                            output_size = self.output_size, dropout = self.dropout)
    
        self.accuracy = torchmetrics.Accuracy()
    
    def on_epoch_start(self):
        print('\n')

    def forward(self, image, text_input_ids, attention_masks = None, token_type_ids = None):
        
        image_features = self.vision_model(image)
        text_features = self.language_model(text_input_ids, attention_masks = attention_masks, token_type_ids = token_type_ids)

        return image_features, text_features
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])
        # image_features, text_features = self.forward(batch[0], batch[1], batch[2], batch[3])
        # loss = self.loss_cls(image_features, text_features, batch[4])
        # print('Loss:', loss)

        # Logging training loss on each training step and also on each epoch
        # self.log('train_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])
        # image_features, text_features = self.forward(batch[0], batch[1], batch[2], batch[3])
        # loss = self.loss_cls(image_features, text_features, batch[4])
        # print("Val loss: ", loss)

        # Logging training loss on each training step and also on each epoch
        # self.log('val_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def test_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])

        # Logging training loss on each training step and also on each epoch
        # self.log('test_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        vision_optimizer = optim.Adam(self.vision_model.parameters(), lr=self.hparams.vision_learning_rate, weight_decay=self.weight_decay)
        language_optimizer = optim.Adam(self.language_model.parameters(), lr=self.hparams.language_learning_rate, weight_decay=self.weight_decay)
        return [vision_optimizer, language_optimizer]