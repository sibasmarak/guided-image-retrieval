# Import the required libraries
import timm
import torch
import torch.nn as nn
from transformers import AutoModel
from env import model_paths

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import losses
import torch.nn.functional as F

# Language model
class LanguageModel(nn.Module):

    def __init__(self, model_name, input_size = 768, output_size = 512):
        super(LanguageModel, self).__init__()
        
        self.model_name = model_name
        self.model_path = model_paths[self.model_name]
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        # Instantiating Pre trained model object 
        self.model = AutoModel.from_pretrained(model_path)
        
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

    def __init__(self, model_name, hidden_size = 2048, output_size = 512):
        super(VisionModel, self).__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.output_size = output_size

        model = timm.create_model(self.model_name, pretrained = True, features_only = True)

        dense = nn.Linear(model.fc.in_features, self.output_size)
        global_pool = nn.AdaptiveAvgPool2d(self.output_size)
        self.model = nn.Sequential(OrderedDict([
            ('feat_extractor', model),
            ('global_pool', global_pool),
            ('dense', dense)
        ]))

        # m = timm.create_model(self.model_name, features_only = True, pretrained = True, num_classes = self.num_classes)
        #     global_pool = nn.AdaptiveAvgPool2d(self.output_size)
        #     self.model = nn.Sequential(OrderedDict([
        #         ('feat_extractor', m),
        #         ('fc', fc)
        #     ]))

    def forward(self, x):
        x = self.model(x)
        return x

class DualEncoder(pl.LightningModule):

    def __init__(self, vision_model_name, language_model_name, langauge_input_size = 768, 
                vision_hidden_size = 2048, output_size = 512, learning_rate = 1e-3):
        super().__init__()

        # 'save_hyperparameters' saves the values of anything in the __init__ for us to the checkpoint.
        # This is a useful feature.

        self.save_hyperparameters()
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.langauge_input_size = langauge_input_size
        self.vision_hidden_size = vision_hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.loss_cls = ContrastiveLoss()

        self.vision_model = VisionModel(self.vision_model_name, hidden_size = self.vision_hidden_size, output_size = self.output_size)
        self.language_model = LanguageModel(self.language_model_name, input_size = langauge_input_size, output_size = self.output_size)
        
        self.accuracy = torchmetrics.Accuracy()
    
    def forward(self, image, text_input_ids, attention_masks = None, token_type_ids = None):

        image_features = self.vision_model(image)
        text_features = self.language_model(text_input_ids, attention_masks = attention_masks, token_type_ids = token_type_ids)

        return image_features, text_features
    
    def training_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['image'], batch['text_input_ids'], 
                                                    batch['attention_masks'], batch['token_type_ids'])
        
        # self-supervised contrastive loss
        loss = self.loss_cls(image_features, text_features)

        # Logging training loss on each training step and also on each epoch
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Implement the validation step here.
        # Make sure to log the val_loss and val_acc as you did in the training step
        image_features, text_features = self.forward(batch['image'], batch['text_input_ids'], 
                                                    batch['attention_masks'], batch['token_type_ids'])
        
        # self-supervised contrastive loss
        loss = self.loss_cls(image_features, text_features)

        # Logging training loss on each training step and also on each epoch
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        # Implement the test step here.
        # Make sure to log the test_loss and test_acc.
        # Code is very similar to that of the validation step.
        image_features, text_features = self.forward(batch['image'], batch['text_input_ids'], 
                                                    batch['attention_masks'], batch['token_type_ids'])
        
        # self-supervised contrastive loss
        loss = self.loss_cls(image_features, text_features)

        # Logging training loss on each training step and also on each epoch
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)