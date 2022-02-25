# Import the required libraries
import timm
import torch
import torch.nn as nn
from transformers import AutoModel

# Language model
class LanguageModel(nn.Module):

    def __init__(self, model_path, input_size = 768, output_size = 512):
        super(LanguageModel, self).__init__()
        
        self.model_path = model_path
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        # Instantiating Pre trained model object 
        self.model = AutoModel.from_pretrained(model_path)
        
        # Layers
        # the first dense layer will have 768 neurons if base model is used and 
        # 1024 neurons if large model is used

        self.dense = nn.Linear(self.input_size, self.output_size)

    def forward(self, input_ids, attention_masks, token_type_ids):

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

    def forward(self,input_ids, attention_masks):

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

        model = timm.create_model(self.model_name, pretrained = True)
        dense = nn.Linear(model.fc.in_features, self.output_size)

        self.model = nn.Sequential(OrderedDict([
            ('feat_extractor', model),
            ('dense', dense)
        ]))

    def forward(self, x):
        x = self.model(x)
        return x