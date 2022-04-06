# Import the required libraries
import torch
import numpy as np
import random as rn
from pytorch_lightning import Trainer, seed_everything

# Set global seed
def set_seed(seed):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True

# Model checkpoint mapping
model_modelpath_mapping = {'bert': './pretrained/bert-base-uncased', 
                            'roberta': './pretrained/roberta-base', 
                            'bart': "./pretrained/facebook/bart-base", 
                            'distilbert': './pretrained/distilbert-base-uncased', 
                            'deberta': './pretrained/microsoft/deberta-base', 
                            'debertalarge': './pretrained/microsoft/deberta-large', 
                            'xlnet' : './pretrained/xlnet-base-cased', 
                            'xlnetlarge' : './pretrained/xlnet-large-cased', 
                            'xlmrobertalarge' : './pretrained/xlm-roberta-large', 
                            'bartlarge' : './pretrained/facebook/bart-large', 
                            'bertlarge': './pretrained/bert-large-uncased', 
                            'robertalarge': './pretrained/roberta-large'}

# Non pooler transformers
model_with_no_token_types =['roberta', 
                            'bart', 
                            'albert', 
                            'distilbert', 
                            'deberta', 
                            'xlmroberta', 
                            'xlnet', 
                            'xlnetlarge', 
                            'robertalarge', 
                            'bartlarge', 
                            'debertalarge', 
                            'xlmrobertalarge', 
                            'albertlarge']