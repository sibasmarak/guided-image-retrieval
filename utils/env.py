import torch
import numpy as np
import random as rn
from pytorch_lightning import Trainer, seed_everything

def set_seed(seed):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True

model_modelpath_mapping = {'bert': './pretrained/bert-base-uncased', 
                            'roberta': 'roberta-base', 
                            'bart': "facebook/bart-base", 
                            'distilbert': 'distilbert-base-uncased', 
                            'deberta': 'microsoft/deberta-base', 
                            'debertalarge': 'microsoft/deberta-large', 
                            'xlnet' : 'xlnet-base-cased', 
                            'xlnetlarge' : 'xlnet-large-cased', 
                            'xlmrobertalarge' : 'xlm-roberta-large', 
                            'bartlarge' : 'facebook/bart-large', 
                            'bertlarge': 'bert-large-uncased', 
                            'robertalarge': 'roberta-large'}

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