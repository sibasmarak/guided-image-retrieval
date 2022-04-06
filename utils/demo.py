# Import the required libraries
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import shutil

def demo_retrieval(model, sentence, dataloader, device, k = 5, train_siam = True): # dataloader should have shuffle = False

    model.to(device)
    model.eval()

    len_recall = 0
    
    with torch.no_grad():
        
        if train_siam:
            text_features = model.dual_encoder.language_model(
                sentence['caption_input_ids'].to(device), 
                attention_masks = sentence['caption_attention_masks'].to(device), 
                token_type_ids = sentence['caption_token_type_ids'].to(device)
            )
        else:
            text_features = model.language_model(
                sentence['caption_input_ids'].to(device), 
                attention_masks = sentence['caption_attention_masks'].to(device), 
                token_type_ids = sentence['caption_token_type_ids'].to(device)
            )
    
    text_features = F.normalize(text_features.cpu(), p=2, dim=1)

    similarities = []
    all_ids = []
    all_images = []
    
    for batch_in in dataloader:
    
        with torch.no_grad():
            if train_siam:
                image_features = model.dual_encoder.vision_model(batch_in['images'].to(device))
            else:
                image_features = model.vision_model(batch_in['images'].to(device))

        image_features = F.normalize(image_features.cpu(), p=2, dim=1)
        all_ids.append(batch_in['image_ids'])
        all_images.append(batch_in['image_path'])
    
        sim_scores = torch.matmul(image_features, text_features.T)
        similarities.append(sim_scores)
    
    similarities = torch.cat(similarities, dim=0).T
    ranked_indexes = torch.argsort(similarities, dim=1)
    all_ids = torch.cat(all_ids, dim=0)
    all_images = np.concatenate(all_images, dim=0)
    ranked_images = all_images[ranked_indexes.cpu().numpy()]
    image_paths = ranked_images[0, :k]
    for i, image_path in enumerate(list(image_paths)):
      shutil.copy(image_path, f"results/{i}.jpg")
    
    model.train()
    