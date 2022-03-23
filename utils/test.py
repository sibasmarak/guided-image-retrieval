from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import torch

def test_retrieval(model, dataloader, k_s = [1, 5]): # dataloader should have shuffle = False\

    with torch.no_grad():
        
        inner_dataloader = deepcopy(dataloader)
        total_recalls = [0]*len(k_s)
        len_recall = 0
        for batch in tqdm(dataloader):
            text_features = model.language_model(
                batch['caption_input_ids'], 
                attention_masks = batch['caption_attention_masks'], 
                token_type_ids = batch['caption_token_type_ids']
            )
            text_features = F.normalize(text_features, p=2, dim=1)
            similarities = []
            all_ids = []
            for batch_in in tqdm(inner_dataloader):
                image_features = model.vision_model(batch_in['images'])
                image_features = F.normalize(image_features, p=2, dim=1)
                all_ids.append(batch_in['image_ids'])
                sim_scores = torch.matmul(image_features, text_features.T)
                similarities.append(sim_scores)
            similarities = torch.cat(similarities, dim=0).T
            ranked_indexes = torch.argsort(similarities, dim=1)
            all_ids = torch.cat(all_ids, dim=0)
            ranked_ids = all_ids[ranked_indexes]
            curr_img_ids = batch['image_ids']
            curr_img_ids = curr_img_ids.unsqueeze(dim=0).T.repeat(1, ranked_ids.size(1))
            retrieval = torch.where(ranked_ids == curr_img_ids, 1, 0)
            for k_i, k in enumerate(k_s):
                total_recall += torch.sum(retrieval[:, :k])
            len_recall += retrieval.size(0)
        recalls = [float(total_recall)/len_recall for total_recall in total_recalls]

    return recalls