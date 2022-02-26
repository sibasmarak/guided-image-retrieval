import os, json
from tqdm import tqdm

def preprocess(data_path, dataset_name = "coco"):

    data = json.load(open(data_path, "rb"))
    preprocessed_dict = {}

    if dataset_name == "coco":

        for image_dict in tqdm(data["images"], position = 0, leave = True):
            Id = image_dict["id"]
            image_path = image_dict["file_name"]
            if "train" in image_path:
                dir_path = "./dataset/coco/train2014"
            elif "val" in image_path:
                dir_path = "./dataset/coco/val2014"

            preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), "captions": []}

        for annotation_dict in data["annotations"]:
            image_id = annotation_dict["image_id"]
            caption = annotation_dict["caption"]
            preprocessed_dict[image_id]["captions"].append(caption)

    elif dataset_name == "flickr30k":

        for image_dict in tqdm(data["images"], position = 0, leave = True):
            Id = image_dict["imgid"]
            image_path = image_dict["file_name"]
            dir_path = "./dataset/flickr30/flickr30k-images"
            preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), "captions": []}
            sentences_list = image_dict["sentences"]
            for token_sent_dict in sentences:
                image_id = token_sent_dict["imgid"]
                caption = token_sent_dict["raw"]
                preprocessed_dict[image_id]["captions"].append(caption)

    elif dataset_name == "cc":
        pass

    return preprocessed_dict

if __name__ == "__main__":

