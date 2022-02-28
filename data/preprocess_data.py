import requests
import urllib.request
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import os, json, argparse, csv

def preprocess(data_path, dataset_name = "coco"):
    preprocessed_dict = {}

    if dataset_name == "coco":

        for image_dict in tqdm(data["images"], position = 0, leave = True):
            data = json.load(open(data_path, "r"))
            Id = image_dict["id"]
            image_path = image_dict["file_name"]
            if "train" in image_path:
                dir_path = "./datasets/coco/train2014"
            elif "val" in image_path:
                dir_path = "./datasets/coco/val2014"

            preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), "captions": []}

        for annotation_dict in data["annotations"]:
            image_id = annotation_dict["image_id"]
            caption = annotation_dict["caption"]
            preprocessed_dict[image_id]["captions"].append(caption)

    elif dataset_name == "flickr30k":
        data = json.load(open(data_path, "r"))

        for image_dict in tqdm(data["images"], position = 0, leave = True):
            Id = image_dict["imgid"]
            image_path = image_dict["file_name"]
            dir_path = "./datasets/flickr30/flickr30k-images"
            preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), "captions": []}
            sentences_list = image_dict["sentences"]
            for token_sent_dict in sentences:
                image_id = token_sent_dict["imgid"]
                caption = token_sent_dict["raw"]
                preprocessed_dict[image_id]["captions"].append(caption)

    elif dataset_name == "cc":

        if 'Train' in data_path:
            dir_path = "./datasets/cc/train"
        elif 'Validation' in data_path:
            dir_path = "./datasets/cc/val"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        url_to_Id = dict()

        with open(data_path) as f:
            num_examples = len(f.readlines())
            f.seek(0)
            
            tsv_file = csv.reader(f, delimiter="\t")
            failed = 0

            for Id, line in enumerate(tsv_file):
                caption = line[0]
                url = line[1]

                if url not in list(url_to_Id.keys()):
                    url_to_Id[url] = Id
                    image_path = os.path.join(dir_path, f'{Id}.jpg')
                    preprocessed_dict[Id] = {"image_path": image_path, "captions": [caption]}
                else:
                    Id = url_to_Id[url]
                    preprocessed_dict[Id]["captions"].append(caption)
                    continue

                try:
                    im = Image.open(requests.get(url, stream=True).raw)
                    im.save(image_path)
                except:
                    print(f'Failed! Id: [{Id + 1}|{num_examples}] URL: {url}')
                    failed += 1

            print(f'Total failed: [{failed}|{num_examples}]')

    return preprocessed_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'coco', type = str, choices=['cc', 'coco', 'flick30k'])
    parser.add_argument('--data_path', type = str)
    parser.add_argument('--split', default='train', type = str, choices=['train', 'val', 'test'])

    args = parser.parse_args()

    preprocessed_dict = preprocess(args.data_path, args.dataset)
    save_path = open(f'./datasets/{args.dataset}/{args.split}_image_captions.json', 'w')
    json.dump(preprocessed_dict, save_path)

