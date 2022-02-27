from tqdm import tqdm
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
        with open(data_path) as f:
            tsv_file = csv.reader(f, delimiter="\t")
            for line in tsv_file:
                print(line)
                exit(0)

        

    return preprocessed_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'coco', type = str, choices=['cc', 'coco', 'flick30k'])
    parser.add_argument('--data_path', type = str)

    args = parser.parse_args()

    preprocessed_dict = preprocess(args.data_path, args.dataset)
    json.dump(preprocessed_dict, open(os.path.join('./datasets', f'{args.dataset}', 'image_captions.json')))

