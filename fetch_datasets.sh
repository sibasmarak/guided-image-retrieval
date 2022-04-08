#!/bin/bash

mkdir datasets
cd datasets
mkdir flickr30k
mkdir coco

# fetch flickr30k
cd flickr30k
wget https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip
unzip flickr30k.zip > flickr30k-captions
cd ..

# fetch coco
cd coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2015.zip
wget http://images.cocodataset.org/zips/test2015.zip


unzip train2014.zip
unzip annotations_trainval2014.zip
unzip val2014.zip
unzip image_info_test2014.zip
unzip test2014.zip
unzip image_info_test2015.zip
unzip test2015.zip
cd ..

# run data preprocessing flickr30k
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k-captions/dataset.json --split train 
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k-captions/dataset.json --split test 
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k-captions/dataset.json --split val 

# run data preprocessing coco
python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_val2014.json --split val
python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_val2014.json --split train