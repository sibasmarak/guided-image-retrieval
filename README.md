# Guided Image Retrieval

Retrieving images from an image collection, given a textual input, has attracted a lot of attention over the past few years. While existing neural methods have shown promising performance over conventional early deep learning techniques, they typically lack in using visual information properly to ground textual inputs (as in the case of dual encoder retrieval models) or tend to take a large amount of time (as in the case of cross attention retrieval models). In this work, we leverage a simple visual-grounding method to aid dual encoders and compare the performance with using only dual encoders.

### File Description

This repository contains 3 python scripts and 3 directories. 

- [run.py](./run.py): Main script to aggregate all the functions, train the Dual Encoder and Spatial Information Aggregator Module, as well as evaluate them.
- [test.py](./test.py): Test script to load the previously trained checkpoint of a model and evaluate the model.
- [demo.py](./demo.py): Script to run a demo. Given a query text, demo retrieves top k (default is 5) images.

The following files are present in the directory ```data```.
- [dataloader.py](./data/dataloader.py): Script to create a custom Dataloader for the image and text datasets (MS COCO, Flickr30k). The class ```ImageCaptionDataset``` uses preprocesses the text and applies required transforms on the images.
- [preprocess_data.py](./data/preprocess_data.py): Script to create a uniform json for each of MS COCO, Flickr30k and Conceptual Captions data. The raw data for all three datasets are in different directory structure. So, to maintain uniformity, run ```preprocess_data.py``` to generate the required ```.json``` files.

The following file is present in the directory ```src```.
- [model.py](./src/model.py): Script to create different model architectures for the task of semantic segmentation. The following architectures are implemented.
  - Different variants of ResNet architecture
  - Vision Model with ResNet50 architecture backbone
  - Language Model with BERT architecture backbone
  - Dual Encoder
  - Spatial Information Aggregator Module

The following files are present in the directory ```utils```.
- [demo.py](./utils/demo.py): Script contains the functions used for demo.
- [env.py](./utils/env.py): Script to define the global random seed environment for the sake of reproducibility.
- [loss.py](./utils/loss.py): Script to define the loss function as Noise Contrastive Estimation loss.
- [test.py](./utils/test.py): Script contains the function to test a model, given its checkpoint.

### Setup the environment  

```
  conda env create -f retrieval.yml
  conda activate retrieval
```

### Data preprocessing

```./fetch_datasets.sh```
This will obtain the Flickr30k and MS-COCO dataset in the required format for training and evaluation. 
NOTE: 
- Images of Flickr30k dataset need to be requested through a form available on the official [website](http://shannon.cs.illinois.edu/DenotationGraph/), hence the above script would not be able to fetch the images of Flickr30k dataset.
- Since MS-COCO dataset has sizes in the range of GB (13 GB for train split, 6GB for validation split and 12GB for test split), running this script would require a couple of hours. 

### Experiments

- To run the code with default arguments: `python run.py` (to train/evaluate)  
- To run the code with user defined arguments: 
  - For DE: `python ./run.py --batch_size 64 --preprocess_text True --weight_decay 0 --output_size 256 --train_siam False --train_de True --gpus 0,1 `      
  - For SIAM: `python ./run.py --batch_size 32 --preprocess_text True --weight_decay 0 --output_size 512 --train_siam True --gpus 1 --siam_max_epochs 10` 
- To test the code with model checkpoint
  - For DE: `python test.py --checkpoint "lightning_logs/version_230/checkpoints/epoch=7-step=14799.ckpt" --train_siam False`
  - For SIAM: `python test.py --checkpoint "lightning_logs/version_230/checkpoints/epoch=7-step=14799.ckpt" --train_siam True`
- To run the demo with model checkpoint
  - For DE: `python test.py --checkpoint "lightning_logs/version_230/checkpoints/epoch=7-step=14799.ckpt" --train_siam False`  
  - For SIAM: `python test.py --checkpoint "lightning_logs/version_230/checkpoints/epoch=7-step=14799.ckpt" --train_siam True`

### Results

The table below enumerates the Recall@1 and Recall@5 for different output sizes (the final dimension of the image and text representation, 256 and 512 here).

| Model | Output | Epochs | R@1  | R@5  |
|-------|--------|--------|------|------|
|       |        |        |      |      |
| DE    | 256    | 5      | 0.04 | 0.15 |
| DE    | 256    | 10     | 0.09 | 0.36 |
| DE    | 512    | 5      | 0.07 | 0.23 |
| DE    | 512    | 10     | 0.20 | 0.64 |
|       |        |        |      |      |
| SIAM  | 256    | 5      | 0.09 | 0.39 |
| SIAM  | 256    | 10     | 0.18 | 0.86 |
| SIAM  | 512    | 5      | 0.16 | 0.65 |
| SIAM  | 512    | 10     | 0.35 | 0.95 |

It is evident that as the output size increases, the performance of both DE and SIAM increases. Moreover, we can also observe that the performance increases as the models are trained for more epochs (5, 10). This provides us the motivation that, if the models are trained completely for around 180 epochs with the whole MS COCO dataset, the performance will comparatively be better. Moreover, comparing DE with SIAM shows us that SIAM performs better than DE. Intuitively, since SIAM is guided by the object present in the image, the representation is more aligned to the text, hence the re-ranking is better.
