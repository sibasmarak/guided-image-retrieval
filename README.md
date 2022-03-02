## Commands to run  

```
  conda env create -f retrieval.yml
  conda activate retrieval
```

- `python data/preprocess_data.py` (to create the required data loading format)   

  For CC  
  - `python data/preprocess_data.py --dataset cc --data_path datasets/cc/Validation_GCC-1.1.0-Validation.tsv --split val`
  - `python data/preprocess_data.py --dataset cc --data_path datasets/cc/Train_GCC-training.tsv --split train`  
  For MS-COCO  
  - `python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_val2014.json --split val`
  - `python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_train2014.json --split train`  
  For Flickr30k  
  - `python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k-captions/dataset.json --split all`
- `python run.py` (to train/evaluate)