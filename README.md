## Commands to run  
**Note**: The commands are incomplete.  

```
  conda env create -f retrieval.yml
  conda activate retrieval
```
- `python data/preprocess_data.py` (to create the required data loading format)  
  - `python data/preprocess_data.py --dataset cc --data_path datasets/cc/Validation_GCC-1.1.0-Validation.tsv --split val`
  - `python data/preprocess_data.py --dataset cc --data_path datasets/cc/Train_GCC-training.tsv --split train`
- `python run.py`