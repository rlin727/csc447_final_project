# ULF DataReader
 This is a customized datareader for ULF dataset, created for CSC 447 final project

 # ULF Dataset
 Download the dataset:
 https://www.cs.rochester.edu/u/gkim21/ulf/resources/

 Split the dataset using:
 ```
 python split-data.py --input ulf-1.0.json --trainpath [path] --devpath [path] --testpath [path]
 ```

# Create data folder
```
mkdir test
mkdir train
mkdir validation
 ```
 Run 
 ```
 python split-data.py --input ulf-1.0.json --trainpath \train --devpath \validation --testpath \test
 ```
 
 # ULFReader.py
This is the implementation of ULF datareader file.

First register the datareader as "ULF_reader" in AllenNlp, so it can be used in the configuration file.

```Python
@DatasetReader.register("ULF_reader")
```

Next, it requires to override three functions:
```Python
__init__(self, ...):

_read(self,file_path):

text_to_instance(self, ...):
```

* ```__init__``` will take parameters defined in the configuration file.

    Here, it takes 4 parameters:
        
        1. max_instance: maximum number of instance the reader will process
        2. tokenizer: the tokenizer that will be used for processing
        3. token_indexer: determines how string tokens get represented as arrays of indices in a model
        4. max_tokens: maximum number of tokens
    Since the dataset is not large, lazy loading is not an option here.

    Default values:

        1. max_instance: 100000
        2. tokenizer: WhitespaceTokenizer()
        3. token_indexer: SingleIdTokenIndexer()
        4. max_tokens: None

* ```_read``` will simply read the dataset, tokenize the text, and call the ```text_to_instance``` function recursively.

* ```text_to_instance``` will create different fields for each column and instance for each row, returning an iterable instance.

# Dataloader.py 
This refers to https://github.com/allenai/allennlp/blob/main/allennlp/data/data_loaders/simple_data_loader.py

It is a implementation of ```simple_data_loader``` in AllenNlp.

Used here only for testing the ULF datareader.

# conf.jsonnet
This is the configuration file for AllenNlp project.

# reader_test.ipynb
Test for datareader
```
allennlp train --dry-run --include-package src -s [non-existent fold path] configs/conf.jsonnet
```

Results:
```
loading instances: 0it [00:00, ?it/s]
loading instances: 458it [00:00, 4578.16it/s]
loading instances: 1378it [00:00, 7324.97it/s]

loading instances: 0it [00:00, ?it/s]
loading instances: 180it [00:00, 6921.42it/s]

building vocab: 0it [00:00, ?it/s]
building vocab: 1558it [00:00, 120033.17it/s]
```

Note AttributeError will happen since model and trainer are set to be null here.

# ULF_reader.ipynb
```ULF_reader.ipynb``` in test folder can be used to see what each batch is through the process. 

This is for testing only.

# Reference

* https://github.com/allenai/allennlp/blob/main/allennlp/data/data_loaders/simple_data_loader.py
* https://zhuanlan.zhihu.com/p/352412971
* https://jbarrow.ai/allennlp-the-hard-way-1/
