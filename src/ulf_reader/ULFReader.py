import pandas as pd
import numpy as np
from typing import Dict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer

@DatasetReader.register("ULF_reader")
class ULFReader(DatasetReader):
    def __init__(self,
                 max_instances = 100000,
                 tokenizer = None,
                 token_indexers = None,
                 max_tokens= None):
        
        super().__init__(max_instances = max_instances)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
    
    def text_to_instance(self, fields):
        return Instance(fields)
    
    def _read(self, file_path: str):
        df = pd.read_json(file_path)
        df.columns = df.columns.astype(str)
        for _ , row in df.iterrows():
            text = row['1']
            tokens = self.tokenizer.tokenize(text)
            if self.max_tokens:
                tokens = tokens[:self.max_tokens]
            
            sentence = TextField(tokens, self.token_indexers)
            ID = LabelField(str(row['0']))
            ULF = LabelField(row['2'])
            ULF_AMR = LabelField(row['3'])
            fields = {'text': sentence, 'ID': ID, 'ULF': ULF, 'ULF_AMR': ULF_AMR}
            yield self.text_to_instance(fields)