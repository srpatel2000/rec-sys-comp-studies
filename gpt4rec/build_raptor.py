""" Set up RAPTOR to be used for slicing down eval set for BM25 search.
Using the following notebook as guide: https://github.com/parthsarthi03/raptor/blob/master/demo.ipynb"""

import os
from raptor import RetrievalAugmentation 
from huggingface_hub import login
login() # login to Hugging Face to access distilgpt2 model

os.environ["OPENAI_API_KEY"] = "not_used"


def initialize_raptor():
    RA = RetrievalAugmentation(
        model="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1000,
        max_rerank=10,
    )
    return RA

def build_raptor_tree(RA, document):
    


def query_raptor_tree(RA, query):