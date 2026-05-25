""" Set up RAPTOR to be used for slicing down eval set for BM25 search.
Using the following notebook as guide: https://github.com/parthsarthi03/raptor/blob/master/demo.ipynb"""

import os
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from huggingface_hub import login
login() # login to Hugging Face to access distilgpt2 model
from config import GPT4RecModelConfig
from sentence_transformers import SentenceTransformer
import torch

os.environ["OPENAI_API_KEY"] = "not_used"


class SBertEmbeddingModel(BaseEmbeddingModel): # default embeddings used in demo notebook
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class GPT4RecSummarizationModel(BaseSummarizationModel):
    def __init__(self, config: GPT4RecModelConfig):
        self.config = config
        self.model = config.hf_model_name # distilgpt2
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model)
        self.lm = GPT2LMHeadModel.from_pretrained(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device} for RAPTOR summarization")
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=self.model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self.device,  # Use "mps"/"cpu" if CUDA is not available
        )

    def summarize(self, context, max_tokens=100): 
        # format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary


def initialize_raptor(summarization_model: GPT4RecSummarizationModel, 
                      embedding_model: SBertEmbeddingModel):
    ra_config = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        embedding_model=embedding_model,
    )
    RA = RetrievalAugmentation(config=ra_config)
    return RA


def build_raptor_tree(RA: RetrievalAugmentation, document: str):
    tree = RA.add_documents(document)
    return tree


def save_raptor_tree(RA, tree, path):
    RA.save(path)