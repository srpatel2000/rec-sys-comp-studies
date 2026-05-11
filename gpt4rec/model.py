'''
Basic GPT4Rec
'''

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT4RecommendationBaseModel(nn.Module):
    """Base collaborative GPT with extra user/item embeddings."""

    def __init__(self, config, gpt2model: GPT2LMHeadModel):
        super().__init__()
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd)
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.gpt2model = gpt2model

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        vocab_mask = (input_ids < self.vocab_size).long()
        user_mask = (
            (input_ids >= self.vocab_size)
            & (input_ids < self.vocab_size + self.num_users)
        ).long()
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()

        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size - 1)
        vocab_embeddings = self.gpt2model.transformer.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)

        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users - 1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)

        item_ids = (
            (input_ids - self.vocab_size - self.num_users) * item_mask
        ).clamp_(0, self.num_items - 1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)
        return vocab_embeddings + user_embeddings + item_embeddings

    def forward(self, input_ids=None, **kwargs):
        input_embeddings = self.embed(input_ids)
        return self.gpt2model(inputs_embeds=input_embeddings, **kwargs)


class GPT4RecGenerationModel(GPT4RecommendationBaseModel):
    """Generation-stage model used for query generation."""

    def forward_train(self, input_ids, attention_mask=None, labels=None):
        embeds = self.embed(input_ids)
        return self.gpt2model(
            inputs_embeds=embeds, attention_mask=attention_mask, labels=labels
        )

    @torch.no_grad()
    def generate_queries(
        self,
        input_ids,
        attention_mask=None,
        num_beams=5,
        num_return_sequences=5,
        max_new_tokens=16,
    ):
        return self.gpt2model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            early_stopping=True,
        )


class GPT4RecCandidateRanker(nn.Module):
    """Simple candidate ranker combining BM25 and model priors."""

    def __init__(self, bm25_weight=1.0):
        super().__init__()
        self.bm25_weight = bm25_weight

    def score_candidates(self, candidate_item_ids, candidate_bm25_scores):
        return {
            int(i): float(s) * self.bm25_weight
            for i, s in zip(candidate_item_ids, candidate_bm25_scores)
        }