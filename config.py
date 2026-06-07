"""
Global project configuration.

This file defines variables shared across the repository. 
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List


@dataclass
class SASRecModelConfig:
	"""SASRec Model/training hyperparameters."""

	# Model vars
	batch_size: int = 128
	lr: float = 0.001
	num_epochs: int = 201
	l2_emb: float = 0.0

	# Model architecture vars
	maxlen: int = 50
	hidden_units: int = 50
	num_blocks: int = 2
	num_heads: int = 2
	dropout_rate: float = 0.2
	num_preds: int = 10  # number of items to return in the ranked list of predictions for each user (for evaluation)
	train_eval_every: int = 5  # epochs between in-training eval of model performance
	val_eval_max_users: int = 512  # max val users scored per in-training eval (speeds up eval to have a cap)


@dataclass
class GPT4RecModelConfig:
	"""GPT4Rec Model/training hyperparameters."""

	# Optimization
	batch_size: int = 128
	lr: float = 0.001
	num_epochs: int = 21
	l2_emb: float = 0.0

	# Backbone / architecture
	hf_model_name: str = "distilgpt2"
	maxlen: int = 50
	num_preds: int = 10
	# n_embd: int = 768
	initializer_range: float = 0.02
	train_eval_every: int = 2
	val_eval_max_users: int = 512

	# Query generation + retrieval
	num_beams: int = 2
	num_queries_per_user: int = 2
	max_query_tokens: int = 16
	search_top_k: int = 50
	bm25_k1_default: float = 1.2
	bm25_b_default: float = 0.75
	# bm25_k1_grid: List[float] = field(default_factory=lambda: [0.8, 1.2, 1.6])
	# bm25_b_grid: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
	bm25_k1_grid: List[float] = field(default_factory=lambda: [1.2])
	bm25_b_grid: List[float] = field(default_factory=lambda: [0.75])
	raptor_prefilter_k: int = 25 # top k items to return from RAPTOR for BM25 prefilter



@dataclass
class GlobalConfig:
	"""Project-level settings."""

	datasets: List[str] = field(default_factory=lambda: [
		"CDs_and_Vinyl",
		"Movies_and_TV",
	])

	train_dir: str = "default"
	samples: int = 50000  # number of samples to load from the full dataset

	project_root: Path = Path(__file__).resolve().parent
	data_dir: Path = project_root / "data"
	eda_dir: Path = project_root / "eda"
	trained_models_dir: Path = project_root / "trained_models"
	
	random_seed: int = 42
	use_gpu: bool = True

	sasRecModel: SASRecModelConfig = field(default_factory=SASRecModelConfig)
	gpt4RecModel: GPT4RecModelConfig = field(default_factory=GPT4RecModelConfig)

	def to_dict(self) -> dict:
		"""Return all config values as a dict."""
		config_dict = asdict(self)
		config_dict["project_root"] = str(self.project_root)
		config_dict["data_dir"] = str(self.data_dir)
		config_dict["eda_dir"] = str(self.eda_dir)
		config_dict["trained_models_dir"] = str(self.trained_models_dir)
		return config_dict

	def model_namespace(self, model_name: str = "sasrec") -> SimpleNamespace:
		if model_name.lower() == "sasrec":
			return SimpleNamespace(**asdict(self.sasRecModel))
		if model_name.lower() == "gpt4rec":
			return SimpleNamespace(**asdict(self.gpt4RecModel))
		raise ValueError(f"Unknown model_name: {model_name}")


# Global singleton used across the project.
config = GlobalConfig()

