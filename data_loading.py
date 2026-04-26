import datasets
datasets.logging.set_verbosity_error()
from datasets import load_dataset
from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig


# This file will load in the dataset for this project 
def loadAmazonReviews(config=GlobalConfig, 
                      core=5,
                      sample_size=100000, 
                      cold_start=False):
    
    categories = config.datasets # loads categories from config
    datasets_list = []

    for cat in categories:
        subset = f"raw_review_{cat}" if cold_start else f"{core}core_raw_review_{cat}"
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            subset,
            split="train",
            trust_remote_code=True,
        )
        datasets_list.append(ds)

    conc_dataset = datasets.concatenate_datasets(datasets_list)
    n = min(sample_size, len(conc_dataset))
    dataset = conc_dataset.shuffle(seed=config.random_seed).select(range(n))

    # save dataset to data directory
    file_name = "cold_start_reviews.csv" if cold_start else "dense_reviews.csv"
    dataset.to_csv(str(config.data_dir/file_name))

    return 


def loadItemMetadata(cold_start=False):



def createUserItemPairings(cold_start=False):