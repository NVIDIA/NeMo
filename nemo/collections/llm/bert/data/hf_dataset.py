from  nemo.collections.llm.gpt.data.hf_dataset import HFDatasetDataModule

from datasets import Dataset, DatasetDict, load_dataset
import torch


class IMDBHFDataModule(HFDatasetDataModule):
    """A data module for handling the HellaSwag dataset using HFDatasetDataModule."""

    def __init__(self, tokenizer, dataset_name="imdb", *args, **kwargs):
        tokenizer.pad_token = tokenizer.pad_token
        dataset = load_dataset(dataset_name)
        sequence_length = kwargs.get("seq_length", 128)
        dataset = IMDBHFDataModule.preprocess_dataset(tokenizer, sequence_length, dataset)
        dataset.pop("unsupervised")
        def collate_fn(batch):
            batch_dict = {
                key: [] for key in batch[0].keys()
            }
            
            # Collect all values for each key
            for example in batch:
                for key in batch_dict.keys():
                    batch_dict[key].append(example[key])
            
            # Convert lists to tensors
            return {
                key: torch.LongTensor(value) 
                for key, value in batch_dict.items()
            }
        super().__init__(dataset, train_aliases=["train"], test_aliases=["test"], val_aliases=["test"], sequence_length=sequence_length,collate_fn=collate_fn, *args, **kwargs)




    # Note: I'm training the model causally not through multiclass classification.
    @staticmethod
    def preprocess_dataset(tokenizer, max_length, dataset, seed=42):
        """Preprocesses a dataset for training a language model."""
        # Format each prompt.
        print("Preprocessing dataset...")
        def tokenize_function(examples):
            return tokenizer.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True  # Adds [CLS] and [SEP] automatically
            )
        dataset = dataset.map(tokenize_function, batched=True).select_columns(["input_ids", "attention_mask", "label"])

       
    
        # Shuffle dataset.
        dataset = dataset.shuffle(seed=seed)

        return dataset

