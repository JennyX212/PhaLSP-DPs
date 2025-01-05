import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List


import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""


def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])


"""
Transform a dna sequence to k-mer string
"""


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)

    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    # One-hot encode the labels for AUC calculation
    n_classes = logits.shape[-1]
    valid_labels_one_hot = np.eye(n_classes)[valid_labels]

    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }

    # Calculate AUC for each class and take the average (macro-average)
    try:
        metrics["auc"] = sklearn.metrics.roc_auc_score(valid_labels_one_hot, logits[valid_mask], average="macro",
                                                       multi_class="ovr")
    except ValueError:
        metrics["auc"] = np.nan

    # Calculate AUPR (Average Precision)
    try:
        precision, recall, _ = sklearn.metrics.precision_recall_curve(valid_labels_one_hot.ravel(),
                                                                      logits[valid_mask].ravel())
        metrics["aupr"] = sklearn.metrics.auc(recall, precision)
    except ValueError:
        metrics["aupr"] = np.nan

    # Calculate specificity
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(valid_labels, valid_predictions).ravel()
    metrics["specificity"] = tn / (tn + fp)

    return metrics


"""
Compute metrics used for hu ggingface trainer.
"""


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=os.path.join(data_args.data_path, "train.csv"),
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer,
                                    data_path=os.path.join(data_args.data_path, "dev.csv"),
                                    kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, "test.csv"),
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    )

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = "E:/xz/pywork/2024PhageLS/DNABERT2/finetune/output/results"
        # results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        # os.makedirs(results_path, exist_ok=True)
        results = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加模型参数
    # model = AutoModel.from_pretrained("E:\\xz\\pywork\\2024PhageLS\\DNABERT2\\DNABERT-2-117M", trust_remote_code=True)
    parser.add_argument('--model_name_or_path', type=str, default="E:\\xz\\pywork\\2024PhageLS\\DNABERT2\\DNABERT-2-117M", help="Model name or path")
    parser.add_argument('--data_path', type=str, default="E:/xz/pywork/2024PhageLS/DNABERT2/finetune/data/576", help="Path to training data")
    parser.add_argument('--kmer', type=int, default=-1, help="k-mer for input sequence")
    parser.add_argument('--run_name', type=str, default="DNABERT2", help="Run name")
    parser.add_argument('--model_max_length', type=int, default=200, help="Maximum sequence length")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Train batch size")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, help="Eval batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--save_steps', type=int, default=200, help="Save every X updates steps")
    parser.add_argument('--output_dir', type=str, default="E:/xz/pywork/2024PhageLS/DNABERT2/finetune/output", help="Output directory")
    parser.add_argument('--evaluation_strategy', type=str, default="steps", help="Evaluation strategy")
    parser.add_argument('--eval_steps', type=int, default=200, help="Evaluate every X steps")
    parser.add_argument('--warmup_steps', type=int, default=50, help="Warmup steps")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite output directory")
    parser.add_argument('--log_level', type=str, default="info", help="Log level")
    parser.add_argument('--find_unused_parameters', type=bool, default=False, help="Find unused parameters")

    args = parser.parse_args()

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        # 其他模型参数
    )

    data_args = DataArguments(
        data_path=args.data_path,
        kmer=args.kmer,
    )

    training_args = TrainingArguments(
        run_name=args.run_name,
        model_max_length=args.model_max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        overwrite_output_dir=args.overwrite_output_dir,
        log_level=args.log_level,
        find_unused_parameters=args.find_unused_parameters,
    )

    train(model_args, data_args, training_args)

