from transformers import MBartForCausalLM
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

# LOAD MODEL AND TOKENIZER
model_id = "vinai/bartpho-syllable"

bartpho = MBartForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# DATA PREPROCESSING
from datasets import load_dataset

datasets = load_dataset("csv", data_files='data/Vinmec/file.csv')

def tokenize_function(examples):
    if examples["Answer"]:
        return tokenizer(examples["Question"]+examples["Answer"], truncation=True, padding='max_length', max_length=512)
    else:
        return tokenizer(examples["Question"], truncation=True, padding='max_length', max_length=512)
tokenized_datasets = datasets.map(tokenize_function, batched=False, remove_columns=datasets["train"].column_names)

block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_datasets.map(group_texts, batched=True)


# DATA COLLATOR

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# TRAINER
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="check_point",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    # push_to_hub=True,
)

trainer = Trainer(
    model=bartpho,
    args=training_args,
    train_dataset=lm_dataset["train"],
    # eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

prompt = "Mũi họng có đờm sau sinh nên dùng thuốc gì?"

inputs = tokenizer(prompt, return_tensors="pt").input_ids
outputs = bartpho.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))