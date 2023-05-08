#!/usr/bin/env python
# coding: utf-8

# In[2]:


from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoConfig, TFAutoModelForCausalLM, AdamWeightDecay
from transformers import PushToHubCallback
from datasets import load_dataset
from transformers import ClassLabel
import pandas as pd
from IPython.display import HTML, display
import random
import math


notebook_login()

datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
datasets["train"][10]


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


show_random_elements(datasets["train"])

model_checkpoint = "gpt2"
tokenizer_checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

tokenized_datasets["train"][1]

# block_size = tokenizer.model_max_length
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=8)

tokenizer.decode(lm_datasets["train"][1]["input_ids"])

config = AutoConfig.from_pretrained(model_checkpoint)
model = TFAutoModelForCausalLM.from_config(config)

learning_rate = 2e-5
weight_decay = 0.01
push_to_hub_model_id = f"{model_checkpoint}-wikitext2"


optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer, jit_compile=True)

train_set = model.prepare_tf_dataset(lm_datasets["train"], shuffle=True, batch_size=16)

validation_set = model.prepare_tf_dataset(lm_datasets["validation"], shuffle=False, batch_size=16)

model_name = model_checkpoint.split("/")[-1]
# push_to_hub_model_id = f"{model_name}-finetuned-wikitext2"
push_to_hub_model_id = f"{model_name}-msxl"
callback = PushToHubCallback(
    output_dir="/content/drive/MyDrive/clm_from_language_model_save", tokenizer=tokenizer, hub_model_id=push_to_hub_model_id,
)

model.fit(train_set, validation_data=validation_set, epochs=10, callbacks=[callback])

eval_loss = model.evaluate(validation_set)

print(f"Perplexity: {math.exp(eval_loss):.2f}")


# In[ ]:





# In[ ]:




