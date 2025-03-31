from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import torch

# Configuration
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_path = "processed_dataset.jsonl"
output_dir = "deepseek-abap-finetuned"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# PEFT Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=50,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="no",
    report_to="tensorboard"
)

# Dataset preparation
dataset = load_dataset('json', data_files=dataset_path, split='train')

# Trainer setup
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=2048,
    tokenizer=tokenizer,
    peft_config=peft_config,
    packing=True
)

# Start training
trainer.train()
trainer.save_model(output_dir)
