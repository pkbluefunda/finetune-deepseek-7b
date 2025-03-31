from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


dataset_path = "processed_dataset.jsonl"
output_dir = "deepseek-abap-finetuned"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Quantization config (4-bit)
bnb_config_4b = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Quantization config (8-bit)
bnb_config_8b = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16
)

# Load model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_8b,
    trust_remote_code=True
)

# Print layer names (look for query/key/value layers)
for name, module in model.named_modules():
    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
        print(name)

# Prepare model for k-bit training (critical for meta tensors)
model = prepare_model_for_kbit_training(model)

for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # Updated for DeepSeek-R1-Distill-Qwen
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",  # Required for 4-bit training
    learning_rate=2e-4,
    fp16=True,                  # Use AMP (if GPU supports it)
    max_grad_norm=0.3,
    num_train_epochs=1,
    logging_steps=30,
    save_steps=1000
)

# Dataset preparation
dataset = load_dataset('json', data_files=dataset_path, split='train')

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

# Start training
trainer.train()
trainer.save_model(output_dir)

