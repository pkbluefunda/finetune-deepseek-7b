from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def init_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

def generate_abap(prompt, generator):
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    result = generator(
        formatted_prompt,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    return result[0]['generated_text'].split("<|im_end|>")[-2].strip()

if __name__ == "__main__":
    generator = init_model("deepseek-abap-finetuned")
    prompt = input("Enter ABAP generation prompt: ")
    print(generate_abap(prompt, generator))
