import os
import re
import json


def process_files(input_dir, output_file):
    dataset = []

    # Regex pattern to extract PROMPT-CODE pairs
    pattern = re.compile(r'PROMPT:\s*(.*?)\s*CODE:\s*(.*?)(?=\nPROMPT:|\Z)', re.DOTALL)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.abap', '.txt')):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = pattern.findall(content)
                    for prompt, code in matches:
                        dataset.append({
                            "messages": [
                                {"role": "user", "content": prompt.strip()},
                                {"role": "assistant", "content": code.strip()}
                            ]
                        })

    # Save as JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    process_files('data/', 'processed_dataset.jsonl')
