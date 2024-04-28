import srt
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen1.5-1.8B-Chat"
model_prompt = (
    "将下面一句对话翻译成中文。"
    "你要翻译的这句对话来自一部二次元番剧。"
    "因此你的翻译需要尽可能简洁，贴近中文语境下的口语，不要使用书面语。"
    "不要翻译番剧中的角色名、地名、人名、专业术语等。"
    "再一次强调，你的翻译需要尽可能贴近口语，下面是你要翻译的句子："
)

print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
    low_cpu_mem_usage=True
)
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def translate_srt(in_path: str, out_path:str):
    # Read and parse srt
    with open(in_path, 'r', encoding='utf-8') as f:
        raw_str = f.read()
        raw_srt = list(srt.parse(raw_str))
    
    for entry in tqdm(raw_srt):
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": model_prompt},
                {"role": "user", "content": entry.content}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=32
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        entry.content = response
        print(response)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(raw_srt))
