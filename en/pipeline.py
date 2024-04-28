import srt
from tqdm import tqdm
import torch
import vllm
import transformers
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str
    model_dtype: torch.dtype
    model_device: str
    model_prompt: str
    vllm_args: dict
    vllm_search_param: dict

config = Config(
    model_name = "Qwen/Qwen1.5-1.8B-Chat",
    model_dtype = torch.float16,
    model_device = "cuda",
    model_prompt = (
        "你的任务是将下面的英文对话翻译成中文，"
        "你的回答只能包含翻译后的中文。"
        "你要翻译的这句对话来自一部二次元番剧。"
        "因此你的翻译需要尽可能简洁，贴近中文语境下的口语，不要使用任何书面语。"
        "不要翻译番剧中的角色名、地名、人名、专业术语等。"
        "再一次强调，你的翻译需要尽可能贴近口语，下面是你要翻译的句子："
    ),
    vllm_args = dict(),
    vllm_search_param = {
        "temperature": 0.9,
        "top_p": 0.95
    }
)

def translate_srt_hf(in_path: str, out_path:str, config: Config):
    print("Loading Model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        device_map=config.model_device,
        low_cpu_mem_usage=True
    )
    print("Loading Tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    print("Loading SRT")
    with open(in_path, 'r', encoding='utf-8') as f:
        raw_srt = list(srt.parse(f.read()))
    
    for entry in tqdm(raw_srt):
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.model_prompt},
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

def translate_srt_vllm(in_path: str, out_path:str, config: Config):
    print("Loading Model")
    llm = vllm.LLM(
        model=config.model_name, 
        tokenizer=config.model_name, 
        dtype=config.model_dtype, 
        quantization="awq" if "-AWQ" in config.model_name else None,
        max_model_len=360, 
        enforce_eager=True
    )
    print("Loading Tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    print("Loading SRT")
    with open(in_path, 'r', encoding='utf-8') as f:
        raw_srt = list(srt.parse(f.read()))
    
    model_input = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.model_prompt},
                {"role": "user", "content": entry.content.replace("\n"," ")}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for entry in raw_srt
    ]
    model_output = llm.generate(model_input, vllm.SamplingParams(**config.vllm_search_param))

    for i in range(len(raw_srt)):
        raw_srt[i].content = "".join(list(out.text for out in model_output[i].outputs))
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(raw_srt))