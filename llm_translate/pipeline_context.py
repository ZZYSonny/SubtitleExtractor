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
    sample_params: vllm.SamplingParams
    prompt_template: str
    prompt_map: dict[str, str]


config = Config(
    model_name="Qwen/Qwen2-1.5B-Instruct",
    model_dtype=torch.float16,
    model_device="cuda",
    sample_params=vllm.SamplingParams(temperature=0, top_k=3),
    prompt_template=(
        "你的任务是将下面的{lang_source}对话翻译成中文，"
        "你的回答只能包含翻译后的中文。"
        "你要翻译的这句对话来自一部二次元番剧。"
        "故事的背景是: {background}"
        "因此你的翻译需要尽可能简洁，贴近中文语境下的口语，不要使用任何书面语。"
        "不要翻译番剧中的角色名、地名、人名、专业术语等，比如{no_translate}。"
        "再一次强调，你的翻译需要尽可能贴近口语。"
        "对话的背景是: {history}"
    ),
    prompt_map={
        "background": "Please Set me in main.py",
        "lang_source": "Please Set me in main.py",
        "no_translate": "Please Set me in main.py",
    },
)


def translate_srt_vllm(in_path: str, out_path: str, config: Config):
    print("Loading Model")
    llm = vllm.LLM(
        model=config.model_name,
        tokenizer=config.model_name,
        dtype=config.model_dtype,
        quantization="awq" if "-AWQ" in config.model_name else None,
        max_model_len=420,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    print("Loading Tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    print("Loading SRT")
    with open(in_path, "r", encoding="utf-8") as f:
        raw_srt = list(srt.parse(f.read()))
        raw_srt_content = [entry.content.replace("\n", " ") for entry in raw_srt]

    model_input = []
    for i in range(len(raw_srt_content)):
        prompt_real = config.prompt_template.format(**config.prompt_map).format(
            history=" ".join(raw_srt_content[max(0, i - 4) : i])
        )
        ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt_real},
                {"role": "user", "content": raw_srt_content[i]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_input.append(ids)

    model_output = llm.generate(model_input, config.sample_params)

    for i in range(len(raw_srt)):
        raw_srt[i].content = "".join(list(out.text for out in model_output[i].outputs))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(raw_srt))
