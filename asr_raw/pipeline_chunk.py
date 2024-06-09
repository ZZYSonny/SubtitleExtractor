import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import srt
import datetime

def pipeline_chunk(in_path: str, out_path:str):
    # Load Model
    model_id = "openai/whisper-large-v3"
    device = "cuda"
    torch_dtype = torch.float16
    print("Loading Model")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    print("Load Processor")
    processor = AutoProcessor.from_pretrained(model_id)
    print("Load pipe")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        batch_size=8,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Run Model")
    result = pipe(
        in_path,
        chunk_length_s=60,
        stride_length_s=10,
        return_timestamps=True,
        generate_kwargs={"language": "japanese"},
    )
    # Convert Srt
    out_srt = []
    for rec in result["chunks"]:
        out_srt.append(srt.Subtitle(
            index=0,
            start=datetime.timedelta(seconds=rec["timestamp"][0]),
            end=datetime.timedelta(rec["timestamp"][1]),
            content=rec["text"],
        ))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(out_srt))
