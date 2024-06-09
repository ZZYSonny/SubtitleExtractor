import os
import srt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio

def slice_and_asr(in_video_path: str, in_srt_path:str, temp_path: str, out_path: str):
    # Open Subtitle file
    with open(in_srt_path, 'r', encoding='utf-8') as f:
        raw_srt = list(srt.parse(f.read()))
        n = len(raw_srt)
    # Open audio file
    track, sample_rate = torchaudio.load(in_video_path)
    # Extract audio
    for i in range(n):
        start = int(raw_srt[i].start.total_seconds() * sample_rate)
        end = int(raw_srt[i].end.total_seconds() * sample_rate)
        torchaudio.save(f"{temp_path}/{i}.wav", track[:, start:end], sample_rate)
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
        [f"{temp_path}/{i}.wav" for i in range(n)],
        generate_kwargs={"language": "japanese"},
    )
    for i in range(n):
        raw_srt[i].content = result[i]["text"]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(raw_srt))
slice_and_asr(".temp/in.mp4", ".temp/in.srt", ".temp", "out.srt")