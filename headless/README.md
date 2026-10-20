```
uv venv --python 3.12 
source .venv/bin/activate
Change this Value to either 0 or 1, OUTPUT_DEVICE_INDEX: int | None = 1
uv add "pipecat-ai[silero,whisper,mlx-whisper,ollama,kokoro,local]" ollama                   
uv run bot.py
```