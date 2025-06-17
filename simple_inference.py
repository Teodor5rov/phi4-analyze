import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_ID = "microsoft/Phi-4-reasoning-plus"
PERSISTENT_DIR = "/workspace"
CACHE_DIR = os.path.join(PERSISTENT_DIR, "huggingface_cache")

os.makedirs(CACHE_DIR, exist_ok=True)

print("--- Configuration ---")
print(f"Model ID: {MODEL_ID}")
print(f"Persistent Cache Directory: {CACHE_DIR}")
print("-" * 20)

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)
print("Model and tokenizer loaded successfully.")
print("-" * 20)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant. Provide a clear and accurate answer to the user's question."},
    {"role": "user", "content": "Hey how are you today?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

outputs = model.generate(
    inputs.to(model.device),
    max_new_tokens=32768,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    streamer=streamer,
)