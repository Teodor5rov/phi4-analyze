import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

MODEL_ID = "microsoft/Phi-4-reasoning-plus"

PERSISTENT_DIR = "/workspace"
CACHE_DIR = os.path.join(PERSISTENT_DIR, "huggingface_cache")
OUTPUT_DIR = os.path.join(PERSISTENT_DIR, "outputs")

TEMPERATURE = 0.8
TOP_K_SAMPLING = 50
TOP_P = 0.95
MAX_NEW_TOKENS = 32768

TOP_K_ANALYSIS = 50
VARIANCE_THRESHOLD = 2e-06

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = os.path.join(
    OUTPUT_DIR, f"analysis_{MODEL_ID.split('/')[-1]}_{timestamp}.json"
)

print("--- Configuration ---")
print(f"Model ID: {MODEL_ID}")
print(f"Persistent Cache Directory: {CACHE_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
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
    MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
)
print("Model and tokenizer loaded successfully.")
print("-" * 20)

system_prompt_content = "You are a helpful AI assistant. Provide a clear and accurate answer to the user's question."
user_prompt_content = """How are you?"""

prompt = (
    f"<|im_start|>system<|im_sep|>{system_prompt_content}<|im_end|>"
    f"<|im_start|>user<|im_sep|>{user_prompt_content}<|im_end|>"
    f"<|im_start|>assistant<|im_sep|>"
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

analysis_log = []
generated_token_ids = []
past_key_values = None

print("--- Starting Live Generation ---")
print(prompt, end="")

with torch.no_grad():
    for step in range(MAX_NEW_TOKENS):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        logits /= TEMPERATURE
        if TOP_K_SAMPLING > 0:
            top_k_values, _ = torch.topk(logits, TOP_K_SAMPLING)
            logits[logits < top_k_values[:, -1, None]] = -float("inf")

        sampling_probs = torch.softmax(logits, dim=-1)
        variance_value = torch.var(sampling_probs).item()
        if variance_value < VARIANCE_THRESHOLD:
            print(f"\033[1m (VARIANCE: {variance_value:.2e}) \033[0m", end="")

        dist = torch.distributions.Categorical(probs=sampling_probs)
        entropy_value = dist.entropy().item()

        if TOP_P < 1.0:
            probs_for_filtering = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs_for_filtering, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > TOP_P
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float("inf")

        final_probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(final_probs, num_samples=1)

        full_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(full_probs, TOP_K_ANALYSIS, dim=-1)
        step_data = {
            "step": step + 1,
            "chosen_token": tokenizer.decode(next_token_id[0]),
            "chosen_token_prob": full_probs[0, next_token_id[0]].item(),
            "variance": variance_value,
            "entropy": entropy_value,
            "top_k_predictions": [
                {
                    "token": tokenizer.decode(int(top_indices[0, i])),
                    "probability": float(top_probs[0, i]),
                }
                for i in range(TOP_K_ANALYSIS)
            ],
        }
        analysis_log.append(step_data)

        generated_token_ids.append(next_token_id.item())
        print(tokenizer.decode(next_token_id[0]), end="", flush=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        input_ids = next_token_id

print("\n\n--- Generation Complete ---")

with open(output_filename, "w") as f:
    json.dump(analysis_log, f, indent=2)

print(f"Detailed analysis saved to: {output_filename}")