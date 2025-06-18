import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

MODEL_ID = "microsoft/Phi-4-reasoning"

PERSISTENT_DIR = "/workspace"
CACHE_DIR = os.path.join(PERSISTENT_DIR, "huggingface_cache")
OUTPUT_DIR = os.path.join(PERSISTENT_DIR, "outputs")

TEMPERATURE = 0.8
TOP_K_SAMPLING = 50
TOP_P = 0.95
MAX_NEW_TOKENS = 32768

TOP_K_ANALYSIS = 50
VARIANCE_THRESHOLD = 2e-06
ENTROPY_THRESHOLD = 2

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

system_prompt_content = "You are Phi, a language model to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"
user_prompt_content = """A and B play a game. At first an integer n > 1 is written on a blackboard. 

In turn, A and B erase the number k they find on the blackboard and replace it either

1. with a positive divisor of k other than 1 and k itself, or

2. with k+1.

Initially each of the players has 1,000 points.

When a player plays move 1), he gains one point. 
when a player plays move 2), he loses one point. 

The game ends when one of the players comes to have zero points, and this player has lost. 

A plays first. 

For what values of n does B have a winning strategy?"""

prompt = (
    f"<|im_start|>system<|im_sep|>{system_prompt_content}<|im_end|>"
    f"<|im_start|>user<|im_sep|>{user_prompt_content}<|im_end|>"
    f"<|im_start|>assistant<|im_sep|>"
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

analysis_log = []
past_key_values = None

print("--- Starting Live Generation ---")
print(prompt, end="")

with torch.no_grad():
    for step in range(MAX_NEW_TOKENS):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        logits = logits.float()
        logits /= TEMPERATURE

        unfiltered_probs = torch.softmax(logits, dim=-1)
        entropy_value = - (unfiltered_probs * torch.log(unfiltered_probs + 1e-12)).sum().item()

        if TOP_K_SAMPLING > 0:
            top_k_vals, _ = torch.topk(logits, TOP_K_SAMPLING)
            logits[logits < top_k_vals[:, -1, None]] = -float("inf")

        sampling_probs = torch.softmax(logits, dim=-1)
        variance_value = torch.var(sampling_probs).item()
        # if variance_value < VARIANCE_THRESHOLD:
        #     print(f"\033[1m (VARIANCE: {variance_value:.2e}) \033[0m", end="")
        if entropy_value > ENTROPY_THRESHOLD:
             print(f"\033[1m (ENTROPY: {entropy_value:.2e}) \033[0m", end="")

        if TOP_P < 1.0:
            probs_for_filtering = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs_for_filtering, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > TOP_P
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            indices_to_remove = sorted_indices[mask]
            logits[:, indices_to_remove] = -float("inf")

        final_probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(final_probs, num_samples=1)

        full_probs = torch.softmax(outputs.logits[:, -1, :].float(), dim=-1)
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

        print(tokenizer.decode(next_token_id[0]), end="", flush=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        input_ids = next_token_id

print("\n\n--- Generation Complete ---")

with open(output_filename, "w") as f:
    json.dump(analysis_log, f, indent=2)

print(f"Detailed analysis saved to: {output_filename}")