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
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)
print("Model and tokenizer loaded successfully.")
print("-" * 20)


messages = [
    {
        "role": "system",
        "content": (
            "You are Phi, a language model trained by Microsoft to help users. "
            "Your role as an assistant involves thoroughly exploring questions "
            "through a systematic thinking process before providing the final "
            "precise and accurate solutions. This requires engaging in a "
            "comprehensive cycle of analysis, summarizing, exploration, "
            "reassessment, reflection, backtracing, and iteration to develop "
            "well-considered thinking process. Please structure your response "
            "into two main sections: Thought and Solution using the specified "
            "format: <think> {Thought section} </think> {Solution section}. "
            "In the Thought section, detail your reasoning process in steps. "
            "Each step should include detailed considerations such as analysing "
            "questions, summarizing relevant findings, brainstorming new ideas, "
            "verifying the accuracy of the current steps, refining any errors, "
            "and revisiting previous steps. In the Solution section, based on "
            "various attempts, explorations, and reflections from the Thought "
            "section, systematically present the final solution that you deem "
            "correct. The Solution section should be logical, accurate, and "
            "concise and detail necessary steps needed to reach the conclusion. "
            "Now, try to solve the following question through the above guidelines:"
        ),
    },
    {
        "role": "user",
        "content": (
            "A sphere is inscribed in a cube. What is the ratio of the "
            "volume of the sphere to the volume of the cube? Provide the "
            "final answer as a simplified fraction involving pi."
        ),
    },
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

analysis_log = []
generated_token_ids = []

print("--- Starting Live Generation ---")
print(prompt, end="")

with torch.no_grad():
    for step in range(MAX_NEW_TOKENS):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]

        # Apply sampling parameters
        logits /= TEMPERATURE
        top_k_values, _ = torch.topk(logits, TOP_K_SAMPLING)
        logits[logits < top_k_values[:, -1, None]] = -float("inf")
        probs_for_filtering = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(
            probs_for_filtering, descending=True
        )
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float("inf")

        final_probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(final_probs, num_samples=1)

        # Log analysis data
        full_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(full_probs, TOP_K_ANALYSIS, dim=-1)
        step_data = {
            "step": step + 1,
            "chosen_token": tokenizer.decode(next_token_id[0]),
            "chosen_token_prob": full_probs[0, next_token_id[0]].item(),
            "top_k_predictions": [
                {
                    "token": tokenizer.decode(int(top_indices[0, i])),
                    "probability": float(top_probs[0, i]),
                }
                for i in range(TOP_K_ANALYSIS)
            ],
        }
        analysis_log.append(step_data)

        # Stream output to console
        generated_token_ids.append(next_token_id.item())
        print(tokenizer.decode(next_token_id[0]), end="", flush=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

print("\n\n--- Generation Complete ---")

with open(output_filename, "w") as f:
    json.dump(analysis_log, f, indent=2)

print(f"Detailed analysis saved to: {output_filename}")