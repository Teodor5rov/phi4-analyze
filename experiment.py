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
ENTROPY_THRESHOLD = 0.35

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

user_prompt_content = """Suppose the positive integers are partitioned in n>1 arithmetic progressions. Does it follow necessarily that two of them have the same jump?"""

extracted_cot = """<think>We are asked: "Suppose the positive integers are partitioned in n>1 arithmetic progressions. Does it follow necessarily that two of them have the same jump?" The problem: partition positive integers into n arithmetic progressions, where n>1. We want to check if it necessarily implies that two of them have the same "jump" (which is the difference, i.e. common difference or "step size" of the arithmetic progression). This is essentially a combinatorial arithmetic structure: Given a partition of positive integers into arithmetic progressions, do two progressions necessarily have the same common difference?

The problem is reminiscent of "Van der Waerden's theorem", but likely a pigeonhole principle argument. Alternatively, is it possible to partition the positive integers into n arithmetic progressions, each with a different jump? We want to know if there is a partition with n arithmetic progressions with distinct differences, or is it forced that at least two of them have same difference? The problem is phrasing "Does it follow necessarily that two of them have the same jump?" It seems the answer is "Yes, if n>1 then among the arithmetic progressions, at least two have the same difference."

But let me check: Suppose we partition the positive integers into arithmetic progressions. In each arithmetic progression, if we denote the progression as a = a0, a0 + d, a0 + 2d, etc. But the partition is of the entire set of positive integers. There are n arithmetic progressions, each one is an arithmetic progression with some positive common difference. We want to see if two of them necessarily share the same difference.

Let's consider a partition of the positive integers into arithmetic progressions. Let them be A1, A2, ... A_n. Let d_i be the difference for A_i. Is it possible that all d_i are different? Let's check: One might attempt a partition: Partition into residue classes modulo a certain number. If we take a modulus m, the common difference is m, and there are m residue classes. But here m equals n maybe. But in that case, all differences are equal. So that's not a counterexample. But can we partition with different differences? Perhaps a partition where the jumps are powers of 2? Let's see: Partition the positive integers into two arithmetic progressions. For example, partition into evens and odds. Then the difference for evens is 2, the difference for odds is 2 as well. So that is same difference. Wait maybe partition into one arithmetic progression with difference 2 and one with difference 3? But then do they cover the entire set? Let's try: Partition positive integers into those congruent to 1 mod 2 and those congruent to 0 mod 2. They both have difference 2. If we try partition into 2 AP's with distinct differences, can we get that? Let's try: A1: numbers that are 1 mod 3, A2: numbers that are 2 mod 3, but then 0 mod 3 is missing. But then not partition of all positive integers. Alternatively, try: Partition the positive integers as: A1: {1, 2, 3, 4, 5, ...} with common difference something? But that's not an arithmetic progression.

Wait, I recall a classic result: "Every partition of the natural numbers into finitely many arithmetic progressions must have at least two progressions with the same difference." I suspect the answer is yes, two of them have the same jump. But let me try to prove it using "minimizing" and "maximizing" expressions. "Minimizing" or "maximizing" the sum of the reciprocals of the differences. For example, if we partition the positive integers, then the density of an arithmetic progression with difference d is 1/d. So if we sum the densities, we get something like sum_{i=1}^n (1/d_i) = 1. But if all the d_i's are distinct, then the minimum possible sum is when they are the smallest possible numbers, i.e. d_1=1, d_2=2, ... but then sum would be 1 + 1/2 + 1/3 + ... which diverges or is at least larger than 1. But wait, let me check: The partition covers all positive integers, so the sum of densities is 1. And the density of an arithmetic progression with difference d is 1/d. So if we had n distinct jumps d_1, d_2, ... d_n, then the sum is at most 1/d_min + 1/d_2 + ... but the maximal"""

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
        sentropy_value = - (unfiltered_probs**2 * torch.log(unfiltered_probs + 1e-12)).sum().item()

        if TOP_K_SAMPLING > 0:
            top_k_vals, _ = torch.topk(logits, TOP_K_SAMPLING)
            logits[logits < top_k_vals[:, -1, None]] = -float("inf")

        sampling_probs = torch.softmax(logits, dim=-1)
        variance_value = torch.var(sampling_probs).item()
        # if variance_value < VARIANCE_THRESHOLD:
        #     print(f"\033[1m (VARIANCE: {variance_value:.2e}) \033[0m", end="")
        if sentropy_value > ENTROPY_THRESHOLD:
             print(f"\033[1m (SENTROPY: {sentropy_value:.2e}) \033[0m", end="")

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
            "sentropy": sentropy_value,
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