import json

with open("outputs/analysis_Phi-4-reasoning_20250621_123636.json", "r", encoding="utf-8") as f:
    data = json.load(f)

text = "".join(step["chosen_token"] for step in data)

with open("reconstructed_text.txt", "w", encoding="utf-8") as f:
    f.write(text)