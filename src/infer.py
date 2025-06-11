from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

model = LlamaForCausalLM.from_pretrained("./results")
tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

prompt = "Explain how reinforcement learning works."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

