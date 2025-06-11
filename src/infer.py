from transformers import LlamaTokenizer, LlamaForCausalLM

def infer(prompt):
    tokenizer = LlamaTokenizer.from_pretrained("./results")
    model = LlamaForCausalLM.from_pretrained("./results")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    prompt = "Explain quantum entanglement in simple terms."
    infer(prompt)

