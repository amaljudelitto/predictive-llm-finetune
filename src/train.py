from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    model_name = "NousResearch/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    train_dataset = load_dataset("data/train.txt", tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model("./results")
    tokenizer.save_pretrained("./results")

if __name__ == "__main__":
    main()

