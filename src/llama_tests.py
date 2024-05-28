import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer


model_path = "/mnt/shared/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

model = LlamaForCausalLM.from_pretrained(model_path)

tokenizer = LlamaTokenizer.from_pretrained(model_path)


pipeline = transformers.pipeline("text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                )

sequences = pipeline('I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=400)

for seq in sequences:
    print(f"{seq['generated_text']}")