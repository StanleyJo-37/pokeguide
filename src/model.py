
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print(torch.cuda.is_available())  # MUST be True
print(torch.version.cuda)

model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
adapter_path = "../unsloth_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()
torch.set_grad_enabled(False)

print("✅ Model loaded and ready!")

@torch.inference_mode()
def generate(prompt, max_new_tokens=256):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
