import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

class InferlessPythonModel:
    def initialize(self):
        model_id = "tenyx/TenyxChat-8x7B-v1"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(new_model, trust_remote_code=True, quantization_config=bnb_config, device_map="cuda")
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,device="cuda")
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        messages = [{"role": "system", "content":prompt}]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self.pipe(prompt, max_new_tokens=256, do_sample=True, top_p=0.9,temperature=0.9)
        generated_text = out[0]["generated_text"][len(prompt):]
        return {'generated_result': generated_text}

    def finalize(self):
        pass