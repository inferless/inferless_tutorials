import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class InferlessPythonModel:
    def initialize(self):
        model_id = "tenyx/TenyxChat-7B-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True)
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