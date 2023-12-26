import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class InferlessPythonModel:
    def initialize(self):
        model_id = 'Deci/DeciLM-7B'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",load_in_4bit=True,trust_remote_code=True)
        self.qtq_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def infer(self, inputs):
        prompt = inputs["prompt"]
        out = self.qtq_pipe(prompt, max_new_tokens=256, do_sample=True, top_p=0.9,temperature=0.9)
        generated_text = out[0]["generated_text"][len(prompt):]

        return {'generated_result': generated_text}

    def finalize(self):
        pass