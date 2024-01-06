import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

class InferlessPythonModel:
    def initialize(self):
        model_id = "tenyx/TenyxChat-7B-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="fp4",
           bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config,trust_remote_code=True)


    def infer(self, inputs):
        prompt = inputs["prompt"]
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model_nf4.generate(**inputs, max_length=256)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {'generated_result': output}

    def finalize(self):
        pass