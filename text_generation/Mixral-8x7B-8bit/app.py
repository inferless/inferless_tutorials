import contextlib
from get_model import model_initialize,encode_tokens,generate
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
        repo_id = "Inferless/Mixtral-8x7B-v0.1-int8-GPTQ"
        model_store = f"/home/{repo_id}"
        os.makedirs(f"/home/{repo_id}", exist_ok=True)
        snapshot_download(repo_id,local_dir=model_store)
        self.tokenizer, self.model = model_initialize(f"{model_store}/model_int8.pth")
        self.callback = lambda x : x

    def infer(self, inputs):
        prompt= inputs['prompt']
        encoded = encode_tokens(self.tokenizer,prompt, bos=True, device="cuda")
        prof = contextlib.nullcontext()
        with prof:
            y, metrics = generate(
                    self.model,
                    encoded,
                    max_new_tokens=256,
                    draft_model=None,
                    speculate_k=5,
                    interactive=False,
                    callback=self.callback,
                    temperature=0.8,
                    top_k=200,)

        return {'generated_result': self.tokenizer.decode(y.tolist())}

    def finalize(self):
        pass