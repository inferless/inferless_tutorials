from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
        repo_id = "google/gemma-7b"
        # Use Inferless volumes to store your model
        # Replace the volume_name with your volume
        model_store = f"/var/nfs-mount/common_llm/{repo_id}"
        os.makedirs(f"/var/nfs-mount/common_llm/{repo_id}", exist_ok=True)
        
        snapshot_download(
                    repo_id,
                    local_dir=model_store,
                    # Hugging face token is required for gated model
                    token="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",
                    ignore_patterns=["*.gguf"])
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=256)
        self.llm = LLM(model=model_store,gpu_memory_utilization=0.9)

    def infer(self, inputs):
        prompts = inputs["prompt"]
        result = self.llm.generate(prompts, self.sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {'gresult': result_output[0]}

    def finalize(self):
        pass
