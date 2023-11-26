import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download


class InferlessPythonModel:
    def initialize(self):
        snapshot_download(
            "TheBloke/CodeLlama-34B-Python-GPTQ",
            local_dir="/model",
            token="<<your_token>>",
        )
        self.llm = LLM(
          model="/model",
          quantization="gptq")
    
    def infer(self, inputs):
        prompts = inputs["prompt"]
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1,
            max_tokens=512
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"result": result_output[0]}

    def finalize(self, args):
        pass