from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):

        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=256)
        self.llm = LLM(model="Inferless/SOLAR-10.7B-Instruct-v1.0-GPTQ", quantization="gptq", dtype="float16")

    def infer(self, inputs):
        prompts = inputs["prompt"]
        result = self.llm.generate(prompts, self.sampling_params)
        result_output = [[[output.outputs[0].text,output.outputs[0].token_ids] for output in result]

        return {'generated_result': result_output[0]}

    def finalize(self):
        pass