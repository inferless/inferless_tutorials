import json
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
import base64
from io import BytesIO


class InferlessPythonModel:
  def initialize(self):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    self.pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo",vae=vae, torch_dtype=torch.float16, variant="fp16",use_safetensors=True)
    self.pipeline = self.pipeline.to("cuda")
    self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)

  def infer(self, inputs):
    prompt = inputs["prompt"]
    pipeline_output_image = self.pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    buff = BytesIO()
    pipeline_output_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return {"generated_image_base64": img_str.decode('utf-8')}

  def finalize(self,args):
    self.generator = None
    
