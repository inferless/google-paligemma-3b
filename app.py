import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

class InferlessPythonModel:
    def initialize(self):
        model_id = "google/paligemma-3b-mix-224"
        device = "cuda:0"
        dtype = torch.bfloat16
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"],revision="bfloat16")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,torch_dtype=dtype,device_map=device,revision="bfloat16").eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(self,inputs):
        prompt = inputs["prompt"]
        image_url = inputs["image_url"]
        image = Image.open(requests.get(image_url, stream=True).raw)
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        return {'response': decoded}

    def finalize(self):
        pass
