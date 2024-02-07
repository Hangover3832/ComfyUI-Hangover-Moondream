'''
An implementation of the moondream visual LLM
https://huggingface.co/vikhyatk/moondream1

https://github.com/Hangover3832

Alex
'''


from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image
import torch
import gc
import numpy as np

class Moondream:
    HUGGINGFACE_MODEL_NAMES = ["vikhyatk/moondream1",] # other/newer models can be added here
    DEVICES = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.modelname = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": False, "default": "Please provide a detailed description of this image."},),
                "huggingface_model": (s.HUGGINGFACE_MODEL_NAMES, {"default": s.HUGGINGFACE_MODEL_NAMES[0]},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "Hangover"

    def interrogate(self, image:torch.Tensor, prompt:str, huggingface_model:str, device:str):
        dev = "cuda" if device.lower() == "gpu" else "cpu"
        if (self.model == None) or (self.tokenizer == None) or (self.modelname != huggingface_model) or (device != self.device):
            del self.model
            del self.tokenizer
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"moondream: loading model {huggingface_model}, please stand by....")
            self.model = AutoModelForCausalLM.from_pretrained(huggingface_model, trust_remote_code=True).to(dev)
            self.tokenizer = Tokenizer.from_pretrained(huggingface_model)
            self.modelname = huggingface_model
            self.device = device

        descriptions = ""
        
        for im in image:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            enc_image = self.model.encode_image(img)
            answer = self.model.answer_question(enc_image, "What is this?", self.tokenizer)
            descriptions += answer
        
        return(descriptions,)
