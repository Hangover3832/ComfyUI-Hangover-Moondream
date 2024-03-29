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
    HUGGINGFACE_MODEL_NAMES = ["vikhyatk/moondream2",] # other/newer models can be added here
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
                "trust_remote_code": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "Hangover"

    def interrogate(self, image:torch.Tensor, prompt:str, huggingface_model:str, device:str, trust_remote_code:bool):
        dev = "cuda" if device.lower() == "gpu" else "cpu"
        if (self.model == None) or (self.tokenizer == None) or (self.modelname != huggingface_model) or (device != self.device):
            del self.model
            del self.tokenizer
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            print(f"moondream: loading model {huggingface_model}, please stand by....")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(huggingface_model, trust_remote_code=trust_remote_code).to(dev)
            except ValueError:
                print("Moondream: You have to trust remote code to use this node!")
                return ("You have to trust remote code execution to use this node!",)
            
            self.tokenizer = Tokenizer.from_pretrained(huggingface_model)
            self.modelname = huggingface_model
            self.device = device

        descriptions = ""
        
        for im in image:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            enc_image = self.model.encode_image(img)
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            descriptions += answer
        
        return(descriptions,)
