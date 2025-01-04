"""
@author: AlexL
@title: ComfyUI-Hangover-Moondream
@nickname: Hangover-Moondream
@description: An implementation of the moondream visual LLM
"""
# https://huggingface.co/vikhyatk/moondream2

# by https://github.com/Hangover3832


from transformers import AutoModelForCausalLM as AutoModel, CodeGenTokenizerFast as Tokenizer
from PIL import Image
import torch
import gc
import numpy as np
import codecs
import subprocess
import os
import requests
import json
from huggingface_hub import list_repo_refs

def Run_git_status(repo: str):
    """Prints a list of all model tag references for this huggingface repo"""
    
    try:
        print('\033[92m\033[4m[Moondream] model revsion references:\033[0m\033[92m')
        refs = list_repo_refs(repo)
        result = []
        for tag in refs.tags:
            if tag:
                print(f"{tag.name} -> {tag.target_commit}")
    except Exception as e:
        print(f"Error fetching repository references: {e}")
    finally:
        print('\033[0m')


class Moondream:
    HUGGINGFACE_MODEL_NAME: str = "vikhyatk/moondream2"
    DEVICES: str = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]
    Versions: str = 'versions.txt'
    Model_Revisions_URL: str = f"https://huggingface.co/{HUGGINGFACE_MODEL_NAME}/raw/main/{Versions}"
    current_path = os.path.abspath(os.path.dirname(__file__))
    try:
        print("[Moondream] trying to update model versions...", end='')
        response = requests.get(Model_Revisions_URL)
        if response.status_code == 200:
            with open(f"{current_path}/{Versions}", 'w') as f:
                f.write(response.text)
            print('ok')
    except Exception as e:
        if hasattr(e, 'message'):
            msg = e.message
        else:
            msg = e
        print(f'failed ({msg})')

    # read the versions file
    with open(f"{current_path}/{Versions}", 'r') as f:
        versions = f.read()

    # read the special names file
    with open(f"{current_path}/special_names.json") as f:
        special_names: json = json.load(f)
    
    # build the model revisions list and apply special names
    MODEL_REVISIONS: list[str] = [] # [v for v in versions.splitlines() if v.strip()]
    for v in versions.splitlines():
        if v.strip():
            if v in special_names:
                MODEL_REVISIONS.append(f"{v} ({special_names[v]})")
            else:
                MODEL_REVISIONS.append(v)

    print(f"[Moondream] found model versions: {', '.join(MODEL_REVISIONS)}")
    MODEL_REVISIONS.insert(0,'ComfyUI/models/moondream2')

    # Run_git_status(HUGGINGFACE_MODEL_NAME)

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.revision = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Please provide a detailed description of this image."},),
                "separator": ("STRING", {"multiline": False, "default": r"\n"},),
                # "huggingface_model": (s.HUGGINGFACE_MODEL_NAMES, {"default": s.HUGGINGFACE_MODEL_NAMES[-1]},),
                "model_revision": (s.MODEL_REVISIONS, {"default": s.MODEL_REVISIONS[-1]},),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},),
                "trust_remote_code": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "Hangover"

    def interrogate(self, image:torch.Tensor, prompt:str, separator:str, model_revision:str, temperature:float, device:str, trust_remote_code:bool):
        if not trust_remote_code:
            raise ValueError("You have to trust remote code to use this node!")

        model_revision = model_revision[:10]
        dev = "cuda" if device.lower() == "gpu" else "cpu"
        if temperature < 0.01:
            temperature = None
            do_sample = None
        else:
            do_sample = True

        if (self.model == None) or (self.tokenizer == None) or (device != self.device) or (model_revision != self.revision):
            del self.model
            del self.tokenizer
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.revision = model_revision

            print(f"[Moondream] loading model moondream2 revision '{model_revision}', please stand by....")
            if model_revision == Moondream.MODEL_REVISIONS[0]:
                model_name = model_revision
                model_revision = None
            else:
                model_name = Moondream.HUGGINGFACE_MODEL_NAME

            try:
                self.model = AutoModel.from_pretrained(
                    model_name, 
                    trust_remote_code=trust_remote_code,
                    revision=model_revision
                ).to(dev)
                self.tokenizer = Tokenizer.from_pretrained(model_name)
            except RuntimeError:
                raise ValueError(f"[Moondream] Please check if the tramsformer package fulfills the requirements. "
                                  "Also note that older models might not work anymore with newer packages.")

            self.device = device

        descriptions = ""
        prompts = list(filter(lambda x: x!="", [s.lstrip() for s in prompt.splitlines()])) # make a prompt list and remove unnecessary whitechars and empty lines
        if len(prompts) == 0:
            prompts = [""]

        try:
            for im in image:
                i = 255. * im.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                enc_image = self.model.encode_image(img)
                descr = ""
                sep = codecs.decode(separator, 'unicode_escape')
                for p in prompts:
                    answer = self.model.answer_question(enc_image, p, self.tokenizer, temperature=temperature, do_sample=do_sample)
                    descr += f"{answer}{sep}"
                descriptions += f"{descr[0:-len(sep)]}\n"
        except RuntimeError:
            raise ValueError(f"[Moondream] Please check if the tramsformer package fulfills the requirements. "
                                  "Also note that older models might not work anymore with newer packages.")
        
        return(descriptions[0:-1],)
