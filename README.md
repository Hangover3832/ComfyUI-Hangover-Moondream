# ConfyUI-Hangover-Moondream
[Moondream](https://huggingface.co/vikhyatk/moondream1) is a lightweight multimodal large languge model.


ℹ️ **IMPORTANT: According to the creator, Moondream is for research purposes only, commercial use is not allowed!**

⚠️ **WARNING: Additional python code will be downloaded from huggingface and executed. You have to trust this creator if you want to use this node!**


![Alt text](images/moondream_workflow.png)

👍 For testing, research and fun. There might be issues when loading this node, and/or additional packages needs to be installed. It worked fine after I reinstalled the latest release of ComfyUI.

## Updates
  * Now passing proper prompt to the model 🐞
  * Model/code update [moondream2](https://huggingface.co/vikhyatk/moondream2). This should resolve issues with the 'Tensor size mismatch' error when using newer versions of transformers. **Make sure that you select the moondream2 model within the node to receive the updated files from huggingface.** If moondream2 works fine, you can remove the old model files, usually located in the user\\.cache\huggingface folder: hub\models--vikhyatk--moondream1 and modules\transformers_modules\vikhyatk\moondream1 to save disk space.🔄

## To do
🔜