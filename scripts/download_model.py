import os
import shutil
from huggingface_hub import hf_hub_download

DEST = "triton_models/resnet50/1/model.onnx"

def download():
    print("Downloading ResNet50 ONNX from HuggingFace...")
    path = hf_hub_download(
        repo_id="onnxmodelzoo/resnet50-v1-7",
        filename="resnet50-v1-7.onnx",
        local_dir="triton_models/resnet50/1/"
    )
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    shutil.move(path, DEST)
    print(f"✅ Model saved to {DEST}")

if __name__ == "__main__":
    download()
