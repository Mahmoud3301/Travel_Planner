import subprocess
import sys

def install_packages():
    packages = [
        "datasets",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "matplotlib",
        "streamlit",
        "pyngrok",
        "speechrecognition",
        "pydub",
        "librosa",
        "faiss-cpu",
        "sentence-transformers",
        "torch"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

if __name__ == "__main__":
    install_packages()
