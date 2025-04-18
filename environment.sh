pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
pip install facenet-pytorch scikit-learn pandas numpy matplotlib

pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download DAMO-NLP-SG/VideoRefer-7B --local-dir ./checkpoints