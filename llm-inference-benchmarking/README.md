# LLM Inference Benchmarking

Based on the huggingface llm inference benchmarking repo:
https://github.com/huggingface/inference-benchmarker

Quickstart:
1. Create container from base image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
2. Start container with HF model weights mounted `/data/<dataset-id>`
3. Install python etc. inc. curl
```
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano curl
```
4. Create and source venv
```
python3 -m virtualenv /opt/venv
source /opt/venv/bin/activate
```
5. Install requirements
```
pip install -r requirements.txt
```
6. Start the vLLM server
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --model /data/<dataset-id> --tensor-parallel-size 4 --host 0.0.0.0 --port 8000
```
7. Curl server from local
```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "/data/<dataset-id>",
    "messages": [
      {"role": "user", "content": "Hello from my laptop"}
    ]
  }'
```
8. Run the `inference-benchmark` binary

- install openssl 
wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb -O ~/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
dpkg -i ~/libssl1.1_1.1.1f-1ubuntu2_amd64.deb

apt update && apt install libssl-dev pkg-config

- install cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

- clone the inference-benchmark repo
https://github.com/huggingface/inference-benchmarker

- do surgery
/root/inference-benchmarker/src/lib.rs - line 82 -> "Tokenizer::from_file(run_config.tokenizer_name.clone())"

/root/inference-benchmarker/src/requests.rs - line 414 -> "Tokenizer::from_file(tokenizer)"

todo: inference-benchmarker is sending the curl request to the server, and trying to use
the tokenizer "name" (which it adopts as the "model name") to generate the curl request.
need to enable passing of an actual "model-name" as well as the "tokenizer-name" and get 
the inference-benchmarker to use this as the "model" in the curl request.

... or maybe there's a sneaky model-name arg?

then build with "cargo build"


9. Profit

