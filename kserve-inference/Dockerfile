FROM vllm/vllm-openai:v0.6.3.post1

# RUN apt-get update && \
#    apt-get install nvidia-driver-565 nvidia-utils-565 cuda-toolkit -y
RUN apt-get update && \
    apt-get install nvidia-driver-550 nvidia-utils-550 cuda-toolkit -y
RUN pip install git+https://github.com/huggingface/transformers.git
