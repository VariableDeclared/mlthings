#!/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer
from pynvml import *

print("Starting up...")
GPU_DEVICE = 0 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map=GPU_DEVICE)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


def print_gpu_util():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    nvmlDeviceGetHandleByIndex(0)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def generate(prompt):
    print(f"Generating from prompt; {prompt}")
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    model.to(GPU_DEVICE)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    return tokenizer.batch_decode(generated_ids)[0]

def main():
    print("GPU Util;")
    print_gpu_util()
    while True:
        print("Mistral 7b generator script 🖥️")
        prompt = input("Enter the prompt \n")
        generated =  generate(prompt)
        print(generated)

if __name__ == "__main__":
    main()
