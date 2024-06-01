from transformers import AutoModelForCausalLM, AutoTokenizer

print("Staring up...")
GPU_DEVICE = 0 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map=GPU_DEVICE)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")



def generate(prompt):
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    model.to(GPU_DEVICE)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    return tokenizer.batch_decode(generated_ids)[0]

while True:
    print("Mistral 7b generator script 🖥️")
    prompt = input("Enter the prompt \n")
    generated =  generate(prompt)
    print(generated)
