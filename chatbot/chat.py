from dotenv import load_dotenv
from transformers import LlamaForCausalLM, LlamaTokenizer

load_dotenv()

model_id = "meta-llama/Llama-2-7b"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id)

while True:
    prompt = input('Type your question here (type "exit" to exit the program): ')
    if prompt == "exit":
        break
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # answer = chat(query)

    print("The answer: ", answer)