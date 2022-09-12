import streamlit as st
import transformers
import model
from model import load_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#Load model
model, tokenizer, device = load_model()

def infer(input_ids, max_length, temperature):

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
    )

    return output_sequences
default_value = """Instruction: 
Answer:"""

#prompts
st.title("Generate code with the GPT codegen 🦄")
st.subheader("This machine learning model is trained on 16 billion parameters and it generates code, you can give in a instruction as a prompt")
st.button("GENERATE CODE", on_click=st.write)

sent = st.text_area("Prompt", default_value, height = 300)
max_length = st.sidebar.slider("Max Length", min_value = 500, max_value=3000)
temperature = st.sidebar.slider("Temperature", value = 1.0, min_value = 0.0, max_value=1.0, step=0.05)

input_ids = tokenizer(default_value, return_tensors="pt").input_ids.to(device)
output_sequences = infer(input_ids, max_length, temperature)
generated_ids = model.generate(output_sequences)
generated_text = tokenizer.decode(input_ids[0])

st.write(print(generated_text))