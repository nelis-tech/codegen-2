import streamlit as st
import transformers
import model
from model import load_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#Load model
device, model, tokenizer = load_model()


def infer(input_ids, max_length, temperature):

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1000,
        temperature=temperature,
        do_sample=True,
    )
    return output_sequences

#Prompts
st.title("Generate code with codegen 🦄")
st.subheader("This machine learning model is trained on 16 billion parameters and it generates code, you can give in a instruction as a prompt and don't fill in the answer")

text_target = st.text_area(label = "Enter your instruction and leave the answer open for the generated code, if you want to set the parameters or a new prompt please press stop top right, set the parameters and rerun", value ="""Instruction: Generate python code for a diffusion model
Answer:""", height = 300)
temperature = st.slider("Temperature", value = 0.9, min_value = 0.0, max_value=1.0, step=0.1)
max_length = 1000

#Generate
with st.spinner("AI is at work......"):
    input_ids = tokenizer(text=text_target, return_tensors="pt").input_ids
    output_sequences = infer(input_ids, max_length, temperature)
    generated_ids = model.generate(output_sequences)
    generated_text = tokenizer.decode(generated_ids[0])
st.success("AI Succesfully generated code")
print(generated_text)

st.text(generated_text)
