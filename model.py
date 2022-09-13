import streamlit as st
import transformers
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-16B-multi", low_cpu_mem_usage=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-multi")
    return model, tokenizer, device