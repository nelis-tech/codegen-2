import streamlit as st
import transformers
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    return model, tokenizer, device