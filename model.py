import streamlit as st
import transformers
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("moyix/codegen-16B-multi-gptj", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-multi") and AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return device, model, tokenizer
