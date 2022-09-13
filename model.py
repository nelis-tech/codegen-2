import streamlit as st
import transformers
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-16B-multi")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-multi")
    return model, tokenizer