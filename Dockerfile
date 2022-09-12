conda create --name my_env python=3.9
conda activate my_env
pipenv install streamlit
pip install streamlit --upgrade
pip install torch
pip install bitsandbytes-cuda112==0.26.0.post2
pip install datasets
pip install accelerate
pip install sentencepiece
pip install git+https://github.com/huggingface/transformers.git
