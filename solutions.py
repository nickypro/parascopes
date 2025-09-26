# %%
from transformerlens import HookedTransformer

model = HookedTransformer.from_pretrained("llama-3.2-3b-instruct")

model.generate("Hello, how are you?")