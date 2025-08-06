from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Modelo de Google: Gemma 2B instruct
MODEL_ID = "google/gemma-2-2b-it"

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    use_auth_token=HF_API_TOKEN
)

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.post("/generate")
async def generate_text(request: PromptRequest):
    # Agrega formato tipo chat si deseas estilo instructivo
    formatted_prompt = f"<start_of_turn>user\n{request.prompt}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": generated_text}
