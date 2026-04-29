from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from pydantic import BaseModel

app = FastAPI()

BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_MODEL="model/lora_support_llm"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model=AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

class Query(BaseModel):
    question : str

@app.post("/ask")
def ask(query: Query):
    prompt = f"### Instruction:\n{query.question}\n\n### Response:\n"
    result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return {"answer": result[0]["generated_text"]}

