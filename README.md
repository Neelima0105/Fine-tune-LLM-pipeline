# Fine-Tuning a Small LLM (LoRA) for a Customer Support Assistant

An end-to-end pipeline for fine-tuning TinyLlama-1.1B-Chat with LoRA to build a domain-specific customer support assistant — dataset prep, supervised fine-tuning with experiment tracking, a FastAPI inference backend, and a React frontend.

## Features

• Parameter-efficient fine-tuning (LoRA via PEFT) on TinyLlama-1.1B-Chat
• Supervised fine-tuning using Hugging Face TRL's SFTTrainer
• Custom instruction/response dataset (JSONL)
• Experiment tracking with MLflow (params, metrics, LoRA adapter artifacts)
• FastAPI backend for serving the fine-tuned model
• React chat frontend

## Tech Stack

Python, Hugging Face Transformers, PEFT (LoRA), TRL, MLflow, FastAPI, React

## Project Structure

```
Fine-tune-LLM-pipeline/
training/
  train_model.py   LoRA fine-tuning + MLflow tracking
  inference.py     Load fine-tuned model and run inference
data/                Instruction/response training data (JSONL)
backend/              FastAPI serving endpoint
frontend/             React chat UI
```

## Setup and Run

```bash
cd training
pip install -r ../backend/requirements.txt
python train_model.py
mlflow ui

cd ../backend
pip install -r requirements.txt
uvicorn main:app --reload

cd ../frontend
npm install
npm run dev
```

## Resume Bullet

Fine-tuned TinyLlama-1.1B using LoRA (PEFT) and Hugging Face TRL to build a domain-specific customer support assistant, tracked experiments with MLflow, and served the model via a FastAPI backend with a React chat interface.
