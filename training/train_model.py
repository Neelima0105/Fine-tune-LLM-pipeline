from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import os
import mlflow

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "support_dataset.jsonl")

dataset= load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

dataset = dataset.map(format_example)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config= LoraConfig(
   r=8,
   lora_alpha=16,
   lora_dropout=0.05,
   target_modules="all-linear",
   task_type="CAUSAL_LM"
)
"""ValueError: Your setup doesn't support bf16/gpu. You need to assign use_cpu if you want to train the model on CPU."""
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="models/lora-support-llm",
    num_train_epochs=1,              # keep small for CPU
    per_device_train_batch_size=1,   # CPU friendly
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    report_to="none",

    bf16=False,        # ❌ disable bf16
    fp16=False,        # ❌ disable fp16
    use_cpu=True       # ✅ force CPU
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config = lora_config,
)
trainer.train()
trainer.save_model("model/lora_support_llm")

mlflow.set_experiment("fine-tuned-support-llm")

with mlflow.start_run():
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("epochs", 1)
    mlflow.log_param("batch_size", 1)
    mlflow.log_param("learning_rate", 2e-4)
    mlflow.log_param("lora_r", 8)
    mlflow.log_param("lora_alpha", 16)

    trainer.train()

    trainer.save_model("models/lora-support-llm")

    mlflow.log_artifacts("models/lora-support-llm", artifact_path="lora_adapter")



