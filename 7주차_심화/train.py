import json
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb


# 🔹 1. 데이터 로드 및 준비
# corpus.json 파일 로드
with open("corpus2.json", "r", encoding="utf-8") as file:
    corpus_data = json.load(file)

# 🔹 2. 학습/검증 데이터 분할
train_data, val_data = train_test_split(corpus_data, test_size=0.2, random_state=42)

# 🔹 3. 데이터셋 생성
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data)
})

# 🔹 4. 모델 및 토크나이저 설정
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
#model_name = "google/gemma-2-2b-it"
#model_name = "facebook/opt-350m"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔹 5. 데이터 포맷팅 함수
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['term'])):
        text = f"### Term: {example['term'][i]}\n ### Description: {example['description'][i]}"
        output_texts.append(text)
    return output_texts

# 🔹 6. 데이터 콜레이터 설정
response_template = " ### Description:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 🔹 7. W&B 설정
wandb.init(project="instruction_tuning", name="Llama-3.2-1B-Instruct")


output_dir = "./llama"

# 🔹 10. SFT Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10000,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=0,
        save_steps=0
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)


# 🔹 11. 학습 시작
trainer.train()

wandb.finish()