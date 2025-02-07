import json
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb


# 🔹 1. 데이터 로드 및 준비
# corpus.json 파일 로드
with open("corpus.json", "r", encoding="utf-8") as file:
    corpus_data = json.load(file)

# 🔹 2. 학습/검증 데이터 분할
train_data, val_data = train_test_split(corpus_data, test_size=0.2, random_state=42)

# 🔹 3. 데이터셋 생성
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data)
})

# 🔹 4. 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
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
wandb.init(project="facebook_opt_instruction_tuning", name=model_name)

# 🔹 8. 로그 함수
from transformers import TrainerCallback, TrainerState, TrainerControl
class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """훈련 중 매 로그 이벤트 발생 시 호출됩니다."""
        if logs is not None:
            # train loss 기록
            if "loss" in logs:
                wandb.log({"train/loss": logs["loss"], "step": state.global_step})

            # validation 평가 및 loss 기록 (평가 주기에 따라 실행됨)
            if "eval_loss" in logs:
                wandb.log({"eval/loss": logs["eval_loss"], "step": state.global_step})

output_dir = "./instruction_facebook"

# 🔹 9. SFT Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        logging_dir="./logs",
        logging_steps=100,
        eval_steps=100,
        save_total_limit=1,
        save_steps=500,
        evaluation_strategy="steps",  # 평가 전략을 'steps'로 설정
        load_best_model_at_end=True,  # 가장 낮은 평가 손실을 갖는 모델을 저장
        metric_for_best_model="eval_loss",  # 가장 낮은 eval_loss를 기준으로 모델 선택
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    callbacks=[WandbLoggingCallback()]  # 콜백 추가
)


# 🔹 10. 학습 시작
trainer.train()

wandb.finish()