import json
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import EarlyStoppingCallback
import evaluate
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
model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 🔹 5. 데이터 포맷팅 함수
def formatting_prompts_func(examples):
    formatted_prompts = []

    for i, example in enumerate(examples):
        # 하나의 입력에 대해 포맷팅된 텍스트를 생성
        input_data = examples["input"][i]
        output_terms = examples["output"][i]

        # input을 텍스트로 변환
        input_text = (
            f"### Title: {input_data['title']}\n"
            f"### Description: {input_data['description']}\n"
            f"### Summary: {input_data['summary']}\n"
            f"### 핵심 용어 리스트를 생성하세요.\n"
        )

        # output을 키포인트별로 포맷팅
        output_text = "\n".join(
            [f"- **{item['term']}**: {item['description']}" for item in output_terms]
        )

        # 최종 프롬프트 결합
        formatted_prompt = f"### Instruction:{input_text}\n### Response:\n{output_text}"

        # 리스트에 추가
        formatted_prompts.append(formatted_prompt)

    return formatted_prompts

# 🔹 6. 데이터 콜레이터 설정
response_template = "### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 🔹 7. W&B 설정
wandb.init(project="llama-instruction-tuning", name="google/llama-2-2b-it")

# 🔹 8. ROUGE 및 BLEU 메트릭 로드
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# 🔹 9. 평가 함수 정의
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 🔸 BLEU 점수 계산
    predictions = [pred.split() for pred in decoded_preds]
    references = [[label.split()] for label in decoded_labels]
    bleu = bleu_metric.compute(predictions=predictions, references=references)

    # 🔸 ROUGE 점수 계산
    rouge = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # 🔸 W&B로 메트릭 기록
    wandb.log({
        "bleu_score": bleu["bleu"],
        "rouge1": rouge["rouge1"].mid.fmeasure,
        "rouge2": rouge["rouge2"].mid.fmeasure,
        "rougeL": rouge["rougeL"].mid.fmeasure,
    })

    return {
        "bleu_score": bleu["bleu"],
        "rouge1": rouge["rouge1"].mid.fmeasure,
        "rouge2": rouge["rouge2"].mid.fmeasure,
        "rougeL": rouge["rougeL"].mid.fmeasure,
    }

#output_dir = "./finetuned_llama"
output_dir = "./finetuned_llama"

# 🔹 10. SFT Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=100,
        load_best_model_at_end=True,  #
        metric_for_best_model="bleu_score",
        evaluation_strategy="steps",
        save_total_limit=2,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    compute_metrics=compute_metrics,  # 메트릭 계산 함수 추가
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # 🔹 조기 종료 콜백 추가
)


trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# 🔹 11. 학습 시작
last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint is None:
    trainer.train()
else:
    trainer.train(resume_from_checkpoint=last_checkpoint)


wandb.finish()