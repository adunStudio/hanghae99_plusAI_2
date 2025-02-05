import os
import sys
import math
import torch
from fontTools.misc.cython import returns

import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

# HuggingFace에서 데이터셋을 로드하기 위한 함수
from datasets import load_dataset

from transformers import (
    AutoConfig,           # 사전 학습된 모델의 설정을 자동으로 가져오기
    AutoModelForCausalLM, # 사전 학습된 causal language model을 자동으로 가져오기
    AutoTokenizer,        # 사전 학습된 토크나이저 자동 로드
    HfArgumentParser,     # 커맨드 라인 인자를 파싱하기 위한 도구
    Trainer,              # 훈련을 간소화하기 위한 HuggingFace Trainer 클래스
    TrainingArguments,    # 훈련 설정 인자를 정의하는 클래스
    default_data_collator # 데이터를 모델에 맞게 정렬하는 기본 데이터 collator
)

# 마지막 체크포인트 가져오는 함수
from transformers.trainer_utils import get_last_checkpoint



# Weights & Biases 프로젝트 초기화
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning'



# 데이터 클래스 정의: 훈련에 필요한 인자들 설정
@dataclass
class Arguments:
    # HuggingFace hub에서 pre-trained 모델로 사용할 모델의 이름
    model_name_or_path: Optional[str] = field(default=None)

    # 우리 모델의 precision(data type이라고 이해하시면 됩니다)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})

    # Fine-tuning으로 사용할 huggingface hub에서의 dataset 이름
    dataset_name: Optional[str] = field(default=None)

    # Fine-tuning으로 사용할 huggingface hub에서의 dataset configuration
    # 데이터셋의 구성 이름 (예: train/test/validation)
    dataset_config_name: Optional[str] = field(default=None)

    # Fine-tuning에 사용할 input text의 길이
    # 텍스트를 나눌 블록의 크기
    block_size: int = field(default=1024)

    # Data를 업로드하거나 전처리할 때 사용할 worker 숫자
    num_workers: Optional[int] = field(default=None)



# ArgumentParser를 사용하여 커맨드 라인 인자들을 파싱
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()



# 로깅 설정
logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # 로그 출력 형식
    datefmt="%m/%d/%Y %H:%M:%S",  # 날짜 형식
    handlers=[logging.StreamHandler(sys.stdout)],  # 로그를 stdout으로 출력
)

# 로그 설정이 가능하면 INFO 레벨로 로깅
if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()  # INFO: 20

# 우리가 가지고 있는 logger와 HuggingFace의 logger의 log level 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# 기타 HuggingFace logger option들을 설정
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# 훈련/평가 설정 출력
logger.info(f"Training/evaluation parameters {training_args}")




# 데이터셋 로드
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# 모델, 토크나이저, 설정 파일 로드
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# 토크나이저의 pad 토큰을 eos 토큰으로 설정
# <PAD>: 시퀀스의 길이를 맞추기 위한 패딩 토큰. 시퀀스들의 길이를 동일하게 맞춰주기 위해 추가되는 토큰이다.
# <EOS>: 문장의 끝을 나타내는 토큰. 번역이나 생성 모델에서 사용되어 문장 생성의 종료를 표시한다.
# 우리가 사용하는 tokenizer는 padding token이 없어서 추가해줍니다.
# GPT 계열은 Casual LM으로 입력 시퀀스의 길이를 동적으로 처리하므로 PAD 토큰이 없다.
# 이러한 경우 패딩이 필요한 상황에서는 pad_token을 명시적으로 설정해야 하며, 보통 eos_token을 패딩 역할로 사용한다.
# -> (이는 tokenizer마다 다르니 유의)
tokenizer.pad_token_id = tokenizer.eos_token_id

# 토큰 임베딩 크기를 토크나이저 크기에 맞게 조정
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# 텍스트 컬럼 이름 설정
column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

# 토크나이저로 텍스트를 토큰화하는 함수 정의
def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output


# 데이터셋에 토큰화 함수 적용
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,  # 배치 단위로 처리
        num_proc=args.num_workers,  # 병렬 워커 수
        remove_columns=column_names  # 토큰화 후 원래 컬럼 삭제
    )


# 최대 위치 임베딩 크기와 블록 크기 설정
max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)


# 텍스트들을 그룹화하는 함수 정의
def group_texts(examples):
    # 주어진 text들을 모두 concat 해줍니다.
    # 예를 들어 examples = {'train': [['Hello!'], ['Yes, that is great!']]}이면 결과물은 {'train': ['Hello! Yes, that is great!']}가 됩니다.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  # 모든 텍스트를 연결

    # 전체 길이를 측정합니다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])  # 전체 길이 계산
    total_length = (total_length // block_size) * block_size  # 블록 크기에 맞춰 길이 조정


    # 블록 단위로 텍스트를 분할
    # 예를 들어 block_size=3일 때 {'train': ['Hello! Yes, that is great!']}는
    # {'train': ['Hel', 'lo!', ' Ye', 's, ', 'tha', ...]}가 됩니다.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # 레이블은 입력 ID와 동일하게 설정
    # Next token prediction이니 label은 자기 자신으로 설정합니다.
    result["labels"] = result["input_ids"].copy()
    return result

# 그룹화된 텍스트 데이터셋 생성
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )


# 학습 및 Validation 데이터셋 준비
train_dataset      = lm_datasets["train"]
validation_dataset = lm_datasets["validation"]


# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,  # validation 데이터셋 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator  # 기본 데이터 collator 사용
)

# 체크포인트 설정
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 마지막 체크포인트 가져오기
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint  # 지정된 체크포인트에서 시작
else:
    checkpoint = last_checkpoint  # 마지막 체크포인트가 있으면 사용


# 모델 훈련
train_result = trainer.train(resume_from_checkpoint=checkpoint)

# 훈련 후 모델과 메트릭 저장
trainer.save_model()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


# Validation 평가 수행
eval_metrics = trainer.evaluate()  # validation 데이터셋을 사용하여 평가 수행

# Wandb에 평가 메트릭 기록
trainer.log_metrics("eval", eval_metrics)
wandb.log({"train/loss": metrics["loss"], "eval/loss": eval_metrics["eval_loss"]})