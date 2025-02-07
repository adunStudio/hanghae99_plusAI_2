from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

"""
- **Instruction**: 어떤 코드를 짜야할지에 대한 가이드를 제공합니다.
- **Input(optional)**: 코드가 입력으로 받을 input을 제공합니다.
- **Output:** Instruction에 맞춰 구현한 코드입니다.
"""

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

