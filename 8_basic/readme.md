# 8주차 기본과제
Instruction-tuning vs LoRA(Rank 8) vs LoRA(Rank 128) vs LoRA(Rank 256)
![result.png](./result.png)
- [train/loss](https://api.wandb.ai/links/iamkimhongil92-lumenasoft/qbrgvsot)
- [eval/loss](https://api.wandb.ai/links/iamkimhongil92-lumenasoft/5cy5d494)
- [train_runtime](https://api.wandb.ai/links/iamkimhongil92-lumenasoft/l5aefrku)
- [max_memory_allocated_gb](https://api.wandb.ai/links/iamkimhongil92-lumenasoft/ri9ubu2z)

Instruction-tuning이 여러 LoRA 설정에 비해 성능이 더 좋아 보입니다.

## 1. train/loss 및 eval/loss 그래프
- 모든 설정에서 훈련이 진행됨에 따라 손실 값이 꾸준히 감소하는 경향을 보입니다.
- Instruction-tuning이 학습 중 더 빠르게 손실(loss)을 감소시키며, 최종 손실 값도 LoRA Rank 설정들에 비해 낮습니다.
- Rank 8, 128, 256은 손실 감소가 다소 느리며 랭크가 높을 수록 조금 더 낮은 loss를 보입니다.

## 2. train_runtime 그래프
- Rank 8은 가장 짧은 훈련 시간을 소요했습니다.
- Instruction-tuning이 LoRA Rank 128 및 256보다 훈련 시간이 짧습니다.
- Rank 256은 가장 긴 훈련 시간을 소요했습니다.

## 3. max_memory_allocated_gb 그래프
- Instruction-tuning은 메모리 사용량이 가장 낮습니다.
- Rank 8에서는 Rank 128, 256에 비해 메모리 사용량이 낮습니다.

## 4. 전반적인 성능 비교
- Instruction-tuning은 빠른 손실 감소와 우수한 최종 성능을 보여주며, 훈련 시간도 비교적 짧습니다.
- LoRA Rank 256은 Instruction-tuning에 가까운 성능을 보이지만, 훈련 시간이 길고 메모리 사용량도 높습니다.
- Rank 128과 Rank 8은 성능이 떨어지며, 특히 Rank 8에서는 충분한 최적화가 이루어지지 않은 것으로 보입니다.

## 5. 종합 해석
- Instruction-tuning은 훈련 및 평가 손실 모두에서 가장 낮은 값(좋은 성능)을 기록했지만, 훈련 시간과 메모리 사용량이 높습니다.
- Rank 8은 Instruction-tuning에 비해 훈련 시간에서 효율적이지만, 손실 감소가 느리고 최종 손실 값이 높아 성능이 낮을 가능성이 있습니다.
- Rank를 크게 설정하면, 좀 더 복잡한 문맥정보를 학습하겠지만, LoRA의 장점인 빠른 학습 속도 및 낮은 리소스 사용량이라는 장점을 희생해야하고,
- Rank를 작게 설정할 수록 장점은 극대화 되지만, 복잡한 문맥정보를 학습하는데 한계가 있다는 점이 있기 때문에, 잘 조정해서 활용해야합니다.


### 현재 LoRA에서 메모리 사용량이 더 높은 이유
**문제**: 모든 레이어에 LoRA가 적용되어 메모리 사용 증가
- 코드에서 `torch.nn.Linear` 타입의 모든 레이어에 LoRA가 적용되고 있습니다. 
- 대규모 모델에서는 모든 `Linear` 레이어에 LoRA를 적용할 경우 메모리 사용량이 크게 증가할 수 있습니다.
  
```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        target_modules.add(name)
```

**해걀 방법**: 특정 주요 레이어에만 LoRA 적용
- LoRA는 모든 레이어에 적용할 필요가 없으며, 주요 레이어(예: attention 관련 레이어)에만 적용하는 것이 메모리 효율을 높이는 데 효과적입니다.
- 주로 Transformer 구조에서 다음과 같은 레이어에 LoRA를 적용합니다:
  - f_attn.q_proj
  -	self_attn.k_proj
  - self_attn.v_proj
  -	self_attn.out_proj
```python
# 주요 레이어에만 LoRA 적용
target_modules = ["q_proj", "v_proj", "out_proj"]

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules
)
```

## 6. 훈련시 메모리 사용량

| 메모리 항목           | Instruction-tuning                | LoRA (r=256)                             |
|-----------------------|------------------------------------|-------------------------------------------|
| **훈련 시 메모리 사용** | 활성값 + 그래디언트 + 옵티마이저   | 동결된 파라미터 제외, 일부 그래디언트만 저장 |
| **모델 전체 메모리**    | 전체 파라미터 메모리               | 추가 행렬 메모리로 인해 증가 가능            |
| **활성값 관리 최적화**  | 기본적으로 중간값 유지             | 활성값 메모리 증가 가능                     |

- **LoRA**는 모델 파라미터가 동결되기 때문에, **훈련 중 그래디언트와 옵티마이저 메모리 사용량**이 줄어듭니다.  
  따라서 **실시간 훈련 메모리 사용량**은 **Instruction-tuning**보다 작을 수 있습니다.

- 그러나 **높은 Rank 값**(예: `r=256`)과 **모든 레이어에 LoRA 적용**으로 인해 **추가 행렬 연산**이 많아질 경우,  
  **전체 메모리 사용량**은 Instruction-tuning보다 오히려 증가할 수 있습니다.

- 훈련 중 메모리 사용을 줄이기 위해 다음과 같은 최적화 기법을 고려할 수 있습니다:
  - **Rank 값**을 `16~64`로 조정
  - **주요 레이어**에만 LoRA 적용 (예: `q_proj`, `k_proj`, `v_proj`, `out_proj`)
  - **Mixed Precision (FP16)** 활성화
  - **Gradient Checkpointing** 활성화

## 7. **최종 메모리 사용량 비교**

| 메모리 항목           | Instruction-tuning                | LoRA (r=256)                             |
|-----------------------|------------------------------------|-------------------------------------------|
| **최대 메모리 사용량** | 활성값 + 모든 파라미터 그래디언트 | 활성값 + 일부 레이어의 그래디언트          |
| **모델 파라미터 메모리** | 전체 파라미터 메모리               | 기존 파라미터 + 추가 행렬 메모리 (`A`, `B`) |

### **설명**
1. **최대 메모리 사용량**  
   - **Instruction-tuning**은 모델의 모든 파라미터에 대해 활성값과 그래디언트를 계산하고 저장하기 때문에 메모리 사용량이 높습니다.
   - **LoRA**는 일부 타겟 레이어에 대해서만 추가 연산을 수행하기 때문에 훈련 중 최대 메모리 사용량이 줄어들 수 있습니다.

2. **모델 파라미터 메모리**  
   - **Instruction-tuning**은 모델의 기본 파라미터만 메모리에 유지됩니다.
   - **LoRA**는 기존 파라미터 외에 Rank 값에 비례하여 추가 행렬(`A`, `B`)이 메모리에 유지되어 최종 메모리 사용량이 증가할 수 있습니다.

3. **결론**  
   - 훈련 중 실시간 메모리 사용량은 **LoRA**가 더 효율적일 수 있지만,  
     훈련이 끝난 후 **최종 모델 파라미터 메모리**는 **LoRA**가 더 많이 사용할 가능성이 있습니다.