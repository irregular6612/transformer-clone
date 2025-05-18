# Transformer 구현 프로젝트

이 프로젝트는 "Attention Is All You Need" 논문에서 제안된 Transformer 아키텍처를 PyTorch로 구현한 것입니다. Self-Attention과 Multi-Head Attention 메커니즘을 단계별로 구현하고 성능을 측정합니다.

## 프로젝트 구조

### 1. 핵심 구현
- `main.ipynb`: Transformer 구현 및 실험 노트북
  - Self-Attention 구현
  - Multi-Head Attention 구현
  - 성능 측정 및 최적화

## 필요 조건

- Python 3.8 이상
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
```

2. 필요한 패키지를 설치합니다:
```bash
pip install torch numpy matplotlib jupyter
```

## 사용 방법

### Jupyter Notebook 실행
```bash
jupyter notebook main.ipynb
```

## 주요 기능

1. **Self-Attention 구현**
   - Query, Key, Value 행렬 계산
   - Attention Score 계산 및 정규화
   - Dropout 적용

2. **Multi-Head Attention**
   - 여러 개의 Attention Head 구현
   - Head 병합 및 출력 변환
   - 마스킹 지원

3. **성능 최적화**
   - 행렬 연산 통합
   - 메모리 접근 최적화
   - 실행 시간 측정

## 구현 세부사항

### Self-Attention
```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        # Attention Score 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # 마스킹 적용
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax 및 Dropout
        attention = self.dropout(F.softmax(scores, dim=-1))
        
        # 출력 계산
        output = torch.matmul(attention, v)
        return output
```

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear 변환 및 Head 분할
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Self-Attention 계산
        output = self.attention(q, k, v, mask)
        
        # Head 병합 및 출력 변환
        output = output.view(batch_size, -1, self.d_model)
        return self.w_o(output)
```

## 성능 측정

`timing_decorator` 함수를 사용하여 각 구현의 실행 시간을 측정하고 비교합니다:

```python
@timing_decorator
def optimized_implementation(input_tensor):
    # 최적화된 구현
    pass

@timing_decorator
def original_implementation(input_tensor):
    # 원본 구현
    pass
```

## 참고 자료

- [Attention Is All You Need 논문](https://arxiv.org/abs/1706.03762)
- [PyTorch 문서](https://pytorch.org/docs/stable/index.html)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) 