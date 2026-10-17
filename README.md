
# 计算图
## 实践1计算图
```markdown
## 实践1计算图示例（简单运算）
```mermaid
graph LR
    X[x] --> A["*  a=x*y"]
    Y[y] --> A
    X --> B["pow  b=x²"]
    A --> C["+  c=a+b"]
    B --> C
    C --> D["pow  f=c²"]

## 实践2计算图
flowchart LR
    X["X<br/>(1,2)"] --> L1["Linear: W1,b1<br/>Z₁ = XW₁+b₁<br/>(1,3)"]
    L1 --> ReLU["ReLU<br/>A₁ = max(0,Z₁)<br/>(1,3)"]
    ReLU --> L2["Linear: W2,b2<br/>Z₂ = A₁W₂+b₂<br/>(1,1)"]
    L2 --> Sig["Sigmoid<br/>ŷ=σ(Z₂)<br/>(1,1)"]
    Sig --> Loss["BCE Loss<br/>标量"]

    Loss -- dL/dZ₂ --> L2
    L2 -- dL/dW₂, dL/db₂ --> W2b2[W₂,b₂参数]
    L2 -- dL/dA₁ --> ReLU
    ReLU -- dL/dZ₁ --> L1
    L1 -- dL/dW₁, dL/db₁ --> W1b1[W₁,b₁参数]
