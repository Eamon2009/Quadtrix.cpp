## Quick Start
```
# Compile the main.cpp file
 g++ -std=c++17 -O2 -I. -Iinclude -o Quadtrix main.cpp
# Train
./Quadtrix data/input.txt
# Output
./Quadtrix data/input.txt --generate
```
# Generate only (uses saved best_model.bin)
```
./Quadtrix data/input.txt --generate
```
Loss
 └─ backward_cross_entropy   → dLogits [B*T, V]   (softmax - one_hot) / BT
     └─ backward_linear       → dLM_in + dW_lmhead, db_lmhead
         └─ backward_layernorm→ dx  (Ba et al. exact formula, uses saved μ, σ⁻¹)
             └─ for each Block (reverse order):
                 ├─ FFN residual pass-through
                 ├─ backward_layernorm (LN2) → dx_ln2
                 ├─ backward_linear (fc2)    → dh_relu + dW2, db2
                 ├─ backward_relu            → dh_pre  (gates on pre-activation)
                 ├─ backward_linear (fc1)    → dx_ffn + dW1, db1
                 ├─ MHA residual pass-through
                 ├─ backward_dropout (proj)
                 ├─ backward_linear (proj)   → dConcat + dW_proj, db_proj
                 ├─ backward_cat_last        → slice per head
                 └─ for each Head:
                     ├─ backward_bmm (wei@v) → d_wei_drop, dv
                     ├─ backward_dropout (attn weights)
                     ├─ causal mask backward  (zero upper triangle)
                     ├─ backward_softmax3d   → d_wei_pre
                     ├─ scale backward       → × 1/√hs
                     ├─ backward_bmm (q@kᵀ) → dq, dk
                     └─ backward_linear ×3   → dx_k, dx_q, dx_v + dWk, dWq, dWv
 └─ tok_emb grad: scatter-add into vocab rows
 └─ pos_emb grad: sum over batch dimension
 ./Quadtrixchat data/input.txt --chat
 # wsl 
  ./Quadtrixchat.exe data/input.txt --chat