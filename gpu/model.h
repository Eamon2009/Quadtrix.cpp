#pragma once
// ============================================================
//  include/model.h  —  GPT Language Model (LibTorch)
//
//  Mirrors the Python class exactly:
//    Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel
//
//  Everything is a proper torch::nn::Module so that:
//    - model->to(device)  moves all weights to GPU/CPU in one call
//    - loss.backward()    computes all gradients automatically
//    - torch::save/load   checkpoints the full model
// ============================================================

#include <torch/torch.h>
#include "config/config.h"
#include <iostream>

//  Head  —  single causal self-attention head

struct HeadImpl : torch::nn::Module
{
      int64_t head_size;

      torch::nn::Linear key{nullptr}, query{nullptr}, value{nullptr};
      torch::Tensor tril; // registered buffer (not a parameter)
      torch::nn::Dropout dropout;

      HeadImpl(int64_t n_embd, int64_t hs)
          : head_size(hs),
            key(register_module("key", torch::nn::Linear(
                                           torch::nn::LinearOptions(n_embd, hs).bias(false)))),
            query(register_module("query", torch::nn::Linear(
                                               torch::nn::LinearOptions(n_embd, hs).bias(false)))),
            value(register_module("value", torch::nn::Linear(
                                               torch::nn::LinearOptions(n_embd, hs).bias(false)))),
            dropout(register_module("dropout",
                                    torch::nn::Dropout(DROPOUT)))
      {
            // causal mask — register as buffer so it moves with .to(device)
            tril = register_buffer("tril",
                                   torch::tril(torch::ones({BLOCK_SIZE, BLOCK_SIZE})));
      }

      torch::Tensor forward(torch::Tensor x)
      {
            auto sizes = x.sizes();
            int64_t T = sizes[1];

            auto k = key->forward(x);   // [B, T, hs]
            auto q = query->forward(x); // [B, T, hs]

            // scaled dot-product  q @ k^T * scale
            auto wei = torch::matmul(q, k.transpose(-2, -1)) * std::pow((double)head_size, -0.5); // [B, T, T]

            // causal mask: fill future positions with -inf
            wei = wei.masked_fill(
                tril.slice(0, 0, T).slice(1, 0, T) == 0,
                -std::numeric_limits<float>::infinity());

            wei = torch::softmax(wei, /*dim=*/-1);
            wei = dropout->forward(wei);

            auto v = value->forward(x);   // [B, T, hs]
            return torch::matmul(wei, v); // [B, T, hs]
      }
};
TORCH_MODULE(Head); // creates shared_ptr wrapper  to "Head"

//  MultiHeadAttention
struct MultiHeadAttentionImpl : torch::nn::Module
{
      torch::nn::ModuleList heads{nullptr};
      torch::nn::Linear proj{nullptr};
      torch::nn::Dropout dropout{nullptr};

      MultiHeadAttentionImpl(int64_t n_embd, int64_t n_head)
      {
            int64_t hs = n_embd / n_head;
            heads = register_module("heads", torch::nn::ModuleList());
            for (int64_t i = 0; i < n_head; ++i)
                  heads->push_back(Head(n_embd, hs));

            proj = register_module("proj",
                                   torch::nn::Linear(n_head * hs, n_embd));
            dropout = register_module("dropout",
                                      torch::nn::Dropout(DROPOUT));
      }

      torch::Tensor forward(torch::Tensor x)
      {
            std::vector<torch::Tensor> outs;
            outs.reserve(heads->size());
            for (auto &h : *heads)
                  outs.push_back(h->as<HeadImpl>()->forward(x));

            auto out = torch::cat(outs, /*dim=*/-1); // [B, T, n_embd]
            out = proj->forward(out);
            return dropout->forward(out);
      }
};
TORCH_MODULE(MultiHeadAttention);

//  FeedForward  —  Linear to ReLU to Linear to Dropout

struct FeedForwardImpl : torch::nn::Module
{
      torch::nn::Sequential net{nullptr};

      explicit FeedForwardImpl(int64_t n_embd)
      {
            net = register_module("net", torch::nn::Sequential(
                                             torch::nn::Linear(n_embd, 4 * n_embd),
                                             torch::nn::ReLU(),
                                             torch::nn::Linear(4 * n_embd, n_embd),
                                             torch::nn::Dropout(DROPOUT)));
      }

      torch::Tensor forward(torch::Tensor x)
      {
            return net->forward(x);
      }
};
TORCH_MODULE(FeedForward);

//  Block  —  pre-LN transformer block
struct BlockImpl : torch::nn::Module
{
      MultiHeadAttention sa{nullptr};
      FeedForward ffwd{nullptr};
      torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};

      BlockImpl(int64_t n_embd, int64_t n_head)
      {
            sa = register_module("sa", MultiHeadAttention(n_embd, n_head));
            ffwd = register_module("ffwd", FeedForward(n_embd));
            ln1 = register_module("ln1",
                                  torch::nn::LayerNorm(
                                      torch::nn::LayerNormOptions({n_embd})));
            ln2 = register_module("ln2",
                                  torch::nn::LayerNorm(
                                      torch::nn::LayerNormOptions({n_embd})));
      }

      torch::Tensor forward(torch::Tensor x)
      {
            x = x + sa->forward(ln1->forward(x));   // residual + MHA
            x = x + ffwd->forward(ln2->forward(x)); // residual + FFN
            return x;
      }
};
TORCH_MODULE(Block);

// ---------------------------------------------
//  GPTLanguageModel
struct GPTLanguageModelImpl : torch::nn::Module
{
      int64_t vocab_size_, block_size_;

      torch::nn::Embedding token_embedding_table{nullptr};
      torch::nn::Embedding position_embedding_table{nullptr};
      torch::nn::Sequential blocks{nullptr};
      torch::nn::LayerNorm ln_f{nullptr};
      torch::nn::Linear lm_head{nullptr};

      GPTLanguageModelImpl(int64_t vocab_size, int64_t block_size,
                           int64_t n_embd, int64_t n_head, int64_t n_layer)
          : vocab_size_(vocab_size), block_size_(block_size)
      {

            token_embedding_table = register_module("token_embedding_table",
                                                    torch::nn::Embedding(vocab_size, n_embd));

            position_embedding_table = register_module("position_embedding_table",
                                                       torch::nn::Embedding(block_size, n_embd));

            blocks = register_module("blocks", torch::nn::Sequential());
            for (int64_t i = 0; i < n_layer; ++i)
                  blocks->push_back(Block(n_embd, n_head));

            ln_f = register_module("ln_f",
                                   torch::nn::LayerNorm(
                                       torch::nn::LayerNormOptions({n_embd})));

            lm_head = register_module("lm_head",
                                      torch::nn::Linear(n_embd, vocab_size));

            // weight initialisation (matches Python _init_weights)
            _init_weights();
      }

      // Weight init
      void _init_weights()
      {
            for (auto &module : modules(/*include_self=*/false))
            {
                  if (auto *lin = module->as<torch::nn::LinearImpl>())
                  {
                        torch::nn::init::normal_(lin->weight, 0.0, 0.02);
                        if (lin->bias.defined())
                              torch::nn::init::zeros_(lin->bias);
                  }
                  else if (auto *emb = module->as<torch::nn::EmbeddingImpl>())
                  {
                        torch::nn::init::normal_(emb->weight, 0.0, 0.02);
                  }
            }
      }

      // Forward
      //   idx     : [B, T]   token indices
      //   targets : [B, T]   next-token targets  (optional)
      //   returns : (logits [B, T, V],  loss scalar)
      //             loss is a zero-dim tensor (undefined if no targets)
      std::pair<torch::Tensor, torch::Tensor>
      forward(torch::Tensor idx,
              torch::Tensor targets = torch::Tensor())
      {

            auto device = idx.device();
            int64_t T = idx.size(1);

            // token + position embeddings
            auto tok_emb = token_embedding_table->forward(idx);    // [B, T, C]
            auto pos = torch::arange(T, device).unsqueeze(0);      // [1, T]
            auto pos_emb = position_embedding_table->forward(pos); // [1, T, C]
            auto x = tok_emb + pos_emb;                            // [B, T, C]

            // transformer blocks
            x = blocks->forward(x);

            // final layer norm
            x = ln_f->forward(x);

            // language-model head
            auto logits = lm_head->forward(x); // [B, T, V]

            torch::Tensor loss;
            if (targets.defined())
            {
                  int64_t B = logits.size(0);
                  int64_t V = logits.size(2);
                  // cross_entropy expects [N, C] logits and [N] targets
                  auto logits_2d = logits.view({B * T, V});
                  auto targets_1d = targets.view({B * T});
                  loss = torch::nn::functional::cross_entropy(logits_2d, targets_1d);
            }

            return {logits, loss};
      }

      // Autoregressive generation
      //   context : [1, T0]  starting tokens
      //   returns : [1, T0 + max_new_tokens]
      torch::Tensor generate(torch::Tensor context, int64_t max_new_tokens)
      {
            torch::NoGradGuard no_grad;
            for (int64_t s = 0; s < max_new_tokens; ++s)
            {
                  // crop context to block_size
                  auto idx_cond = context.size(1) > block_size_
                                      ? context.slice(1, context.size(1) - block_size_)
                                      : context;

                  auto [logits, _loss] = forward(idx_cond);

                  // last time-step logits → probabilities
                  auto logits_last = logits.select(1, logits.size(1) - 1); // [1, V]
                  auto probs = torch::softmax(logits_last, /*dim=*/-1);

                  // multinomial sample
                  auto next_tok = torch::multinomial(probs, /*num_samples=*/1); // [1,1]
                  context = torch::cat({context, next_tok}, /*dim=*/1);
            }
            return context;
      }

      //  Parameter count
      int64_t num_params() const
      {
            int64_t n = 0;
            for (auto &p : parameters())
                  n += p.numel();
            return n;
      }
};
TORCH_MODULE(GPTLanguageModel);