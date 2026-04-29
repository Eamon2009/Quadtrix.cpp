#pragma once
// ============================================================
//  include/tensor.h  –  Lightweight 2-D / 3-D float tensor
//  (CPU only – mirrors what PyTorch tensors do in the model)
// ============================================================

#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <functional>

// ------------------------------------------------------------------
// Tensor  (row-major, float32)
//   shape is stored as {d0, d1}  or  {d0, d1, d2}
// ------------------------------------------------------------------
struct Tensor
{
      std::vector<int> shape;
      std::vector<float> data;

      Tensor() = default;

      Tensor(std::vector<int> sh, float fill = 0.0f)
          : shape(sh)
      {
            int total = 1;
            for (int d : sh)
                  total *= d;
            data.assign(total, fill);
      }

      int numel() const
      {
            int n = 1;
            for (int d : shape)
                  n *= d;
            return n;
      }

      int ndim() const { return (int)shape.size(); }

      // ---- element access helpers --------------------------------
      float &at(int i)
      {
            assert(i >= 0 && i < (int)data.size());
            return data[i];
      }
      float at(int i) const
      {
            assert(i >= 0 && i < (int)data.size());
            return data[i];
      }

      // 2-D
      float &at(int r, int c)
      {
            return data[r * shape[1] + c];
      }
      float at(int r, int c) const
      {
            return data[r * shape[1] + c];
      }

      // 3-D
      float &at(int b, int r, int c)
      {
            return data[b * shape[1] * shape[2] + r * shape[2] + c];
      }
      float at(int b, int r, int c) const
      {
            return data[b * shape[1] * shape[2] + r * shape[2] + c];
      }

      // ---- factory helpers ---------------------------------------
      static Tensor zeros(std::vector<int> sh) { return Tensor(sh, 0.0f); }
      static Tensor ones(std::vector<int> sh) { return Tensor(sh, 1.0f); }

      static Tensor randn(std::vector<int> sh, float mean, float std,
                          std::mt19937 &rng)
      {
            std::normal_distribution<float> dist(mean, std);
            Tensor t(sh);
            for (auto &v : t.data)
                  v = dist(rng);
            return t;
      }

      void fill(float v) { std::fill(data.begin(), data.end(), v); }

      // ---- print shape -------------------------------------------
      void print_shape(const std::string &name = "") const
      {
            if (!name.empty())
                  std::cout << name << ": ";
            std::cout << "[";
            for (int i = 0; i < (int)shape.size(); ++i)
            {
                  std::cout << shape[i];
                  if (i + 1 < (int)shape.size())
                        std::cout << ", ";
            }
            std::cout << "]" << std::endl;
      }
};

// ------------------------------------------------------------------
// Basic math ops  (in-place and returning new tensors)
// ------------------------------------------------------------------

// element-wise add (same shape)
inline Tensor add(const Tensor &a, const Tensor &b)
{
      assert(a.data.size() == b.data.size());
      Tensor c(a.shape);
      for (int i = 0; i < (int)a.data.size(); ++i)
            c.data[i] = a.data[i] + b.data[i];
      return c;
}

// scalar multiply
inline Tensor scale(const Tensor &a, float s)
{
      Tensor c(a.shape);
      for (int i = 0; i < (int)a.data.size(); ++i)
            c.data[i] = a.data[i] * s;
      return c;
}

// ReLU
inline Tensor relu(const Tensor &a)
{
      Tensor c(a.shape);
      for (int i = 0; i < (int)a.data.size(); ++i)
            c.data[i] = std::max(0.0f, a.data[i]);
      return c;
}

// Softmax along last dim for 3-D tensor [B, T, C]
inline Tensor softmax3d(const Tensor &a)
{
      int B = a.shape[0], T = a.shape[1], C = a.shape[2];
      Tensor out(a.shape);
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float maxv = -1e30f;
                  for (int c = 0; c < C; ++c)
                        maxv = std::max(maxv, a.at(b, t, c));
                  float sumv = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float e = std::exp(a.at(b, t, c) - maxv);
                        out.at(b, t, c) = e;
                        sumv += e;
                  }
                  for (int c = 0; c < C; ++c)
                        out.at(b, t, c) /= sumv;
            }
      }
      return out;
}

// Softmax along last dim for 2-D tensor [T, C]
inline Tensor softmax2d(const Tensor &a)
{
      int T = a.shape[0], C = a.shape[1];
      Tensor out(a.shape);
      for (int t = 0; t < T; ++t)
      {
            float maxv = -1e30f;
            for (int c = 0; c < C; ++c)
                  maxv = std::max(maxv, a.at(t, c));
            float sumv = 0.0f;
            for (int c = 0; c < C; ++c)
            {
                  float e = std::exp(a.at(t, c) - maxv);
                  out.at(t, c) = e;
                  sumv += e;
            }
            for (int c = 0; c < C; ++c)
                  out.at(t, c) /= sumv;
      }
      return out;
}

// Layer-norm along last dim  [B, T, C]  → same shape
inline Tensor layer_norm(const Tensor &x,
                         const Tensor &gamma, // [C]
                         const Tensor &beta,  // [C]
                         float eps = 1e-5f)
{
      int B = x.shape[0], T = x.shape[1], C = x.shape[2];
      Tensor out(x.shape);
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float mu = 0.0f;
                  for (int c = 0; c < C; ++c)
                        mu += x.at(b, t, c);
                  mu /= C;
                  float var = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float d = x.at(b, t, c) - mu;
                        var += d * d;
                  }
                  var /= C;
                  float inv = 1.0f / std::sqrt(var + eps);
                  for (int c = 0; c < C; ++c)
                        out.at(b, t, c) = (x.at(b, t, c) - mu) * inv * gamma.at(c) + beta.at(c);
            }
      }
      return out;
}

// matmul:  [B, T, D] x [D, E]  →  [B, T, E]
inline Tensor matmul(const Tensor &a, const Tensor &w)
{
      // a: [B, T, D]  or  [B, T, D]
      // w: [D, E]
      assert(a.ndim() == 3 && w.ndim() == 2);
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      int E = w.shape[1];
      assert(w.shape[0] == D);
      Tensor out({B, T, E}, 0.0f);
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int e = 0; e < E; ++e)
                  {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d)
                              s += a.at(b, t, d) * w.at(d, e);
                        out.at(b, t, e) = s;
                  }
      return out;
}

// add bias [E] broadcast over [B, T, E]
inline Tensor add_bias(const Tensor &x, const Tensor &bias)
{
      assert(x.shape.back() == bias.shape[0]);
      Tensor out = x;
      int E = bias.shape[0];
      int stride = E;
      int n = x.numel() / E;
      for (int i = 0; i < n; ++i)
            for (int e = 0; e < E; ++e)
                  out.data[i * stride + e] += bias.data[e];
      return out;
}

// batched matmul:  [B, T, D] x [B, D, T2]  →  [B, T, T2]
inline Tensor bmm(const Tensor &a, const Tensor &b)
{
      assert(a.ndim() == 3 && b.ndim() == 3);
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      int T2 = b.shape[2];
      assert(b.shape[0] == B && b.shape[1] == D);
      Tensor out({B, T, T2}, 0.0f);
      for (int bb = 0; bb < B; ++bb)
            for (int t = 0; t < T; ++t)
                  for (int t2 = 0; t2 < T2; ++t2)
                  {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d)
                              s += a.at(bb, t, d) * b.at(bb, d, t2);
                        out.at(bb, t, t2) = s;
                  }
      return out;
}

// transpose last two dims of 3-D tensor [B, T, D] → [B, D, T]
inline Tensor transpose23(const Tensor &a)
{
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      Tensor out({B, D, T});
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                        out.at(b, d, t) = a.at(b, t, d);
      return out;
}

// concat along last dim:  [B,T,D1] + [B,T,D2] → [B,T,D1+D2]
inline Tensor cat_last(const std::vector<Tensor> &ts)
{
      int B = ts[0].shape[0], T = ts[0].shape[1];
      int total = 0;
      for (auto &t : ts)
            total += t.shape[2];
      Tensor out({B, T, total}, 0.0f);
      int offset = 0;
      for (auto &t : ts)
      {
            int D = t.shape[2];
            for (int b = 0; b < B; ++b)
                  for (int tt = 0; tt < T; ++tt)
                        for (int d = 0; d < D; ++d)
                              out.at(b, tt, offset + d) = t.at(b, tt, d);
            offset += D;
      }
      return out;
}

// dropout mask (applied only during training)
inline Tensor dropout(const Tensor &x, float p, bool training, std::mt19937 &rng)
{
      if (!training || p == 0.0f)
            return x;
      std::bernoulli_distribution dist(1.0f - p);
      Tensor out = x;
      float scale_v = 1.0f / (1.0f - p);
      for (auto &v : out.data)
            v = dist(rng) ? v * scale_v : 0.0f;
      return out;
}