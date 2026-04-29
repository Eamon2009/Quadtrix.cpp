#pragma once

#include "tensor.h"
#include <cstdint>
#include <cstring>
#include <torch/torch.h>
#include <stdexcept>

inline torch::Tensor to_torch_tensor(const Tensor &src)
{
      if (src.shape.empty() || src.shape.size() > 3)
            throw std::invalid_argument("Tensor rank must be 1, 2, or 3");

      std::vector<int64_t> sizes(src.shape.begin(), src.shape.end());
      auto tensor = torch::from_blob(
                        const_cast<float *>(src.data.data()),
                        torch::IntArrayRef(sizes),
                        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone();
      return tensor;
}

inline Tensor from_torch_tensor(const torch::Tensor &src)
{
      auto cpu = src.contiguous().to(torch::kCPU).to(torch::kFloat32);
      std::vector<int> shape;
      shape.reserve(cpu.dim());
      for (int64_t i = 0; i < cpu.dim(); ++i)
            shape.push_back(static_cast<int>(cpu.size(i)));

      Tensor out(shape);
      std::memcpy(out.data.data(), cpu.data_ptr<float>(), out.numel() * sizeof(float));
      return out;
}
