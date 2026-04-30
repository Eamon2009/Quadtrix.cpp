#pragma once
// ============================================================
//  include/dataloader.h  —  Character tokeniser + batch sampler
//  Returns torch::Tensor batches, placed on the given device.
// ============================================================

#include <torch/torch.h>
#include "config/config.h"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

struct DataLoader
{
      std::string text;
      std::vector<char> chars;
      std::map<char, int> stoi;
      std::map<int, char> itos;
      int vocab_size{0};

      std::vector<int64_t> train_data;
      std::vector<int64_t> val_data;

      // Load text, build vocab, split
      void load(const std::string &path,
                double train_split = TRAIN_SPLIT)
      {
            std::ifstream f(path);
            if (!f.is_open())
                  throw std::runtime_error("[DataLoader] Cannot open: " + path);

            std::ostringstream ss;
            ss << f.rdbuf();
            text = ss.str();
            if (text.empty())
                  throw std::runtime_error("[DataLoader] File is empty: " + path);

            // vocabulary
            std::set<char> cs(text.begin(), text.end());
            chars = {cs.begin(), cs.end()};
            std::sort(chars.begin(), chars.end());
            vocab_size = (int)chars.size();

            for (int i = 0; i < vocab_size; ++i)
            {
                  stoi[chars[i]] = i;
                  itos[i] = chars[i];
            }

            // encode
            std::vector<int64_t> data;
            data.reserve(text.size());
            for (char c : text)
                  data.push_back(stoi.at(c));

            int64_t n = (int64_t)(train_split * (double)data.size());
            train_data = {data.begin(), data.begin() + n};
            val_data = {data.begin() + n, data.end()};

            std::cout << "[DATA]  Total chars  : " << text.size() << "\n"
                      << "[DATA]  Vocab size   : " << vocab_size << "\n"
                      << "[DATA]  Train tokens : " << train_data.size() << "\n"
                      << "[DATA]  Val   tokens : " << val_data.size() << "\n";
      }

      // encode / decode
      std::vector<int64_t> encode(const std::string &s) const
      {
            std::vector<int64_t> out;
            for (char c : s)
            {
                  auto it = stoi.find(c);
                  if (it != stoi.end())
                        out.push_back(it->second);
            }
            return out;
      }

      std::string decode(const std::vector<int64_t> &ids) const
      {
            std::string out;
            for (int64_t id : ids)
            {
                  auto it = itos.find((int)id);
                  if (it != itos.end())
                        out += it->second;
            }
            return out;
      }

      // get_batch  - (x, y) both [B, T] on `device`
      std::pair<torch::Tensor, torch::Tensor>
      get_batch(const std::string &split,
                int64_t batch_size,
                int64_t block_size,
                torch::Device device,
                std::mt19937 &rng) const
      {

            const auto &d = (split == "train") ? train_data : val_data;
            std::uniform_int_distribution<int64_t> dist(
                0, (int64_t)d.size() - block_size - 1);

            // build [B, T] on CPU first, then move to device
            auto x_cpu = torch::zeros({batch_size, block_size},
                                      torch::TensorOptions()
                                          .dtype(torch::kInt64));
            auto y_cpu = torch::zeros_like(x_cpu);

            auto x_acc = x_cpu.accessor<int64_t, 2>();
            auto y_acc = y_cpu.accessor<int64_t, 2>();

            for (int64_t b = 0; b < batch_size; ++b)
            {
                  int64_t start = dist(rng);
                  for (int64_t t = 0; t < block_size; ++t)
                  {
                        x_acc[b][t] = d[start + t];
                        y_acc[b][t] = d[start + t + 1];
                  }
            }

            return {x_cpu.to(device), y_cpu.to(device)};
      }
};