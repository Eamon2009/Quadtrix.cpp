#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <csignal>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "config/config.h"
#include "include/dataloader.h"
#include "include/gpt.h"
#include "include/backward.h"

// Signal handler (Ctrl-C stops generation gracefully)
static volatile bool g_interrupted = false;
static void sig_handler(int) { g_interrupted = true; }

// Timing helpers
static std::string now_str()
{
      std::time_t t = std::time(nullptr);
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
      return buf;
}

static double wall_secs()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static bool file_exists(const std::string &path)
{
      std::ifstream f(path.c_str(), std::ios::binary);
      return f.good();
}

static std::string dir_name(const std::string &path)
{
      std::string::size_type pos = path.find_last_of("/\\");
      if (pos == std::string::npos)
            return ".";
      if (pos == 0)
            return path.substr(0, 1);
      return path.substr(0, pos);
}

static bool is_absolute_path(const std::string &path)
{
      if (path.empty())
            return false;
      if (path.size() > 1 && path[1] == ':')
            return true;
      return path[0] == '/' || path[0] == '\\';
}

static std::string join_path(const std::string &base, const std::string &child)
{
      if (base.empty() || base == ".")
            return child;
      char last = base[base.size() - 1];
      if (last == '/' || last == '\\')
            return base + child;
      return base + "/" + child;
}

static std::string choose_existing_path(const std::string &requested_path,
                                        const std::string &argv0)
{
      if (requested_path.empty())
            return requested_path;
      if (file_exists(requested_path))
            return requested_path;
      if (is_absolute_path(requested_path))
            return requested_path;

      std::vector<std::string> candidates;
      candidates.push_back(join_path(dir_name(argv0), requested_path));
      candidates.push_back(join_path(".", requested_path));

      for (size_t i = 0; i < candidates.size(); ++i)
      {
            if (file_exists(candidates[i]))
                  return candidates[i];
      }
      return requested_path;
}

static std::string choose_output_path(const std::string &requested_path,
                                      const std::string &argv0)
{
      if (requested_path.empty() || is_absolute_path(requested_path))
            return requested_path;

      std::string exe_relative = join_path(dir_name(argv0), requested_path);
      if (file_exists(requested_path) || !file_exists(exe_relative))
            return requested_path;
      return exe_relative;
}

// estimate loss — no gradients, training=false
static float estimate_loss(GPTLanguageModel &model,
                           DataLoader &dl,
                           const std::string &split,
                           std::mt19937 &rng)
{
      float total = 0.0f;
      for (int k = 0; k < EVAL_ITERS; ++k)
      {
            std::pair<std::vector<int>, std::vector<int>> batch =
                dl.get_batch(split, BATCH_SIZE, BLOCK_SIZE, rng);
            std::pair<Tensor, float> result =
                model.forward(batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, false);
            total += result.second;
      }
      return total / EVAL_ITERS;
}

// Chat window
// Encodes a user prompt string into token indices using the
// DataLoader's vocabulary, then feeds them into model.generate().
// Only touches the public encode/decode/generate interface —
// zero changes to math or training logic.
static void run_chat(GPTLanguageModel &model,
                     DataLoader &dl,
                     int max_new_tokens)
{
      std::cout << "\n"
                << std::string(60, '=') << "\n";
      std::cout << "  Quadtrix CHAT MODE\n";
      std::cout << "  Type your prompt and press Enter. "
                   "Type 'quit' or 'exit' to leave.\n";
      std::cout << std::string(60, '=') << "\n\n";

      while (!g_interrupted)
      {
            // input
            std::cout << "\033[1;32mYou>\033[0m ";
            std::cout.flush();

            std::string prompt;
            if (!std::getline(std::cin, prompt))
                  break; // EOF (piped input ended)

            // Trim leading/trailing whitespace
            size_t s = prompt.find_first_not_of(" \t\r\n");
            size_t e = prompt.find_last_not_of(" \t\r\n");
            if (s == std::string::npos)
                  continue; // blank line — ask again
            prompt = prompt.substr(s, e - s + 1);

            if (prompt == "quit" || prompt == "exit")
            {
                  std::cout << "[Chat] Bye!\n";
                  break;
            }

            // Encode prompt
            // dl.encode() maps the raw string through the same
            // char-level (or BPE) vocab built during data loading.
            std::vector<int> ctx = dl.encode(prompt);
            if (ctx.empty())
            {
                  // If the vocab doesn't cover some characters,
                  // fall back to the BOS token so generation can
                  // still start rather than crashing.
                  ctx = {0};
            }

            // Clamp context to BLOCK_SIZE (model's max sequence length)
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());

            // Stream model response token-by-token
            std::cout << "\033[1;36mQuadtrix>\033[0m ";
            std::cout.flush();

            for (int tok = 0; tok < max_new_tokens && !g_interrupted; ++tok)
            {
                  ctx = model.generate(ctx, 1);
                  std::cout << dl.decode({ctx.back()}) << std::flush;

                  // Keep context within BLOCK_SIZE window
                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n\n";
      }
}

// Main
int main(int argc, char *argv[])
{
      std::signal(SIGINT, sig_handler);

      // Banner
      std::cout << std::string(60, '=') << "\n";
      std::cout << " Quadtrix v1.0 (C++)\n";
      std::cout << std::string(60, '=') << "\n";
      std::cout << "\n[INFO] Starting at: " << now_str() << "\n";

      std::string data_path = DEFAULT_CLEANED_PATH;
      const char *env_data_path = std::getenv(DATA_PATH_ENV_VAR.c_str());
      if (env_data_path != nullptr && env_data_path[0] != '\0')
            data_path = env_data_path;

      std::string model_path = BEST_MODEL_PATH;
      const char *env_model_path = std::getenv(MODEL_PATH_ENV_VAR.c_str());
      if (env_model_path != nullptr && env_model_path[0] != '\0')
            model_path = env_model_path;

      bool gen_mode = false;
      bool chat_mode = false; // ← NEW flag
      int chat_tokens = 200;  // default tokens per reply

      for (int i = 1; i < argc; ++i)
      {
            std::string a = argv[i];
            if (a == "--generate")
                  gen_mode = true;
            else if (a == "--chat") // ← NEW
                  chat_mode = true;
            else if (a == "--chat-tokens" && i + 1 < argc) // ← NEW (optional)
                  chat_tokens = std::atoi(argv[++i]);
            else
                  data_path = a;
      }

      data_path = choose_existing_path(data_path, argv[0]);
      model_path = choose_output_path(model_path, argv[0]);

      // Config print
      std::cout << "\n[CONFIG] Hyperparameters:\n";
      std::cout << "         batch_size=" << BATCH_SIZE
                << "  block_size=" << BLOCK_SIZE << "\n";
      std::cout << "         max_iters=" << MAX_ITERS
                << "  learning_rate=" << LEARNING_RATE << "\n";
      std::cout << "         n_embd=" << N_EMBD
                << "  n_head=" << N_HEAD
                << "  n_layer=" << N_LAYER
                << "  dropout=" << DROPOUT << "\n";

      //  Data
      DataLoader dl;
      try
      {
            dl.load(data_path);
      }
      catch (const std::exception &e)
      {
            std::cerr << e.what() << "\n";
            std::cerr << "[HINT]  Put your text at " << DEFAULT_CLEANED_PATH
                      << ", pass a file path as the first argument, or set "
                      << DATA_PATH_ENV_VAR << ".\n";
            return 1;
      }
      GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);

      long n_params = model.num_params();
      std::cout << "[MODEL] Parameters  : "
                << std::fixed << std::setprecision(2)
                << n_params / 1.0e6f << " M  (" << n_params << " total)\n";
      std::cout << "[MODEL] Architecture: "
                << N_LAYER << " layers x "
                << N_HEAD << " heads x "
                << N_EMBD << " embedding dim\n";

      // chat mode
      if (chat_mode) // NEW block
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot start chat because model weights were not found at "
                            << model_path << "\n";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.\n";
                  return 1;
            }

            model.load(model_path);
            std::cout << "[CHAT]  Weights loaded from " << model_path << "\n";
            std::cout << "[CHAT]  Max tokens per reply: " << chat_tokens
                      << "  (override with --chat-tokens N)\n";

            run_chat(model, dl, chat_tokens);
            return 0;
      }

      // Generate-only mode
      if (gen_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot generate because model weights were not found at "
                            << model_path << "\n";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.\n";
                  return 1;
            }

            model.load(model_path);
            std::cout << "\n"
                      << std::string(60, '-') << "\n";
            std::cout << "  Quadtrix OUTPUT  (Ctrl+C to stop)\n";
            std::cout << std::string(60, '-') << "\n\n";
            std::vector<int> ctx = {0};
            while (!g_interrupted)
            {
                  ctx = model.generate(ctx, 1);
                  std::cout << dl.decode({ctx.back()}) << std::flush;
                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n\n[Stopped by user]\n";
            return 0;
      }

      // optimizer
      AdamWState opt = build_optimizer(model, LEARNING_RATE);

      // Separate RNG for batch sampling
      std::mt19937 rng(SEED);

      // training loop
      std::cout << "\n"
                << std::string(60, '-') << "\n";
      std::cout << "  TRAINING  ("
                << MAX_ITERS << " iters, eval every "
                << EVAL_INTERVAL << ")\n";
      std::cout << std::string(60, '-') << "\n";

      float best_val_loss = 1e30f;
      double train_start = wall_secs();

      for (int iter = 0; iter <= MAX_ITERS && !g_interrupted; ++iter)
      {

            // Periodic eval checkpoint
            if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
            {
                  if (iter == 0)
                  {
                        std::cout << "[INFO] Running initial loss estimate (" << EVAL_ITERS
                                  << " train batches + " << EVAL_ITERS
                                  << " val batches). This can take a while on CPU...\n";
                  }
                  else
                  {
                        std::cout << "[INFO] Evaluating checkpoint at iter " << iter
                                  << "/" << MAX_ITERS << "...\n";
                  }
                  std::cout.flush();

                  float tl = estimate_loss(model, dl, "train", rng);
                  float vl = estimate_loss(model, dl, "val", rng);

                  double elapsed = wall_secs() - train_start;
                  double eta = (iter > 0) ? elapsed / iter * (MAX_ITERS - iter) : 0.0;
                  float pct = 100.0f * iter / MAX_ITERS;
                  bool better = vl < best_val_loss;

                  if (better)
                  {
                        best_val_loss = vl;
                        model.save(model_path);
                  }

                  std::cout << "[" << std::setw(5) << iter << "/" << MAX_ITERS << "] "
                            << std::fixed << std::setprecision(1) << pct << "%  "
                            << "train=" << std::setprecision(4) << tl
                            << "  val=" << vl
                            << "  elapsed=" << std::setprecision(0) << elapsed << "s"
                            << "  ETA=" << eta << "s"
                            << (better ? "  << best!" : "")
                            << "\n";
                  std::cout.flush();

                  if (iter == MAX_ITERS)
                        break;
            }

            // Sample training batch
            std::pair<std::vector<int>, std::vector<int>> batch =
                dl.get_batch("train", BATCH_SIZE, BLOCK_SIZE, rng);

            // Forward — saves all intermediate activations
            SavedForward saved = forward_save(model,
                                              batch.first, BATCH_SIZE, BLOCK_SIZE,
                                              batch.second, /*training=*/true);

            //  Backward — exact analytical gradients
            Grads grads = backward(model, saved);

            // AdamW parameter update
            apply_grads(model, grads, opt);
      }

      double total = wall_secs() - train_start;
      std::cout << "\n[DONE]  Training finished in "
                << std::fixed << std::setprecision(1) << total << "s ("
                << total / 60.0 << " min)  |  Best val loss: "
                << std::setprecision(4) << best_val_loss << "\n";
      std::cout << "[SAVE]  Best weights saved to " << model_path << "\n";

      //  Continuous generation (mirrors Python's while True loop)
      std::cout << "\n"
                << std::string(60, '-') << "\n";
      std::cout << "  MODEL OUTPUT  (Ctrl+C to stop)\n";
      std::cout << std::string(60, '-') << "\n\n";

      model.load(model_path);
      model.rng = std::mt19937(SEED + 42);

      std::vector<int> ctx = {0};
      while (!g_interrupted)
      {
            ctx = model.generate(ctx, 1);
            std::cout << dl.decode({ctx.back()}) << std::flush;
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
      }

      std::cout << "\n\n[Stopped by user]\n";
      std::cout << "[TOTAL] Wall-clock: "
                << std::fixed << std::setprecision(1)
                << (wall_secs() - train_start) << "s\n";
      return 0;
}