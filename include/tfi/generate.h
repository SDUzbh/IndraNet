// Text generation: continuing a sequence one token at a time, with a KV cache (Model::forward_cached)
// so that each step runs only the new token, and hooks that apply at every step.
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "tfi/model.h"

namespace tfi {

struct GenerateOptions {
    int64_t max_new_tokens = 32;
    double temperature = 0;  // 0: greedy, the most likely token (the lowest id among equal ones)
    int64_t top_k = 0;       // sample from the k most likely tokens only; 0: all
    double top_p = 1;        // ... from the fewest most likely tokens whose probabilities reach top_p
    uint64_t seed = 0;       // the same seed, model, prompt and options give the same tokens
    std::vector<int64_t> stop_tokens;  // stop after generating one of these (it is returned)
    bool stop_at_eos = true;           // ... or one of the model's eos_tokens()
    // Called with each new token as it is chosen; returning false stops after it.
    std::function<bool(int64_t token)> on_token;
};

// Picks the next token from a row of logits by GenerateOptions' rules: temperature, then top_k, then
// top_p. Logits that are NaN or -inf are never picked.
class Sampler {
  public:
    explicit Sampler(const GenerateOptions& options);
    int64_t operator()(const float* logits, int64_t n);

  private:
    double temperature_, top_p_;
    int64_t top_k_;
    std::mt19937_64 rng_;
    std::vector<std::pair<double, int64_t>> candidates_;
};

// New tokens continuing `prompt` (not including it), at most max_new_tokens. The prompt runs in one
// pass, then each new token alone. `hooks` apply to every pass: the prompt's activations first, then
// each new token's (see Model::forward_cached for their shapes), so an intervention lasts through the
// whole text.
Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options = {},
                const Hooks* hooks = nullptr);

}  // namespace tfi
