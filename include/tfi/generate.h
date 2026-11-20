// Text generation: continuing a sequence one token at a time, with a KV cache (Model::forward_cached)
// so that each step runs only the new token, and hooks that apply at every step.
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "tfi/analysis.h"
#include "tfi/model.h"

namespace tfi {

// What generate() records about each new token, from the pass whose logits chose it (the prompt's last
// position for the first token, then each new token's position): taken with capture hooks, so a pass
// on the GPU doesn't stop for them.
struct TraceOptions {
    int64_t top_k = 5;          // the most likely tokens at each step, with their probabilities
    bool lens = true;           // logit lens at the step's position (every stage, one batched unembedding)
    bool dla = true;            // direct logit attribution of the chosen token's logit (not post-norm models)
    int64_t attention_top = 0;  // per head, the positions the step attended to most (0: none). Needs the
                                // attention pattern, so cached steps run attention on the CPU
};

struct TraceStep {
    int64_t token = -1;     // the chosen token
    int64_t position = -1;  // the position whose logits chose it (the token goes at position + 1)
    float prob = 0, logprob = 0;  // its probability under the model's distribution (before temperature, top_k, top_p)
    std::vector<int64_t> top_tokens;
    std::vector<float> top_probs;
    std::vector<LensEntry> lens;  // target_prob / target_rank: the chosen token's, at each stage
    DLAResult dla;                // parts of the chosen token's logit
    struct Attention {
        std::string head;              // "L3H5"
        std::vector<int64_t> positions;  // most attended first
        std::vector<float> weights;
    };
    std::vector<Attention> attention;  // every head, with TraceOptions::attention_top
};

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
    TraceOptions trace;  // what generate() records when given a trace to fill
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
// With `trace`, it also receives a TraceStep per new token (options.trace says what each holds), before
// on_token is called for that token. Activations are recorded after `hooks` ran: the trace shows what
// the model computed, interventions included.
Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options = {},
                const Hooks* hooks = nullptr, std::vector<TraceStep>* trace = nullptr);

// Several sequences at once: sequence i continues prompts[i] as generate() would with seed options.seed
// + i, each step's passes stacked into one (Model::forward_cached on a KVBatch), so the weights are
// read once for all of them. A sequence that stops sits the later steps out. hooks (empty, or one per
// sequence, each may be null) apply to their sequence only: each sees its own rows of every activation
// (hook_pattern is not available). traces: one per sequence, as generate()'s trace (attention_top must
// be 0). options.on_token is not used: give generate() one sequence to follow it token by token.
std::vector<Tokens> generate_batch(const Model& model, const std::vector<Tokens>& prompts, const GenerateOptions& options = {},
                                   const std::vector<const Hooks*>& hooks = {},
                                   std::vector<std::vector<TraceStep>>* traces = nullptr);

}  // namespace tfi
