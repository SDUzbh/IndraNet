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
    int64_t token = -1;           // the chosen token
    int64_t position = -1;        // the position whose logits chose it (the token goes at position + 1)
    float prob = 0, logprob = 0;  // its probability under the model's distribution (before temperature, top_k, top_p)
    std::vector<int64_t> top_tokens;
    std::vector<float> top_probs;
    std::vector<LensEntry> lens;  // target_prob / target_rank: the chosen token's, at each stage
    DLAResult dla;                // parts of the chosen token's logit
    struct Attention {
        std::string head;                // "L3H5"
        std::vector<int64_t> positions;  // most attended first
        std::vector<float> weights;
    };
    std::vector<Attention> attention;  // every head, with TraceOptions::attention_top
};

// Speculative decoding (GenerateOptions::speculative): each step proposes a few tokens cheaply, then the
// model checks them all in one pass (its weights read once, as for a single token) and keeps those it
// agrees with, plus one token of its own. The draft only changes the speed, never what is generated:
// greedy decoding gives the tokens it gives without speculation (up to rounding, since the checking pass
// computes several rows at once), and sampling draws from the same distribution by the standard
// acceptance rule (Leviathan et al. 2023, Chen et al. 2023), though not the same tokens for a given seed.
// Hooks run as they do without speculation (see DecodeRuntime, tfi/runtime.h).
struct SpeculativeStats {
    int64_t steps = 0;     // passes of the model that chose tokens, the prompt's included
    int64_t drafted = 0;   // tokens proposed
    int64_t accepted = 0;  // ... and kept
    int64_t reruns = 0;    // kept tokens computed again, because a hook didn't do what was predicted
};

struct SpeculativeOptions {
    // The most tokens proposed per step; 0: no speculation. Steps propose fewer after a rejection (down
    // to 1) and more after a step that kept them all, so text the draft can't predict costs little.
    int64_t draft_tokens = 0;
    // Who proposes them: `draft`, a smaller model with the same tokenizer (e.g. Llama 3.2 1B for Llama
    // 3.1 8B), sampling with the same options; once the text holds a token past its vocabulary, the model
    // continues alone. Or, without one, prompt lookup: the tokens that followed
    // the most recent earlier occurrence of the text's last `ngram` tokens (or fewer, down to 1). Prompt
    // lookup costs nothing and helps where the output repeats the text before it (editing, summarizing,
    // quoting, code); elsewhere steps without a match are plain steps.
    const Model* draft = nullptr;
    int64_t ngram = 3;
    SpeculativeStats* stats = nullptr;  // if set, receives the counts
};

struct GenerateOptions {
    int64_t max_new_tokens = 32;
    double temperature = 0;  // 0: greedy, the most likely token (the lowest id among equal ones)
    int64_t top_k = 0;       // sample from the k most likely tokens only; 0: all
    double top_p = 1;        // ... from the fewest most likely tokens whose probabilities reach top_p
    uint64_t seed = 0;       // the same seed, model, prompt and options give the same tokens, on every platform
                             // (as far as the logits are the same: Sampler's arithmetic is)
    std::vector<int64_t> stop_tokens;  // stop after generating one of these (it is returned)
    bool stop_at_eos = true;           // ... or one of the model's eos_tokens()
    // Called with each new token as it is chosen; returning false stops after it.
    std::function<bool(int64_t token)> on_token;
    TraceOptions trace;              // what generate() records when given a trace to fill
    SpeculativeOptions speculative;  // generate() only
};

// Picks the next token from a row of logits by GenerateOptions' rules: temperature, then top_k, then
// top_p. Logits that are NaN or -inf are never picked. Each candidate's weight is exp((logit - max) /
// temperature) in float (detail::exp_weights: the same bits on every platform); top_k keeps the k largest
// logits and top_p the fewest most likely tokens whose weights reach top_p of the total (lower ids first
// among equal ones), both found in linear time without sorting; the pick is then drawn from the kept
// tokens in id order, with one uniform number from the generator's top 53 bits.
class Sampler {
  public:
    explicit Sampler(const GenerateOptions& options);
    int64_t operator()(const float* logits, int64_t n);
    // The distribution operator() samples from, over all n tokens: zero outside the candidates, one-hot
    // when it would pick the most likely token (greedy). For speculative decoding's acceptance rule.
    void distribution(const float* logits, int64_t n, std::vector<double>& p);
    double uniform();                            // in [0, 1), from the generator's top 53 bits
    int64_t draw(const std::vector<double>& w);  // index i with probability w[i] / sum(w)
    // Whether every pick is the most likely token, whatever the logits: temperature 0, top_k 1, or a
    // temperature whose reciprocal overflows a float. (A row with an infinite logit is picked greedily too.)
    bool greedy() const;

  private:
    // The tokens that can be picked, lowest id first: their weights in w_ and ids in ids_ (empty: every
    // token, w_ then holding one weight per id), and in `total` the weights' sum. False when the pick is
    // `best` alone (greedy, top_k 1, a temperature whose reciprocal overflows a float, or an infinite logit).
    bool select(const float* logits, int64_t n, int64_t& best, double& total);
    int64_t id(size_t j) const { return ids_.empty() ? static_cast<int64_t>(j) : ids_[j]; }
    double temperature_, top_p_;
    int64_t top_k_;
    std::mt19937_64 rng_;
    std::vector<float> w_, gathered_;
    std::vector<int64_t> ids_, sel_;
    std::vector<uint32_t> keys_;
    std::vector<double> hist_;
};

// New tokens continuing `prompt` (not including it), at most max_new_tokens. The prompt runs in one
// pass, then each new token alone. `hooks` apply to every pass: the prompt's activations first, then
// each new token's (see Model::forward_cached for their shapes), so an intervention lasts through the
// whole text.
// With `trace`, it also receives a TraceStep per new token (options.trace says what each holds), before
// on_token is called for that token. Activations are recorded after `hooks` ran: the trace shows what
// the model computed, interventions included.
// With options.speculative.draft_tokens, steps are speculative (see SpeculativeOptions), through a
// DecodeRuntime (tfi/runtime.h): hooks and on_token see what they see without speculation.
Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options = {},
                const Hooks* hooks = nullptr, std::vector<TraceStep>* trace = nullptr);

// Several sequences at once: sequence i continues prompts[i] as generate() would with seed options.seed
// + i, each step's passes stacked into one (Model::forward_cached on a KVBatch), so the weights are
// read once for all of them. A sequence that stops sits the later steps out. hooks (empty, or one per
// sequence, each may be null) apply to their sequence only: each sees its own rows of every activation
// (hook_pattern is not available). traces: one per sequence, as generate()'s trace (attention_top must
// be 0). options.on_token and options.speculative are not used: give generate() one sequence for them.
std::vector<Tokens> generate_batch(const Model& model, const std::vector<Tokens>& prompts,
                                   const GenerateOptions& options = {}, const std::vector<const Hooks*>& hooks = {},
                                   std::vector<std::vector<TraceStep>>* traces = nullptr);

}  // namespace tfi
