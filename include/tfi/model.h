// Inference-only re-implementation of indranet.DecoderLM with named hook points.
//
// Hook names and activation semantics match indranet exactly (minus the batch dimension),
// e.g. "blocks.0.attn.hook_pattern" is (H, T, T) here and (1, H, T, T) in PyTorch.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tfi/config.h"
#include "tfi/tokenizer.h"
#include "tfi/tensor.h"
#include "tfi/weight.h"

namespace tfi {

namespace detail {
struct RowBatch;  // internal (src/ops.h)
struct TokenizerSlot;  // a stored tokenizer.json and, once parsed, the Tokenizer (src/model.cpp)
class Gpu;        // internal (src/gpu.h, macOS)
struct GpuKV;     // internal (src/gpu.h, macOS): a KVCache's copy on the GPU

// A model's GPU context, created on first use. Copies start empty: each model has its own context
// (a copy shares the weights' memory, which each context uses in place).
struct GpuSlot {
    std::shared_ptr<Gpu> gpu;
    bool checked = false;
    GpuSlot() = default;
    GpuSlot(const GpuSlot&) {}
    GpuSlot& operator=(const GpuSlot&) {
        gpu.reset();
        checked = false;
        return *this;
    }
};
}  // namespace detail

using Tokens = std::vector<int64_t>;
using ActivationCache = std::map<std::string, Tensor>;
// Called with the hook name and the activation; may modify the activation in place.
using HookFn = std::function<void(const std::string& name, Tensor& act)>;
using NameFilter = std::function<bool(const std::string& name)>;

class Hooks {
  public:
    Hooks& add(const std::string& name, HookFn fn);       // exact hook name
    Hooks& add_filter(NameFilter filter, HookFn fn);      // every hook the filter accepts
    // A hook that only reads (e.g. caches) the activations the filter accepts: on the GPU it may be
    // called after the pass has moved on, and out of order with other hooks, with a copy of the
    // activation as it was at the hook point, so the pass need not stop for it.
    Hooks& add_capture(NameFilter filter, HookFn fn);
    Hooks& extend(const Hooks& other);
    void run(const std::string& name, Tensor& act) const;  // in registration order
    bool matches(const std::string& name) const;            // whether run(name) would call anything
    bool changes(const std::string& name) const;            // ... anything but captures
    bool empty() const { return entries_.empty(); }

  private:
    struct Entry {
        NameFilter filter;
        HookFn fn;
        bool capture;
    };
    std::vector<Entry> entries_;
};

// The longest vocabulary entry at each point of text (indranet's character vocabularies).
std::vector<int64_t> tokenize_longest_match(const std::vector<std::string>& vocab, const std::string& text);

class Model;

// The keys and values every layer computed for the positions of a sequence run so far (after qk_norm
// and RoPE, as hook_k and hook_v see them), so that Model::forward_cached runs only new tokens. A copy
// continues on its own: several continuations of one prompt can share its prefill.
//
// With a model whose weights are fp16 (WeightFormat::fp16, the indra CLI's --fp16), the keys and values are
// rounded to fp16 where they are stored (after the hooks, clamped to fp16's finite range), and the GPU's
// copy is held in fp16: half the memory, at an error about that of fp16 weights. Float32 models keep them
// in float32, exactly.
//
// Two places may hold them: the CPU's tensors and (macOS, when steps run on the GPU) the GPU's buffers,
// so that a sequence generated on the GPU isn't kept twice. Both are prefixes of the sequence, since a
// cache only grows at its end or is truncated there: positions [0, cpu_len_) are up to date on the CPU,
// [0, gpu_len_) on the GPU, and every position on at least one side (max(cpu_len_, gpu_len_) == size()).
// GPU steps append on the GPU only; whatever reads the CPU side (a CPU step, keys() / values(), a copy, the
// GPU's buffers growing) first brings it up to date. A cache is not safe to use from two threads at once,
// keys() and values() included.
class KVCache {
  public:
    KVCache() = default;
    // A copy has the keys and values (on the CPU), not the GPU copy of them: it makes its own when it needs one.
    KVCache(const KVCache& o);
    KVCache& operator=(const KVCache& o);
    KVCache(KVCache&&) = default;
    KVCache& operator=(KVCache&&) = default;

    int64_t size() const { return len_; }  // positions held
    bool empty() const { return len_ == 0; }
    void clear();
    void truncate(int64_t n);  // keep positions 0..n-1 (0 <= n <= size()), e.g. to rerun from there
    // Layer `layer`'s keys or values, (Hkv, size(), D): what hook_k / hook_v gave for these positions.
    Tensor keys(int64_t layer) const;
    Tensor values(int64_t layer) const;

  private:
    friend class Model;
    friend class KVBatch;
    // Per layer (Hkv, capacity, D): room to grow without copying each step. Mutable: reading the CPU side
    // brings it up to date from the GPU first.
    mutable std::vector<Tensor> k_, v_;
    int64_t len_ = 0;
    mutable int64_t cpu_len_ = 0;
    // macOS: the GPU's copy (its region `region_` of the buffers: a KVBatch keeps every sequence's in one).
    std::shared_ptr<detail::GpuKV> gpu_;
    int64_t region_ = 0;
    int64_t gpu_len_ = 0;
    // fp16 values (a model with fp16 weights): keys and values rounded to fp16 where they are stored, and
    // the GPU's copy held in fp16 (half the memory); the CPU copy holds the same values as float32. Set by
    // the first step: a model of the other precision can't continue the cache.
    bool half_ = false;
    void append(const std::vector<Tensor>& k, const std::vector<Tensor>& v, int64_t n);  // on the CPU, which is up to date
    void sync_cpu() const;                // [cpu_len_, size()) from the GPU
    void reserve_cpu(int64_t n) const;    // CPU capacity for n positions, keeping [0, cpu_len_)
    void shapes(const std::vector<int64_t>& heads, const std::vector<int64_t>& dims);  // per-layer shapes, if none yet
    void check() const;                   // with TFI_CHECK_KV set: the invariant, and both sides equal where both hold
};

// Several sequences' KV caches, continued together by Model::forward_cached's batched form: each call
// stacks the sequences given tokens into one pass, so the weights are read once for all of them. The
// caches are read-only from outside (a sequence's history changes only through the batch, which keeps a
// GPU copy of all of them, a region each). A copy has the keys and values, not the GPU copy.
class KVBatch {
  public:
    explicit KVBatch(int64_t n = 0) : caches_(static_cast<size_t>(n)) {}
    KVBatch(const KVBatch& o) : caches_(o.caches_) {}
    KVBatch& operator=(const KVBatch& o) {
        caches_ = o.caches_;
        gpu_.reset();
        return *this;
    }
    KVBatch(KVBatch&&) = default;
    KVBatch& operator=(KVBatch&&) = default;
    int64_t size() const { return static_cast<int64_t>(caches_.size()); }
    const KVCache& operator[](int64_t i) const { return caches_.at(static_cast<size_t>(i)); }

  private:
    friend class Model;
    std::vector<KVCache> caches_;
    std::shared_ptr<detail::GpuKV> gpu_;  // macOS: every sequence's keys and values on the GPU, a region each
};

class Model {
  public:
    // `format` fp16 keeps the 2-D weights in half precision (see WeightFormat).
    Model(ModelConfig cfg, std::map<std::string, Tensor> weights, WeightFormat format = WeightFormat::fp32);
    // A safetensors file written by indranet.export. With fp16, each weight is converted as it is
    // read, so the float32 copy of the whole model never exists.
    static Model load(const std::string& path, WeightFormat format = WeightFormat::fp32);

    const ModelConfig& cfg() const { return cfg_; }
    const ModelConfig& layer_cfg(int64_t layer) const { return layer_cfgs_.at(layer); }

    // Logits (T, vocab) for one sequence. Hooks may read or overwrite activations.
    Tensor forward(const Tokens& tokens, const Hooks* hooks = nullptr) const;
    // Logits for `tokens` continuing the sequence held in `cache`, at positions cache.size() onwards: the
    // rows forward() on the whole sequence gives for them, (T, vocab), or with last_only just the last,
    // (1, vocab). `cache` then holds these positions too; if anything throws it is left as it was.
    // With an empty cache this is forward() (on the GPU when that is worth it). Later calls compute
    // only the new rows, so hooks see those: activations are (T, ...), hook_pattern is
    // (H, T, cache.size() + T), each new query over every key, and hook_pos_embed is the new positions'.
    // Whatever hook_k and hook_v leave is what later tokens attend to.
    // macOS: these calls go to the GPU for a prompt (empty cache) whose single pass would, for later
    // chunks of tokens large enough, for single tokens of models of a few hundred million weights or more,
    // and always in GPU-only mode; then the cache keeps a copy of its keys and values on the GPU, appended
    // to there (a prompt's go straight in), and the pass, final norm and unembedding included, is one GPU
    // submission, also with fp16 multiplies (set_gpu_fp16). Not when a hook reads hook_pattern, or a
    // layer's heads are too wide for the GPU's attention: the CPU then (a prompt: forward()'s whole pass).
    Tensor forward_cached(const Tokens& tokens, KVCache& cache, const Hooks* hooks = nullptr, bool last_only = false) const;
    // Several sequences in one pass: tokens[i] continues sequence i of `batch` (empty: it sits this call
    // out). Returns each sequence's logits as forward_cached would give them (an empty tensor for one that
    // sat out). The sequences' rows are stacked in order, so hooks see (rows of every sequence given
    // tokens, ...): hook_q / hook_k / hook_v (H, rows, D), the others (rows, ...). hook_pattern is not
    // available (each sequence has its own pattern): a hook on it is refused. On the GPU under the same
    // rules as forward_cached, the batch keeping every sequence's keys and values there.
    std::vector<Tensor> forward_cached(const std::vector<Tokens>& tokens, KVBatch& batch, const Hooks* hooks = nullptr,
                                       bool last_only = false) const;
    // Run only blocks layer..n_layers-1 (then the final norm and unembedding), starting from
    // `resid`, the (T, d_model) input to block `layer`: what hook_resid_pre of that block sees.
    // With a cached blocks.<layer>.hook_resid_pre this reproduces forward() exactly while skipping
    // the earlier layers, which is how ablation avoids recomputing them (patching uses forward_rows).
    Tensor forward_from(int64_t layer, Tensor resid, const Hooks* hooks = nullptr) const;
    // Internal, for patching: run blocks layer..n_layers-1 on several sequences' rows stacked into
    // one (rows, d_model) residual (see detail::RowBatch). Hooks see the stacked activations
    // (hook_pattern is not called). Returns the logits of each segment's last row: (segments, vocab).
    Tensor forward_rows(int64_t layer, Tensor resid, const detail::RowBatch& batch, const Hooks* hooks = nullptr) const;
    // Internal, for ablation: like forward_rows, but returns every row after the final norm,
    // (rows, d_model), for the caller to unembed (logits_from_final) in pieces.
    Tensor forward_rows_final(int64_t layer, Tensor resid, const detail::RowBatch& batch,
                              const Hooks* hooks = nullptr) const;
    // Internal, for attribution patching (macOS, needs batch.gpu): forward and backward passes of
    // several whole sequences stacked as in forward_rows (every segment starts at position 0), from
    // `x0`, the (rows, d_model) input to block 0, everything but attention on the GPU.
    // `dmetric(s, logits)` gives dF/d(logits) for segment s's scalar F, from the logits (1, vocab) of
    // its last row. Returns each segment's gradients, as run_with_gradients would, for the hook points
    // the filter accepts among hook_embed, hook_resid_pre / _mid / _post, hook_attn_out, hook_mlp_out
    // and attn.hook_z (no others).
    std::vector<ActivationCache> gradients_rows(const Tensor& x0, const detail::RowBatch& batch,
                                                const std::function<Tensor(int64_t, const Tensor&)>& dmetric,
                                                const NameFilter& filter) const;
    // Internal (macOS): this model's GPU context for batched patching, created on first use and
    // kept with its copies of the weights; nullptr without a Metal device or on other systems.
    detail::Gpu* gpu() const;
    bool has_gpu() const;  // Internal: whether gpu() has created the context (its start-up is paid)
    // Like forward, but also stores activations (after `hooks` ran) for hooks the filter accepts.
    Tensor run_with_cache(const Tokens& tokens, ActivationCache& cache, NameFilter filter = nullptr,
                          const Hooks* hooks = nullptr) const;
    std::vector<std::string> hook_names() const;

    // Forward pass, then a backward pass from a gradient on the logits: `dlogits(logits)` returns
    // dF/dlogits (T, vocab) for some scalar function F of the logits. For every hook point the
    // filter accepts, `grads` receives dF/d(activation), the activation's shape. Each hook point is
    // its own variable: e.g. hook_resid_pre's gradient counts every path from it, hook_attn_in's
    // only the path through attention (as PyTorch autograd on indranet's hook points gives).
    // The forward pass is forward()'s. `hooks` may edit activations; the backward pass then treats
    // each edited value as the activation itself (e.g. replacing hook_embed runs the model on other
    // embeddings). If `acts` is given it receives every activation. Returns the logits.
    Tensor run_with_gradients(const Tokens& tokens, const std::function<Tensor(const Tensor& logits)>& dlogits,
                              ActivationCache& grads, NameFilter filter = nullptr,
                              ActivationCache* acts = nullptr, const Hooks* hooks = nullptr) const;

    const Weight& weight(const std::string& name) const;  // throws, naming the missing tensor
    const Weight* weight_or_null(const std::string& name) const;
    const Weight& unembed() const;  // (vocab, d_model); the embedding when tied
    int64_t num_parameters() const;
    int64_t max_heads() const;  // largest head count of any layer (layer_overrides may change n_heads)
    const std::map<std::string, Weight>& weights() const { return w_; }
    WeightFormat weight_format() const { return format_; }  // as requested at load
    int64_t weight_bytes() const;                           // memory the weights' values take
    // With WeightFormat::fp16: the 2-D weights kept in float32 because a value did not fit fp16.
    std::vector<std::string> fp32_weights() const;

    // Optional vocabulary (one string per token id), read from the file when indranet.export
    // was given one. Without it, tokens are plain ids.
    const std::vector<std::string>& vocab() const { return vocab_; }
    void set_vocab(std::vector<std::string> vocab);
    std::string token_str(int64_t id) const;     // the token's text, or its id
    // Optional tokenizer: a Hugging Face tokenizer.json (stored by indranet_hf), parsed on first use.
    bool has_tokenizer() const;
    const Tokenizer& tokenizer() const;  // throws std::runtime_error without one, or if it can't be read
    void set_tokenizer_json(std::string tokenizer_json);
    // Token ids for text: the model's tokenizer when it has one (add_special_tokens: its BOS and other
    // post-processor tokens, as Hugging Face adds by default), otherwise the longest vocabulary entry
    // at each point (indranet's character vocabularies; add_special_tokens has no effect).
    Tokens tokenize(const std::string& text, bool add_special_tokens = true) const;
    // Text for token ids: the tokenizer's decoding when the model has one (see Tokenizer::decode), else
    // the vocabulary's strings joined. Bytes, which may end inside a UTF-8 character. Throws without either.
    std::string decode(const Tokens& ids, bool skip_special_tokens = false) const;
    // Token ids that end a text (Hugging Face's eos_token_id, stored with converted models; empty for
    // others). generate() stops at them by default.
    const std::vector<int64_t>& eos_tokens() const { return eos_; }
    void set_eos_tokens(std::vector<int64_t> ids);

    // Building blocks reused by the analysis tools.
    Tensor norm(const std::string& prefix, const ModelConfig& c, const Tensor& x) const;  // over the last dim
    Tensor final_norm(const Tensor& resid) const;    // ln_final (identity for post-norm models)
    Tensor logits_from_final(const Tensor& normed) const;  // unembed + final softcap
    // The same, with the unembedding on the GPU when `batch` allows it (as in forward_rows).
    Tensor logits_from_final(const Tensor& normed, const detail::RowBatch* batch) const;
    // The input to block 0, (T, d_model): token (+ positional) embeddings, as hook_resid_pre of
    // block 0 sees it, for tokens at positions start, start + 1, ... Calls hook_embed and hook_pos_embed.
    Tensor embed(const Tokens& tokens, const Hooks* hooks = nullptr, int64_t start = 0) const;

  private:
    Model(ModelConfig cfg, std::map<std::string, Weight> weights, WeightFormat format);
    void check_shapes();

    ModelConfig cfg_;
    std::vector<ModelConfig> layer_cfgs_;
    WeightFormat format_ = WeightFormat::fp32;
    std::map<std::string, Weight> w_;  // declared before gpu_, which may use their memory in place
    std::vector<std::string> vocab_;
    std::vector<int64_t> eos_;
    std::shared_ptr<detail::TokenizerSlot> tokenizer_;
    mutable detail::GpuSlot gpu_;

    Tensor attention(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const;
    // Attention from the projections q (T, H*D), k and v (T, Hkv*D): qk_norm, RoPE, hook_q/k/v, the
    // pattern and hook_z. Returns z as (T, H*D), before o_proj.
    Tensor attend(int64_t layer, Tensor q, Tensor k, Tensor v, const Hooks* hooks, const detail::RowBatch* batch) const;
    // attend() reading q, k, v as row-major arrays of T rows and writing z (T, H*D) to z_out: the
    // batched GPU passes use it on their GPU buffers in place.
    void attend_into(int64_t layer, const float* q, const float* k, const float* v, int64_t T, const Hooks* hooks,
                     const detail::RowBatch* batch, float* z_out) const;
    // Blocks layer..n_layers-1 of a batched pass; returns the residual stream after the last one:
    // every row, or with last_rows only each segment's last row, (segments, d_model). With a KV cache on
    // the GPU (batch.kv) and `logits`, the final norm (hook_final, every row) and the unembedding (of the
    // last row only with logits_last_only) run in the same GPU submission, into *logits, before any final
    // softcap.
    Tensor run_rows(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks,
                    bool last_rows = false, Tensor* logits = nullptr, bool logits_last_only = false) const;
    Tensor mlp(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const;
    Tensor block(int64_t layer, Tensor x, const Hooks* hooks, const detail::RowBatch* batch = nullptr) const;
    Tensor positional_embedding(int64_t start, int64_t T) const;  // positions start..start+T-1
    Tensor positional_embedding_at(const std::vector<int64_t>& positions) const;  // one row per position
    // embed() with each token's own position (several sequences stacked, each at its own place).
    Tensor embed_at(const Tokens& tokens, const std::vector<int64_t>& positions, const Hooks* hooks) const;
    // forward_from, with only the last row unembedded when last_only (hook_final still sees every row).
    Tensor run_from(int64_t layer, Tensor resid, const Hooks* hooks, bool last_only) const;
};

}  // namespace tfi
