#include "tfi/analysis.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <thread>

#include "analysis/testing.h"
#include "gpu/gpu.h"
#include "gpu/gpu_chunks.h"
#include "model/ops.h"
#include "model/placement.h"
#include "support/half.h"
#include "support/parallel.h"

#ifdef TFI_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace tfi {

using detail::parallel_for;

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();

std::string head_label(int64_t layer, int64_t head) { return std::format("L{}H{}", layer, head); }

// (n_layers, widest layer's head count), `fill` where a head exists and NaN where a layer has
// fewer heads (layer_overrides may change n_heads per layer).
Tensor per_head_matrix(const Model& m, float fill) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    Tensor t({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) t.data[i * H + h] = fill;
    return t;
}
std::string lp(int64_t layer) { return std::format("blocks.{}.", layer); }

// A cache filter accepting exactly these hook names, so a tool copies only what it reads.
NameFilter only(std::set<std::string> names) {
    return [names = std::move(names)](const std::string& n) { return names.contains(n); };
}

void check_pos(const Tokens& tokens, int64_t pos) {
    if (pos < 0 || pos >= std::ssize(tokens))
        throw std::out_of_range(std::format("position {} is outside the sequence", pos));
}

void check_token(const Model& m, int64_t t, const char* what) {
    if (t < 0 || t >= m.cfg().vocab_size)
        throw std::out_of_range(std::string(what) + " token is outside the vocabulary");
}

// Unembedding direction for the metric: W_U[target] - W_U[other] (or W_U[target]).
std::vector<float> logit_direction(const Model& m, int64_t target, int64_t other) {
    const Weight& U = m.unembed();
    const Tensor ut = U.row(target);
    std::vector<float> u(ut.data);
    if (other >= 0) {
        const Tensor uo = U.row(other);
        for (size_t j = 0; j < u.size(); ++j) u[j] -= uo.data[j];
    }
    return u;
}

// tr(A) for a square matrix.
float trace(const Tensor& A) {
    double s = 0;
    for (int64_t i = 0; i < A.shape[0]; ++i) s += A.data[i * A.shape[1] + i];
    return static_cast<float>(s);
}

// || L^T C R ||_F for L, R of shape (r, d) and C (r, r), computed through r x r Gram matrices:
// tr(C^T (L L^T) C (R R^T)).
float frob_lcr(const Tensor& GL, const Tensor& C, const Tensor& GR) {
    const Tensor M = matmul(matmul(transpose2d(C), GL), matmul(C, GR));
    return std::sqrt(std::max(0.f, trace(M)));
}

// Composition of writer (L0 index i) into reader j, given each head's weights and Gram matrices.
struct HeadCache {
    HeadWeights w;
    Tensor GQ, GK, GV, GOt;  // W_Q W_Q^T, W_K W_K^T, W_V W_V^T, W_O^T W_O, all (r, r)
    float qk_norm, ov_norm;  // ||W_Q^T W_K||_F, ||W_O W_V||_F
};

HeadCache cache_head(HeadWeights w) {
    HeadCache c;
    c.GQ = matmul(w.W_Q, w.W_Q, true);
    c.GK = matmul(w.W_K, w.W_K, true);
    c.GV = matmul(w.W_V, w.W_V, true);
    c.GOt = matmul(transpose2d(w.W_O), w.W_O);
    c.qk_norm = std::sqrt(std::max(0.f, trace(matmul(c.GQ, c.GK))));
    c.ov_norm = std::sqrt(std::max(0.f, trace(matmul(c.GOt, c.GV))));
    c.w = std::move(w);
    return c;
}

struct CompScores {
    float q, k, v;
};

CompScores compose(const HeadCache& writer, const HeadCache& reader) {
    // Q: W_QK^T W_OV = W_K^T (W_Q W_O) W_V;  K: W_QK W_OV = W_Q^T (W_K W_O) W_V;
    // V: W_OV' W_OV = W_O' (W_V' W_O) W_V, with W_O' = (W_O'^T)^T.
    const Tensor Cq = matmul(reader.w.W_Q, writer.w.W_O);
    const Tensor Ck = matmul(reader.w.W_K, writer.w.W_O);
    const Tensor Cv = matmul(reader.w.W_V, writer.w.W_O);
    const float denom_qk = reader.qk_norm * writer.ov_norm, denom_ov = reader.ov_norm * writer.ov_norm;
    return {frob_lcr(reader.GK, Cq, writer.GV) / denom_qk, frob_lcr(reader.GQ, Ck, writer.GV) / denom_qk,
            frob_lcr(reader.GOt, Cv, writer.GV) / denom_ov};
}

}  // namespace

float logit_metric(const Tensor& logits, int64_t pos, int64_t target, int64_t other) {
    const int64_t V = logits.shape[1];
    const float* l = logits.ptr() + pos * V;
    return other >= 0 ? l[target] - l[other] : log_softmax_at(logits.row_view(pos), target);
}

float next_token_loss(const Tensor& logits, const Tokens& tokens) {
    const int64_t T = std::ssize(tokens);
    if (T < 2) throw std::invalid_argument("loss needs at least two tokens");
    double s = 0;
    for (int64_t t = 0; t + 1 < T; ++t) s -= log_softmax_at(logits.row_view(t), tokens[t + 1]);
    return static_cast<float>(s / (T - 1));
}

// --- logit lens ---------------------------------------------------------------------

namespace {
// (label, hook point) of every stage the logit lens decodes.
std::vector<std::pair<std::string, std::string>> lens_stages(const Model& m) {
    std::vector<std::pair<std::string, std::string>> stages = {{"embed", "blocks.0.hook_resid_pre"}};
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) {
        const std::string L = std::format("L{}", i);
        if (m.layer_cfg(i).has_mlp()) {
            stages.push_back({L + ".attn", lp(i) + "hook_resid_mid"});
            stages.push_back({L + ".mlp", lp(i) + "hook_resid_post"});
        } else {
            stages.push_back({L + ".attn", lp(i) + "hook_resid_post"});
        }
    }
    return stages;
}
}  // namespace

std::vector<std::string> lens_hook_points(const Model& m) {
    std::vector<std::string> out;
    for (const auto& [label, hook] : lens_stages(m)) out.push_back(hook);
    return out;
}

std::vector<LensEntry> logit_lens_from(const Model& m, const ActivationCache& cache, int64_t row, int64_t target,
                                       int64_t k) {
    if (target >= 0) check_token(m, target, "target");
    const std::vector<std::pair<std::string, std::string>> stages = lens_stages(m);
    const int64_t d = m.cfg().d_model, S = std::ssize(stages);
    // Every stage's row, decoded together: one pass over the unembedding instead of one per stage.
    Tensor stack({S, d});
    for (int64_t s = 0; s < S; ++s) {
        const Tensor& a = cache.at(stages[s].second);
        if (a.dim() != 2 || a.shape[1] != d || row < 0 || row >= a.shape[0])
            throw std::out_of_range(std::format("logit lens: row {} is outside {}", row, stages[s].second));
        std::copy(a.ptr() + row * d, a.ptr() + (row + 1) * d, stack.ptr() + s * d);
    }
    Tensor probs = m.logits_from_final(m.final_norm(stack));
    std::vector<LensEntry> out;
    for (int64_t s = 0; s < S; ++s) {
        const std::span<float> p = probs.row_view(s);
        softmax_(p);
        LensEntry e;
        e.stage = stages[s].first;
        e.top_tokens = topk(p, k);
        for (int64_t t : e.top_tokens) e.top_probs.push_back(p[t]);
        if (target >= 0) {
            e.target_prob = p[target];
            e.target_rank = rank_of(p, target);
        }
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<LensEntry> logit_lens(const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t k) {
    check_pos(tokens, pos);
    if (target >= 0) check_token(m, target, "target");
    // Only the stages decoded: later layers' resid_pre would just duplicate the previous resid_post.
    const std::vector<std::string> points = lens_hook_points(m);
    ActivationCache cache;
    m.run_with_cache(tokens, cache, only(std::set<std::string>(points.begin(), points.end())));
    return logit_lens_from(m, cache, pos, target, k);
}

// --- direct logit attribution --------------------------------------------------------

namespace {
void check_dla_model(const Model& m) {
    if (m.cfg().norm_position == "post")
        throw std::runtime_error(
            "direct logit attribution needs an additive residual stream; post-norm models have none");
    for (int64_t i = 0; i < m.cfg().n_layers; ++i)  // a single post-norm layer (via layer_overrides) breaks it too
        if (m.layer_cfg(i).norm_position == "post")
            throw std::runtime_error(
                std::format("direct logit attribution needs an additive residual stream; layer {} is post-norm", i));
}
}  // namespace

std::vector<std::string> dla_hook_points(const Model& m) {
    const int64_t L = m.cfg().n_layers;
    std::vector<std::string> out = {"hook_embed", "hook_final", lp(L - 1) + "hook_resid_post"};
    if (m.cfg().embedding_positional()) out.push_back("hook_pos_embed");
    for (int64_t i = 0; i < L; ++i) {
        const ModelConfig& c = m.layer_cfg(i);
        out.push_back(lp(i) + (c.norm_position == "sandwich" ? "hook_attn_out" : "attn.hook_z"));
        if (c.has_mlp()) out.push_back(lp(i) + "hook_mlp_out");
    }
    return out;
}

DLAResult direct_logit_attribution_from(const Model& m, const ActivationCache& cache, int64_t row, int64_t target,
                                        int64_t other) {
    check_dla_model(m);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");
    const ModelConfig& cfg = m.cfg();
    const int64_t d = cfg.d_model, L = cfg.n_layers;
    auto at_row = [&](const std::string& name, int64_t width) {  // row `row` of a (rows, width...) activation
        const Tensor& a = cache.at(name);
        if (a.dim() < 2 || row < 0 || row >= a.shape[0] || a.numel() / a.shape[0] != width)
            throw std::out_of_range(std::format("direct logit attribution: row {} is outside {}", row, name));
        return std::span<const float>(a.ptr() + row * width, static_cast<size_t>(width));
    };
    const std::vector<float> u = logit_direction(m, target, other);

    // Final norm as a fixed linear map: f(c) = gamma * (c - mean(c)) / sigma for LayerNorm,
    // gamma * c / rms for RMSNorm, with sigma / rms taken from the real final residual.
    const std::span<const float> resid = at_row(lp(L - 1) + "hook_resid_post", d);
    std::vector<float> w(u);  // u * gamma / scale
    bool center = false;
    float bias_term = 0;
    if (cfg.norm != "none") {
        const Tensor& gamma = m.weight("ln_final.weight").f32();
        double mean = 0, sq = 0;
        for (int64_t j = 0; j < d; ++j) mean += resid[j];
        mean /= d;
        if (cfg.norm == "layernorm") {
            for (int64_t j = 0; j < d; ++j) sq += (resid[j] - mean) * (resid[j] - mean);
            center = true;
        } else {
            for (int64_t j = 0; j < d; ++j) sq += double(resid[j]) * resid[j];
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sq / d) + static_cast<float>(cfg.norm_eps));
        for (int64_t j = 0; j < d; ++j) w[j] = u[j] * gamma.data[j] * inv;
        if (const Weight* beta = m.weight_or_null("ln_final.bias")) bias_term = dot(u, beta->f32().data);
    }
    // LayerNorm's centring is linear too: w . (c - mean(c)) = (w - mean(w)) . c.
    if (center) {
        double wm = 0;
        for (float v : w) wm += v;
        wm /= d;
        for (float& v : w) v -= static_cast<float>(wm);
    }
    auto contribution = [&](std::span<const float> c) { return dot(w, c); };

    DLAResult res;
    auto push = [&](const std::string& label, float v) {
        res.labels.push_back(label);
        res.values.push_back(v);
        res.sum += v;
    };
    push("embed", contribution(at_row("hook_embed", d)));
    if (cache.contains("hook_pos_embed")) push("pos_embed", contribution(at_row("hook_pos_embed", d)));
    const Tensor wrow({1, d}, w);
    for (int64_t i = 0; i < L; ++i) {
        const ModelConfig& c = m.layer_cfg(i);
        if (c.norm_position == "sandwich") {  // per-head outputs pass through a norm; attribute the layer
            push(std::format("L{}.attn", i), contribution(at_row(lp(i) + "hook_attn_out", d)));
        } else {
            // Head h writes o_proj's columns h*D..(h+1)*D times z_h, so its part is (w W_O)[h*D..] . z_h:
            // one pass over W_O for the layer, then a dot of D per head.
            const int64_t H = c.n_heads, D = c.head_dim();
            const std::span<const float> z = at_row(lp(i) + "attn.hook_z", H * D);  // (H, D) at this row
            const Tensor g = matmul(wrow, m.weight(lp(i) + "attn.o_proj.weight"));  // (1, H*D)
            for (int64_t h = 0; h < H; ++h)
                push(head_label(i, h), dot(std::span(g.data).subspan(h * D, D), z.subspan(h * D)));
            if (const Weight* bo = m.weight_or_null(lp(i) + "attn.o_proj.bias"))
                push(std::format("L{}.attn_bias", i), contribution(bo->f32().data));
        }
        if (c.has_mlp()) push(std::format("L{}.mlp", i), contribution(at_row(lp(i) + "hook_mlp_out", d)));
    }
    if (bias_term != 0) push("final_norm_bias", bias_term);
    res.actual = dot(u, at_row("hook_final", d));
    return res;
}

DLAResult direct_logit_attribution(const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t other) {
    check_dla_model(m);
    check_pos(tokens, pos);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");
    const std::vector<std::string> points = dla_hook_points(m);
    ActivationCache cache;
    m.run_with_cache(tokens, cache, only(std::set<std::string>(points.begin(), points.end())));
    return direct_logit_attribution_from(m, cache, pos, target, other);
}

// --- head pattern scores --------------------------------------------------------------

HeadScores head_scores(const Model& m, int64_t seg, int64_t n_samples, uint64_t seed) {
    const ModelConfig& cfg = m.cfg();
    if (cfg.positional == "learned") seg = std::min(seg, cfg.max_seq_len / 2);
    if (seg < 2) throw std::invalid_argument("segment length must be at least 2");
    const int64_t L = cfg.n_layers, H = m.max_heads(), T = 2 * seg;  // H: row stride; layer i has heads(i)
    auto heads = [&m](int64_t layer) { return m.layer_cfg(layer).n_heads; };
    HeadScores s;
    for (Tensor* t : {&s.previous_token, &s.induction, &s.duplicate_token, &s.first_token, &s.self})
        *t = per_head_matrix(m, 0.f);
    std::vector<float> two_hop(static_cast<size_t>(L * H * L * H), 0.f);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> tok(0, cfg.vocab_size - 1);
    for (int64_t n = 0; n < n_samples; ++n) {
        Tokens tokens(T);
        for (int64_t t = 0; t < seg; ++t) tokens[t] = tokens[t + seg] = tok(rng);
        ActivationCache cache;
        m.run_with_cache(tokens, cache,
                         [](const std::string& name) { return name.find("hook_pattern") != std::string::npos; });
        std::vector<const Tensor*> P(L);
        for (int64_t i = 0; i < L; ++i) P[i] = &cache.at(lp(i) + "attn.hook_pattern");
        for (int64_t i = 0; i < L; ++i)
            for (int64_t h = 0; h < heads(i); ++h) {
                const float* p = P[i]->ptr() + h * T * T;
                double prev = 0, ind = 0, dup = 0, first = 0, self = 0;
                for (int64_t t = seg; t < T; ++t) {
                    prev += p[t * T + t - 1];
                    ind += p[t * T + t - seg + 1];
                    dup += p[t * T + t - seg];
                    first += p[t * T];
                    self += p[t * T + t];
                }
                const double norm = double(seg) * n_samples;
                s.previous_token.data[i * H + h] += float(prev / norm);
                s.induction.data[i * H + h] += float(ind / norm);
                s.duplicate_token.data[i * H + h] += float(dup / norm);
                s.first_token.data[i * H + h] += float(first / norm);
                s.self.data[i * H + h] += float(self / norm);
            }
        // Two-hop virtual attention reader(t -> r) * writer(r -> t-seg+1), summed over r.
        for (int64_t a = 0; a < L; ++a)
            for (int64_t b = a + 1; b < L; ++b)
                for (int64_t ha = 0; ha < heads(a); ++ha)
                    for (int64_t hb = 0; hb < heads(b); ++hb) {
                        const float* pa = P[a]->ptr() + ha * T * T;
                        const float* pb = P[b]->ptr() + hb * T * T;
                        double acc = 0;
                        for (int64_t t = seg; t < T; ++t) {
                            const int64_t src = t - seg + 1;
                            for (int64_t r = src; r <= t; ++r) acc += double(pb[t * T + r]) * pa[r * T + src];
                        }
                        two_hop[((a * H + ha) * L + b) * H + hb] += float(acc / (double(seg) * n_samples));
                    }
    }
    for (int64_t a = 0; a < L; ++a)
        for (int64_t b = a + 1; b < L; ++b)
            for (int64_t ha = 0; ha < heads(a); ++ha)
                for (int64_t hb = 0; hb < heads(b); ++hb)
                    s.top_two_hop.push_back(
                        {head_label(a, ha), head_label(b, hb), two_hop[((a * H + ha) * L + b) * H + hb]});
    std::ranges::sort(s.top_two_hop, std::ranges::greater{}, &TwoHop::score);
    if (s.top_two_hop.size() > 5) s.top_two_hop.resize(5);
    return s;
}

// --- activation patching ----------------------------------------------------------------

bool PatchResult::normalized() const {
    const float scale = std::max({1.f, std::fabs(clean_metric), std::fabs(corrupt_metric)});
    return std::isfinite(gap()) && std::fabs(gap()) > 1e-5f * scale;
}

// Fill `values` from `effect`, or leave it all NaN when the gap is (numerically) zero.
static void normalize_patch(PatchResult& r) {
    r.values = Tensor(r.effect.shape, kNaN);
    if (!r.normalized()) return;
    for (int64_t i = 0; i < r.effect.numel(); ++i) r.values.data[i] = r.effect.data[i] / r.gap();
}

static void check_patch_inputs(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                               int64_t other) {
    if (clean.size() != corrupt.size())
        throw std::invalid_argument("clean and corrupted inputs must have the same length");
    check_pos(clean, pos);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");
}

// Exact patching needs one patched forward pass per site. It is made cheap in ways that change
// results by float rounding at most (and are the same for any thread count):
// - The metric at `pos` depends only on tokens 0..pos, so both inputs are cut to that prefix.
// - Sites that can't matter are not run; their effect is exactly 0: positions after `pos`, patches
//   that change nothing (clean and corrupted activations identical), and positions before `pos`
//   after the last layer's attention (nothing carries them to `pos`).
// - A patch at layer i leaves layers 0..i-1 as in the corrupted run, so its pass starts at layer i
//   from the cached corrupted residual stream.
// - Passes that start at the same layer run together through Model::forward_rows: their rows are
//   stacked, so every linear layer is one large matrix multiply instead of one small one per pass.
//   On macOS large multiplies run faster on the matrix units, and go to the GPU (gpu.h).
// - A residual-stream patch at position t recomputes only positions t..pos: earlier positions are
//   as in the corrupted run at every layer, and attention reads their keys and values from it.
// - The batches run in parallel (set_num_threads). They are fixed by the sites alone.
namespace {

using Site = std::pair<int64_t, int64_t>;  // (layer, position) or (layer, head)

struct PatchRuns {
    ActivationCache clean_cache, corrupt_cache;  // inputs cut to positions 0..pos
    PatchResult result;
};

// The clean and corrupted runs, caching `clean_hooks` and `corrupt_hooks` (plus every layer's input
// for the corrupted run).
PatchRuns prepare_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other, std::set<std::string> clean_hooks, std::set<std::string> corrupt_hooks) {
    PatchRuns p;
    const Tokens c(clean.begin(), clean.begin() + pos + 1), x(corrupt.begin(), corrupt.begin() + pos + 1);
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) corrupt_hooks.insert(lp(i) + "hook_resid_pre");
    p.result.clean_metric =
        logit_metric(m.run_with_cache(c, p.clean_cache, only(std::move(clean_hooks))), pos, target, other);
    p.result.corrupt_metric =
        logit_metric(m.run_with_cache(x, p.corrupt_cache, only(std::move(corrupt_hooks))), pos, target, other);
    return p;
}

#ifdef TFI_USE_ACCELERATE
// Stacked rows per batch, tuned on an M5 Pro: larger multiplies run faster, but head patching has
// few passes per layer and needs smaller batches to keep every thread busy. Ablation and reference
// means have fewer passes still: 256 rows is the GPU threshold and leaves several batches.
constexpr int64_t kBatchRows = 1024, kHeadBatchRows = 512, kAblationBatchRows = 256;
#else
// The portable matrix multiply is no faster on larger inputs, so one pass per batch keeps every
// thread busy.
constexpr int64_t kBatchRows = 1, kHeadBatchRows = 1, kAblationBatchRows = 1;
#endif
std::atomic<int64_t> g_batch_rows{0}, g_head_batch_rows{0};  // test overrides
int64_t batch_rows() { return g_batch_rows ? g_batch_rows.load() : kBatchRows; }
int64_t head_batch_rows() { return g_head_batch_rows ? g_head_batch_rows.load() : kHeadBatchRows; }
int64_t ablation_batch_rows() { return g_head_batch_rows ? g_head_batch_rows.load() : kAblationBatchRows; }
#ifdef TFI_USE_ACCELERATE
// Attribution patching's gradient runs (steps sequences of P rows), tuned on an M5 Pro: aim for 8
// batches running in parallel, at most 256 rows each, and at least the rows of a batch that goes to the GPU
// (detail::min_gradient_batch_rows).
int64_t gradient_batch_rows(int64_t steps, int64_t P) {
    if (g_head_batch_rows) return g_head_batch_rows.load();
    const int64_t min_rows = detail::min_gradient_batch_rows();
    const int64_t per = std::min(std::max((steps + 7) / 8, (min_rows + P - 1) / P), std::max<int64_t>(1, 256 / P));
    return per * P;
}
#endif

struct Batch {
    int64_t layer;                 // where its passes start
    std::vector<int64_t> items{};  // positions (residual stream) or heads
    int64_t rows, cost;            // stacked rows; rows x layers run
};

// Each layer's items in batches of at most max_rows rows (rows(item) each), most expensive first
// so the threads finish together.
std::vector<Batch> make_batches(std::vector<std::vector<int64_t>> items_per_layer,
                                const std::function<int64_t(int64_t)>& rows, int64_t max_rows) {
    const int64_t L = std::ssize(items_per_layer);
    std::vector<Batch> batches;
    for (int64_t i = 0; i < L; ++i) {
        std::ranges::sort(items_per_layer[i]);  // batches depend only on the set of sites
        Batch b{.layer = i, .rows = 0, .cost = 0};
        auto flush = [&] {
            b.cost = b.rows * (L - i);
            batches.push_back(std::move(b));
            b = Batch{.layer = i, .rows = 0, .cost = 0};
        };
        for (int64_t item : items_per_layer[i]) {
            if (!b.items.empty() && b.rows + rows(item) > max_rows) flush();
            b.items.push_back(item);
            b.rows += rows(item);
        }
        if (!b.items.empty()) flush();
    }
    std::ranges::stable_sort(batches, std::ranges::greater{}, &Batch::cost);
    return batches;
}

#ifdef TFI_USE_ACCELERATE
// Multiply-adds per row per block, about: the blocks' 2-D weights over the layers.

// Multiply-adds of the batches big enough for the GPU (gpu_min_rows): each runs its rows from its
// layer to the last.
double gpu_work(const Model& m, const std::vector<Batch>& batches) {
    double rows_x_layers = 0;
    for (const Batch& b : batches)
        if (b.rows >= detail::gpu_min_rows())
            rows_x_layers += static_cast<double>(b.rows) * static_cast<double>(m.cfg().n_layers - b.layer);
    return rows_x_layers > 0 ? rows_x_layers * detail::block_weights_per_layer(m) : 0.0;
}
#endif

// The model's GPU context (macOS), only when the batches large enough for it (gpu_min_rows) add up to
// enough work to repay starting it (gpu_min_work), or it has started already (its start-up is paid):
// otherwise everything runs on the CPU.
detail::Gpu* gpu_for(const Model& m, const std::vector<Batch>& batches) {
#ifdef TFI_USE_ACCELERATE
    return detail::gpu_for_work(m, gpu_work(m, batches));
#else
    (void)m;
    (void)batches;
    return nullptr;
#endif
}

bool same_floats(std::span<const float> a, const float* b) { return std::memcmp(a.data(), b, a.size_bytes()) == 0; }

// Exact patching of every site, or with `selected` only those sites: the others are left NaN
// (except positions after `pos`, whose effect is exactly 0 without running anything).
PatchResult exact_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                                 int64_t pos, int64_t target, int64_t other, const std::vector<Site>* selected,
                                 const PatchRuns* runs = nullptr) {
    static const std::vector<std::string> sites = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"};
    if (std::ranges::find(sites, site) == sites.end())
        throw std::invalid_argument(std::format("unknown patch site '{}'", site));
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    const int64_t L = m.cfg().n_layers, T = std::ssize(clean), d = m.cfg().d_model, P = pos + 1;
    auto hook_name = [&](int64_t i) { return std::format("{}hook_{}", lp(i), site); };

    std::set<std::string> needed, corrupt_needed;
    for (int64_t i = 0; i < L; ++i) {
        needed.insert(hook_name(i));
        corrupt_needed.insert({hook_name(i), lp(i) + "attn.hook_k", lp(i) + "attn.hook_v"});
    }
    PatchRuns own;
    if (!runs) own = prepare_patch(m, clean, corrupt, pos, target, other, needed, corrupt_needed);
    const PatchRuns& p = runs ? *runs : own;
    PatchResult r = p.result;
    r.effect = Tensor({L, T}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back(std::format("L{}", i));
    for (int64_t t = 0; t < T; ++t) r.col_labels.push_back(std::to_string(t));

    // Sites to run, per layer. A layer without the site (e.g. mlp_out on an attention-only layer) stays NaN.
    std::vector<const Tensor*> src(L, nullptr), start(L), prefix_k(L), prefix_v(L);
    std::vector<std::vector<int64_t>> todo(L);
    auto consider = [&](int64_t i, int64_t t) {
        const Tensor& cor = p.corrupt_cache.at(hook_name(i));
        if (same_floats(src[i]->row_view(t), cor.ptr() + t * d) || (i == L - 1 && site != "resid_pre" && t < pos))
            r.effect.data[i * T + t] = 0.f;
        else todo[i].push_back(t);
    };
    for (int64_t i = 0; i < L; ++i) {
        start[i] = &p.corrupt_cache.at(lp(i) + "hook_resid_pre");
        prefix_k[i] = &p.corrupt_cache.at(lp(i) + "attn.hook_k");
        prefix_v[i] = &p.corrupt_cache.at(lp(i) + "attn.hook_v");
        auto it = p.clean_cache.find(hook_name(i));
        if (it == p.clean_cache.end()) continue;
        src[i] = &it->second;
        for (int64_t t = P; t < T; ++t) r.effect.data[i * T + t] = 0.f;  // cannot reach `pos`
        if (!selected)
            for (int64_t t = 0; t < P; ++t) consider(i, t);
    }
    if (selected)
        for (const Site& s : *selected)
            if (s.first >= 0 && s.first < L && src[s.first] && s.second >= 0 && s.second < P)
                consider(s.first, s.second);

    const std::vector<Batch> batches = make_batches(todo, [P](int64_t t) { return P - t; }, batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = m.weight_format() == WeightFormat::fp16;  // fp16 weights: the GPU's multiplies take fp16 inputs
    parallel_for(std::ssize(batches), [&](int64_t j) {
        const Batch& b = batches[j];
        const int64_t i = b.layer;
        detail::RowBatch rb;
        rb.prefix_k.assign(prefix_k.begin(), prefix_k.end());
        rb.prefix_v.assign(prefix_v.begin(), prefix_v.end());
        rb.gpu = gpu;
        rb.fp16 = fp16;
        for (int64_t t : b.items) rb.add(t, P - t);
        Tensor x({rb.rows(), d});  // the corrupted residual stream, positions t..pos of each pass
        for (size_t s = 0; s < b.items.size(); ++s)
            std::copy(start[i]->ptr() + b.items[s] * d, start[i]->ptr() + P * d, x.ptr() + rb.offset[s] * d);
        const Tensor* sp = src[i];
        Hooks hooks;
        hooks.add(hook_name(i), [&](const std::string&, Tensor& act) {  // patch each pass's first row
            for (size_t s = 0; s < b.items.size(); ++s) {
                const int64_t t = b.items[s];
                std::copy(sp->ptr() + t * d, sp->ptr() + (t + 1) * d, act.ptr() + rb.offset[s] * d);
            }
        });
        const Tensor logits = m.forward_rows(i, std::move(x), rb, &hooks);
        for (size_t s = 0; s < b.items.size(); ++s)
            r.effect.data[i * T + b.items[s]] =
                logit_metric(logits, static_cast<int64_t>(s), target, other) - r.corrupt_metric;
    });
    normalize_patch(r);
    return r;
}

PatchResult exact_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                              int64_t other, const std::vector<Site>* selected, const PatchRuns* runs = nullptr) {
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    const int64_t L = m.cfg().n_layers, H = m.max_heads(), d = m.cfg().d_model, P = pos + 1;
    std::set<std::string> needed;
    for (int64_t i = 0; i < L; ++i) needed.insert(lp(i) + "attn.hook_z");
    PatchRuns own;
    if (!runs) own = prepare_patch(m, clean, corrupt, pos, target, other, needed, needed);
    const PatchRuns& p = runs ? *runs : own;
    PatchResult r = p.result;
    r.effect = Tensor({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back(std::format("L{}", i));
    for (int64_t h = 0; h < H; ++h) r.col_labels.push_back(std::format("H{}", h));

    std::vector<const Tensor*> src(L), start(L);
    std::vector<std::vector<int64_t>> todo(L);
    auto consider = [&](int64_t i, int64_t h) {  // run unless the head's output is the same in both runs
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        const Tensor& cor = p.corrupt_cache.at(lp(i) + "attn.hook_z");
        bool same = true;
        for (int64_t t = 0; t < P && same; ++t)
            same = same_floats(std::span(src[i]->data).subspan((t * Hl + h) * D, D), cor.ptr() + (t * Hl + h) * D);
        if (same) r.effect.data[i * H + h] = 0.f;
        else todo[i].push_back(h);
    };
    for (int64_t i = 0; i < L; ++i) {
        src[i] = &p.clean_cache.at(lp(i) + "attn.hook_z");  // (pos+1, H, D)
        start[i] = &p.corrupt_cache.at(lp(i) + "hook_resid_pre");
        if (!selected)
            for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) consider(i, h);
    }
    if (selected)
        for (const Site& s : *selected)
            if (s.first >= 0 && s.first < L && s.second >= 0 && s.second < m.layer_cfg(s.first).n_heads)
                consider(s.first, s.second);

    const std::vector<Batch> batches = make_batches(todo, [P](int64_t) { return P; }, head_batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = m.weight_format() == WeightFormat::fp16;  // fp16 weights: the GPU's multiplies take fp16 inputs
    parallel_for(std::ssize(batches), [&](int64_t j) {
        const Batch& b = batches[j];
        const int64_t i = b.layer, Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        detail::RowBatch rb;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        for (size_t s = 0; s < b.items.size(); ++s) rb.add(0, P);
        Tensor x({rb.rows(), d});
        for (size_t s = 0; s < b.items.size(); ++s) std::copy_n(start[i]->ptr(), P * d, x.ptr() + rb.offset[s] * d);
        const Tensor* sp = src[i];
        Hooks hooks;
        hooks.add(lp(i) + "attn.hook_z", [&](const std::string&, Tensor& act) {  // each pass's head, every position
            for (size_t s = 0; s < b.items.size(); ++s) {
                const int64_t h = b.items[s], off = rb.offset[s];
                for (int64_t t = 0; t < P; ++t)
                    std::copy(sp->ptr() + (t * Hl + h) * D, sp->ptr() + (t * Hl + h + 1) * D,
                              act.ptr() + ((off + t) * Hl + h) * D);
            }
        });
        const Tensor logits = m.forward_rows(i, std::move(x), rb, &hooks);
        for (size_t s = 0; s < b.items.size(); ++s)
            r.effect.data[i * H + b.items[s]] =
                logit_metric(logits, static_cast<int64_t>(s), target, other) - r.corrupt_metric;
    });
    normalize_patch(r);
    return r;
}

}  // namespace

void detail::set_patch_batch_rows(int64_t residual, int64_t heads) {
    g_batch_rows = residual;
    g_head_batch_rows = heads;
}

PatchResult patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                           int64_t pos, int64_t target, int64_t other) {
    return exact_patch_residual(m, clean, corrupt, site, pos, target, other, nullptr);
}

PatchResult patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other) {
    return exact_patch_heads(m, clean, corrupt, pos, target, other, nullptr);
}

// --- attribution patching ----------------------------------------------------------------

namespace {

// dF/dlogits for F = logit_metric(pos, target, other).
Tensor metric_dlogits(const Tensor& logits, int64_t pos, int64_t target, int64_t other) {
    const int64_t V = logits.shape[1];
    Tensor dl(logits.shape, 0.f);
    float* row = dl.ptr() + pos * V;
    if (other >= 0) {
        row[target] += 1.f;
        row[other] -= 1.f;
    } else {  // d log_softmax(l)[target] / dl = onehot(target) - softmax(l)
        std::copy(logits.ptr() + pos * V, logits.ptr() + (pos + 1) * V, row);
        softmax_(std::span(row, V));
        for (int64_t j = 0; j < V; ++j) row[j] = -row[j];
        row[target] += 1.f;
    }
    return dl;
}

}  // namespace

float metric_gradients(const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t other,
                       ActivationCache& grads, NameFilter filter, ActivationCache* acts) {
    check_pos(tokens, pos);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");
    float metric = 0;
    m.run_with_gradients(
        tokens,
        [&](const Tensor& logits) {
            metric = logit_metric(logits, pos, target, other);
            return metric_dlogits(logits, pos, target, other);
        },
        grads, std::move(filter), acts);
    return metric;
}

namespace {

// Clean activations, corrupted activations and the metric's gradient at `names`, on the inputs cut
// to positions 0..pos (later positions cannot affect the metric). With steps = 1 the gradient is
// the corrupted run's; with more, it is averaged over runs whose token embeddings are interpolated
// from corrupted to clean, at alpha = 0, 1/steps, ..., (steps-1)/steps. All the runs are
// independent, so they run in parallel. `clean_extra` and `corrupt_extra` name more activations to
// cache, so that mixed patching can reuse the runs for exact patching.
struct AttributionRuns {
    PatchRuns runs;  // the clean and corrupted caches and metrics, as prepare_patch gives them
    ActivationCache grads;
};

AttributionRuns prepare_attribution(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other, const std::set<std::string>& names, int64_t steps,
                                    std::set<std::string> clean_extra = {}, std::set<std::string> corrupt_extra = {}) {
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    if (steps < 1) throw std::invalid_argument("attribution patching needs at least one step");
    AttributionRuns a;
    const Tokens c(clean.begin(), clean.begin() + pos + 1), x(corrupt.begin(), corrupt.begin() + pos + 1);
    clean_extra.insert(names.begin(), names.end());
    corrupt_extra.insert(names.begin(), names.end());
    // Token embeddings (hook_embed) of both inputs, for the interpolated runs.
    Tensor e_clean, e_corrupt;
    if (steps > 1) {
        Hooks grab;
        grab.add_capture([](const std::string& n) { return n == "hook_embed"; },
                         [&](const std::string&, Tensor& e) { (e_clean.numel() ? e_corrupt : e_clean) = e; });
        m.embed(c, &grab);
        m.embed(x, &grab);
    }
    // Step k's run: the corrupted input with its token embeddings moved k / steps of the way to the
    // clean ones (step 0 is the corrupted run itself).
    auto interpolate = [&](int64_t k) {
        Hooks hooks;
        if (k == 0) return hooks;
        const float alpha = static_cast<float>(k) / static_cast<float>(steps);
        hooks.add("hook_embed", [&, alpha](const std::string&, Tensor& e) {
            for (int64_t i = 0; i < e.numel(); ++i)
                e.data[i] = e_corrupt.data[i] + alpha * (e_clean.data[i] - e_corrupt.data[i]);
        });
        return hooks;
    };
    // On the GPU (macOS), the steps' gradient runs go in batches of stacked sequences
    // (Model::gradients_rows); batches too small for the GPU, and other systems, run one sequence
    // at a time on the CPU. The batches depend only on the number of steps and the length.
    const int64_t P = pos + 1;
    std::vector<int64_t> all_steps(steps);
    for (int64_t k = 0; k < steps; ++k) all_steps[k] = k;
    std::vector<Batch> gpu_batches;
    std::vector<int64_t> cpu_steps;
    detail::Gpu* gpu = nullptr;
#ifdef TFI_USE_ACCELERATE
    if (detail::gradient_runs_use_gpu(m, steps * P))
        for (Batch& b : make_batches({all_steps}, [P](int64_t) { return P; }, gradient_batch_rows(steps, P))) {
            if (b.rows >= detail::min_gradient_batch_rows() && (gpu || (gpu = m.gpu())))
                gpu_batches.push_back(std::move(b));
            else cpu_steps.insert(cpu_steps.end(), b.items.begin(), b.items.end());
        }
    else cpu_steps = all_steps;
#else
    cpu_steps = all_steps;
#endif
    const bool corrupt_on_cpu = std::ranges::find(cpu_steps, 0) != cpu_steps.end();
    // The corrupted run's activations come from a forward pass like the clean run's, so that what the
    // corruption can't reach is the same in both, bit for bit: so from step 0's gradient pass only where its
    // forward half is that pass (on the CPU). A gradient pass on the GPU multiplies in float32, a forward
    // pass with fp16 weights there doesn't.
    const bool corrupt_from_gradients = corrupt_on_cpu && !detail::gradients_go_to_gpu(m, P);
    std::vector<ActivationCache> step_grads(steps);
    // Jobs: the clean run, the corrupted run (with step 0's gradients when that is on the CPU), the
    // GPU batches one after another (they share the GPU, so running them side by side saves no time, and each
    // holds every layer's activations: M5 Pro, 0.9B model, 4 steps of 2048 tokens took 8.7 s and 37 GB side by
    // side, 7.7 s and 14.5 GB in turn), then the other CPU steps.
    const int64_t n_cpu_jobs = std::ssize(cpu_steps) - (corrupt_on_cpu ? 1 : 0);
    const int64_t n_gpu = std::ssize(gpu_batches);
    std::vector<int64_t> later_cpu;
    for (int64_t k : cpu_steps)
        if (k != 0) later_cpu.push_back(k);
    const int64_t gpu_job = n_gpu ? 1 : 0;
    parallel_for(2 + gpu_job + n_cpu_jobs, [&](int64_t j) {
        if (j == 0) {
            a.runs.result.clean_metric =
                logit_metric(m.run_with_cache(c, a.runs.clean_cache, only(clean_extra)), pos, target, other);
        } else if (j == 1) {
            if (corrupt_from_gradients) {
                ActivationCache all;
                a.runs.result.corrupt_metric =
                    metric_gradients(m, x, pos, target, other, step_grads[0], only(names), &all);
                for (const auto& n : corrupt_extra)
                    if (all.contains(n)) a.runs.corrupt_cache[n] = std::move(all[n]);
            } else {
                a.runs.result.corrupt_metric =
                    logit_metric(m.run_with_cache(x, a.runs.corrupt_cache, only(corrupt_extra)), pos, target, other);
                if (corrupt_on_cpu) metric_gradients(m, x, pos, target, other, step_grads[0], only(names));
            }
        } else if (j < 2 + gpu_job) {
            for (const Batch& b : gpu_batches) {
                detail::RowBatch rb;
                rb.gpu = gpu;
                Tensor x0({b.rows, m.cfg().d_model});
                for (int64_t k : b.items) {
                    const Hooks h = interpolate(k);
                    const Tensor e = m.embed(x, &h);
                    std::copy_n(e.ptr(), e.numel(), x0.ptr() + rb.rows() * m.cfg().d_model);
                    rb.add(0, P);
                }
                std::vector<ActivationCache> g = m.gradients_rows(
                    x0, rb, [&](int64_t, const Tensor& logits) { return metric_dlogits(logits, 0, target, other); },
                    only(names));
                for (size_t i = 0; i < b.items.size(); ++i) step_grads[b.items[i]] = std::move(g[i]);
            }
        } else {
            const int64_t k = later_cpu[j - 2 - gpu_job];
            const Hooks hooks = interpolate(k);
            m.run_with_gradients(
                x, [&](const Tensor& logits) { return metric_dlogits(logits, pos, target, other); }, step_grads[k],
                only(names), nullptr, &hooks);
        }
    });
    a.grads = std::move(step_grads[0]);
    if (steps > 1)
        for (auto& [name, t] : a.grads) {  // average, summing in step order: deterministic
            for (int64_t k = 1; k < steps; ++k) add_(t, step_grads[k].at(name));
            t = scale(t, 1.f / static_cast<float>(steps));
        }
    return a;
}

std::set<std::string> residual_names(const Model& m, const std::string& site) {
    static const std::vector<std::string> sites = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"};
    if (std::ranges::find(sites, site) == sites.end())
        throw std::invalid_argument(std::format("unknown patch site '{}'", site));
    std::set<std::string> names;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) names.insert(std::format("{}hook_{}", lp(i), site));
    return names;
}

std::set<std::string> head_names(const Model& m) {
    std::set<std::string> names;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) names.insert(lp(i) + "attn.hook_z");
    return names;
}

// What exact patching caches beyond the patched activations: every layer's input (and for residual
// sites, attention's keys and values) in the corrupted run.
std::set<std::string> exact_corrupt_extra(const Model& m, bool residual) {
    std::set<std::string> names;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) {
        names.insert(lp(i) + "hook_resid_pre");
        if (residual) names.insert({lp(i) + "attn.hook_k", lp(i) + "attn.hook_v"});
    }
    return names;
}

PatchResult attribution_residual_result(const Model& m, const AttributionRuns& a, const std::string& site, int64_t pos,
                                        int64_t T) {
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model;
    PatchResult r = a.runs.result;
    r.effect = Tensor({L, T}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back(std::format("L{}", i));
    for (int64_t t = 0; t < T; ++t) r.col_labels.push_back(std::to_string(t));
    for (int64_t i = 0; i < L; ++i) {
        const std::string n = std::format("{}hook_{}", lp(i), site);
        if (!a.runs.clean_cache.contains(n)) continue;  // site doesn't exist in this layer: stays NaN
        const Tensor &cl = a.runs.clean_cache.at(n), &co = a.runs.corrupt_cache.at(n), &g = a.grads.at(n);
        for (int64_t t = 0; t < T; ++t) {
            double s = 0;
            if (t <= pos)
                for (int64_t j = 0; j < d; ++j)
                    s += double(cl.data[t * d + j] - co.data[t * d + j]) * g.data[t * d + j];
            r.effect.data[i * T + t] = static_cast<float>(s);
        }
    }
    normalize_patch(r);
    return r;
}

PatchResult attribution_heads_result(const Model& m, const AttributionRuns& a) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    PatchResult r = a.runs.result;
    r.effect = Tensor({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back(std::format("L{}", i));
    for (int64_t h = 0; h < H; ++h) r.col_labels.push_back(std::format("H{}", h));
    for (int64_t i = 0; i < L; ++i) {
        const std::string n = lp(i) + "attn.hook_z";  // (pos+1, H, D)
        const Tensor &cl = a.runs.clean_cache.at(n), &co = a.runs.corrupt_cache.at(n), &g = a.grads.at(n);
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim(), T = cl.shape[0];
        for (int64_t h = 0; h < Hl; ++h) {
            double s = 0;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t j = 0; j < D; ++j) {
                    const int64_t idx = (t * Hl + h) * D + j;
                    s += double(cl.data[idx] - co.data[idx]) * g.data[idx];
                }
            r.effect.data[i * H + h] = static_cast<float>(s);
        }
    }
    normalize_patch(r);
    return r;
}

}  // namespace

PatchResult attribution_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt,
                                       const std::string& site, int64_t pos, int64_t target, int64_t other,
                                       int64_t steps) {
    const std::set<std::string> names = residual_names(m, site);
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, names, steps);
    return attribution_residual_result(m, a, site, pos, std::ssize(clean));
}

PatchResult attribution_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other, int64_t steps) {
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, head_names(m), steps);
    return attribution_heads_result(m, a);
}

// --- mixed patching ---------------------------------------------------------------------------

namespace {

// Rows x cols site of each of the `verify` largest finite |estimates|, ties broken by position in the
// table so the choice is deterministic. `usable(row, col)` excludes sites that need no measurement.
template <std::predicate<int64_t, int64_t> Usable>
std::vector<Site> top_sites(const Tensor& estimate, int64_t verify, Usable usable) {
    if (verify < 0) throw std::invalid_argument("mixed patching: verify must be >= 0");
    const int64_t cols = estimate.shape[1];
    std::vector<int64_t> idx;
    for (int64_t i = 0; i < estimate.numel(); ++i)
        if (!std::isnan(estimate.data[i]) && usable(i / cols, i % cols)) idx.push_back(i);
    std::ranges::stable_sort(idx, std::ranges::greater{}, [&](int64_t i) { return std::fabs(estimate.data[i]); });
    if (std::ssize(idx) > verify) idx.resize(static_cast<size_t>(verify));
    std::vector<Site> sites;
    for (int64_t i : idx) sites.emplace_back(i / cols, i % cols);
    return sites;
}

// Start from the estimates and overwrite the measured sites.
PatchResult merge_mixed(PatchResult atp, const PatchResult& exact, const std::vector<Site>& sites) {
    const int64_t cols = atp.effect.shape[1];
    atp.estimate = atp.effect;
    atp.exact.assign(static_cast<size_t>(atp.effect.numel()), 0);
    for (const Site& s : sites) {
        const int64_t i = s.first * cols + s.second;
        atp.effect.data[i] = exact.effect.data[i];
        atp.exact[i] = 1;
    }
    normalize_patch(atp);
    return atp;
}

}  // namespace

// Mixed patching reuses the attribution's clean and corrupted runs for exact patching (they are the
// same runs prepare_patch would do, so the exact values are the same bit for bit).
PatchResult mixed_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                                 int64_t pos, int64_t target, int64_t other, int64_t verify, int64_t steps) {
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, residual_names(m, site), steps,
                                                  {}, exact_corrupt_extra(m, true));
    PatchResult atp = attribution_residual_result(m, a, site, pos, std::ssize(clean));
    const std::vector<Site> sites = top_sites(atp.effect, verify, [pos](int64_t, int64_t t) { return t <= pos; });
    const PatchResult exact = exact_patch_residual(m, clean, corrupt, site, pos, target, other, &sites, &a.runs);
    return merge_mixed(std::move(atp), exact, sites);
}

PatchResult mixed_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                              int64_t other, int64_t verify, int64_t steps) {
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, head_names(m), steps, {},
                                                  exact_corrupt_extra(m, false));
    PatchResult atp = attribution_heads_result(m, a);
    const std::vector<Site> sites = top_sites(atp.effect, verify, [](int64_t, int64_t) { return true; });
    const PatchResult exact = exact_patch_heads(m, clean, corrupt, pos, target, other, &sites, &a.runs);
    return merge_mixed(std::move(atp), exact, sites);
}

// --- ablation -----------------------------------------------------------------------------

// Reference sequences run like patching passes (see exact patching above): stacked into batches
// of at most head_batch_rows() rows, on the GPU on macOS when large enough, batches in parallel.
// The batches depend only on the references, and the sums run in reference order, so the means are
// the same for any thread count.
HeadMeans head_means(const Model& m, const std::vector<Tokens>& reference, int64_t T) {
    if (reference.empty()) throw std::invalid_argument("mean ablation needs at least one reference sequence");
    const int64_t L = m.cfg().n_layers, R = std::ssize(reference);
    const bool per_position = std::ranges::all_of(reference, [T](const Tokens& r) { return std::ssize(r) == T; });
    HeadMeans means;
    std::vector<double> counts(L, 0.0);
    for (int64_t i = 0; i < L; ++i)
        means.layers.emplace_back(Shape{per_position ? T : 1, m.layer_cfg(i).n_heads, m.layer_cfg(i).head_dim()});
    std::vector<int64_t> all(R);
    for (int64_t n = 0; n < R; ++n) all[n] = n;
    const std::vector<Batch> batches =
        make_batches({all}, [&](int64_t n) { return std::ssize(reference[n]); }, ablation_batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = m.weight_format() == WeightFormat::fp16;  // fp16 weights: the GPU's multiplies take fp16 inputs
    std::vector<std::vector<Tensor>> zs(R, std::vector<Tensor>(L));  // reference n, layer i: (len, H, D)
    parallel_for(std::ssize(batches), [&](int64_t j) {
        const Batch& b = batches[j];
        detail::RowBatch rb;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        std::vector<Tensor> inputs;
        for (int64_t n : b.items) {
            inputs.push_back(m.embed(reference[n]));
            rb.add(0, std::ssize(reference[n]));
        }
        Tensor x({rb.rows(), m.cfg().d_model});
        for (size_t s = 0; s < inputs.size(); ++s)
            std::ranges::copy(inputs[s].data, x.ptr() + rb.offset[s] * m.cfg().d_model);
        Hooks hooks;  // captures: they only read, so a GPU pass need not stop for them
        for (int64_t i = 0; i < L; ++i) {
            auto keep_z = [&, i](const std::string&, Tensor& z) {  // (rows, H, D)
                const int64_t row = z.numel() / z.shape[0];
                for (size_t s = 0; s < b.items.size(); ++s) {
                    Tensor& dst = zs[b.items[s]][i];
                    dst = Tensor({rb.len[s], z.shape[1], z.shape[2]});
                    std::copy(z.ptr() + rb.offset[s] * row, z.ptr() + (rb.offset[s] + rb.len[s]) * row, dst.ptr());
                }
            };
            hooks.add_capture([name = lp(i) + "attn.hook_z"](const std::string& n) { return n == name; }, keep_z);
        }
        m.forward_rows(0, std::move(x), rb, &hooks);
    });
    for (int64_t n = 0; n < R; ++n) {  // reduce in reference order: deterministic sums
        for (int64_t i = 0; i < L; ++i) {
            const Tensor& z = zs[n][i];
            Tensor& acc = means.layers[i];
            const int64_t row = acc.numel() / acc.shape[0];
            for (int64_t t = 0; t < z.shape[0]; ++t) {
                float* dst = acc.ptr() + (per_position ? t : 0) * row;
                for (int64_t j = 0; j < row; ++j) dst[j] += z.data[t * row + j];
            }
            counts[i] += per_position ? 1.0 : static_cast<double>(z.shape[0]);
        }
    }
    for (int64_t i = 0; i < L; ++i)
        for (float& v : means.layers[i].data) v = static_cast<float>(v / counts[i]);
    return means;
}

Hooks ablate_head(const Model& m, int64_t layer, int64_t head, Ablation mode, const HeadMeans* means) {
    if (layer < 0 || layer >= m.cfg().n_layers) throw std::out_of_range("layer index out of range");
    const int64_t Hl = m.layer_cfg(layer).n_heads, D = m.layer_cfg(layer).head_dim();
    if (head < 0 || head >= Hl) throw std::out_of_range("head index out of range");
    const Tensor* mean = nullptr;
    if (mode == Ablation::Mean) {
        if (!means || std::ssize(means->layers) != m.cfg().n_layers)
            throw std::invalid_argument("mean ablation needs reference means (see head_means)");
        mean = &means->layers[layer];
    }
    Hooks hooks;
    hooks.add(lp(layer) + "attn.hook_z", [head, Hl, D, mean](const std::string&, Tensor& z) {
        const int64_t T = z.shape[0];
        if (mean && mean->shape[0] != 1 && mean->shape[0] != T)
            throw std::invalid_argument(std::format(
                "per-position reference means were computed for length {}, input has length {}", mean->shape[0], T));
        for (int64_t t = 0; t < T; ++t) {
            float* dst = z.ptr() + (t * Hl + head) * D;
            if (!mean) std::fill_n(dst, D, 0.f);
            else {
                const float* src = mean->ptr() + ((mean->shape[0] == 1 ? 0 : t) * Hl + head) * D;
                std::copy_n(src, D, dst);
            }
        }
    });
    return hooks;
}

// Ablation passes run like head patching passes (see exact patching above): a head ablated in layer
// i starts at layer i from the cached residual stream, passes starting at the same layer are
// stacked into one batch, and the batches run in parallel. A head whose output already equals its
// replacement at every position (zeros, or its mean) changes nothing: exactly 0, without a pass.
Tensor head_ablation(const Model& m, const Tokens& tokens, Ablation mode, const std::vector<Tokens>& reference) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads(), T = std::ssize(tokens);
    const int64_t d = m.cfg().d_model, V = m.cfg().vocab_size;
    if (T < 2) throw std::invalid_argument("loss needs at least two tokens");
#ifdef TFI_USE_ACCELERATE
    if (mode == Ablation::Mean && use_gpu()) {
        // The reference passes and the ablation passes each choose the GPU on their own work. If the
        // two together are worth it, start it now: its start-up is paid once, and then both use it.
        std::vector<std::vector<int64_t>> heads(L);
        for (int64_t i = 0; i < L; ++i)
            for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) heads[i].push_back(h);
        std::vector<int64_t> refs(reference.size());
        for (size_t n = 0; n < refs.size(); ++n) refs[n] = static_cast<int64_t>(n);
        const double work =
            gpu_work(m, make_batches(
                            heads, [T](int64_t) { return T; }, ablation_batch_rows())) +
            gpu_work(m,
                     make_batches({refs}, [&](int64_t n) { return std::ssize(reference[n]); }, ablation_batch_rows()));
        if (work >= detail::gpu_min_work()) m.gpu();
    }
#endif
    const HeadMeans means = mode == Ablation::Mean ? head_means(m, reference, T) : HeadMeans{};
    std::set<std::string> needed;
    for (int64_t i = 0; i < L; ++i) {
        needed.insert(lp(i) + "hook_resid_pre");
        needed.insert(lp(i) + "attn.hook_z");
    }
    ActivationCache cache;
    const float base = next_token_loss(m.run_with_cache(tokens, cache, only(std::move(needed))), tokens);
    Tensor out({L, H}, kNaN);

    // The replacement for head h of layer i at position t (D floats).
    int64_t max_d = 0;
    for (int64_t i = 0; i < L; ++i) max_d = std::max(max_d, m.layer_cfg(i).head_dim());
    const std::vector<float> zeros(static_cast<size_t>(max_d), 0.f);
    auto replacement = [&](int64_t i, int64_t h, int64_t t) -> const float* {
        if (mode == Ablation::Zero) return zeros.data();
        const Tensor& mean = means.layers[i];
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        return mean.ptr() + ((mean.shape[0] == 1 ? 0 : t) * Hl + h) * D;
    };
    std::vector<std::vector<int64_t>> todo(L);
    std::vector<const Tensor*> start(L);
    for (int64_t i = 0; i < L; ++i) {
        start[i] = &cache.at(lp(i) + "hook_resid_pre");
        const Tensor& z = cache.at(lp(i) + "attn.hook_z");
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        for (int64_t h = 0; h < Hl; ++h) {
            bool same = true;
            for (int64_t t = 0; t < T && same; ++t)
                same = same_floats(std::span(z.data).subspan((t * Hl + h) * D, D), replacement(i, h, t));
            if (same) out.data[i * H + h] = 0.f;
            else todo[i].push_back(h);
        }
    }

    const std::vector<Batch> batches = make_batches(todo, [T](int64_t) { return T; }, ablation_batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = m.weight_format() == WeightFormat::fp16;  // fp16 weights: the GPU's multiplies take fp16 inputs
    // Logits are unembedded a few passes at a time: at most this many floats at once.
    constexpr int64_t kMaxLogits = int64_t(1) << 23;
    parallel_for(std::ssize(batches), [&](int64_t j) {
        const Batch& b = batches[j];
        const int64_t i = b.layer, Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        const int64_t S = std::ssize(b.items);
        detail::RowBatch rb;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        for (int64_t s = 0; s < S; ++s) rb.add(0, T);
        Tensor x({rb.rows(), d});
        for (int64_t s = 0; s < S; ++s) std::copy_n(start[i]->ptr(), T * d, x.ptr() + rb.offset[s] * d);
        Hooks hooks;
        hooks.add(lp(i) + "attn.hook_z", [&](const std::string&, Tensor& z) {  // each pass's head, every position
            for (int64_t s = 0; s < S; ++s) {
                const int64_t h = b.items[s];
                for (int64_t t = 0; t < T; ++t) {
                    const float* src = replacement(i, h, t);
                    std::copy_n(src, D, z.ptr() + ((rb.offset[s] + t) * Hl + h) * D);
                }
            }
        });
        const Tensor fin = m.forward_rows_final(i, std::move(x), rb, &hooks);
        // Next-token loss of each pass: rows 0..T-2 predict tokens 1..T-1.
        const int64_t per = std::max<int64_t>(1, kMaxLogits / ((T - 1) * V));
        for (int64_t s0 = 0; s0 < S; s0 += per) {
            const int64_t n = std::min(per, S - s0);
            Tensor rows_in({n * (T - 1), d});
            for (int64_t s = 0; s < n; ++s)
                std::copy(fin.ptr() + rb.offset[s0 + s] * d, fin.ptr() + (rb.offset[s0 + s] + T - 1) * d,
                          rows_in.ptr() + s * (T - 1) * d);
            const Tensor logits = m.logits_from_final(rows_in, &rb);
            for (int64_t s = 0; s < n; ++s) {
                double loss = 0;
                for (int64_t t = 0; t + 1 < T; ++t)
                    loss -= log_softmax_at(logits.row_view(s * (T - 1) + t), tokens[t + 1]);
                out.data[i * H + b.items[s0 + s]] = static_cast<float>(loss / (T - 1)) - base;
            }
        }
    });
    return out;
}

// --- weight-based circuits -------------------------------------------------------------------

HeadWeights head_weights(const Model& m, int64_t layer, int64_t head) {
    const ModelConfig& c = m.layer_cfg(layer);
    const int64_t H = c.n_heads, D = c.head_dim(), rep = H / c.kv_heads(), g = head / rep;
    if (head < 0 || head >= H) throw std::out_of_range("head index out of range");
    const std::string p = lp(layer) + "attn.";
    HeadWeights w;
    w.W_Q = m.weight(p + "q_proj.weight").rows(head * D, (head + 1) * D);
    w.W_K = m.weight(p + "k_proj.weight").rows(g * D, (g + 1) * D);
    w.W_V = m.weight(p + "v_proj.weight").rows(g * D, (g + 1) * D);
    w.W_O = m.weight(p + "o_proj.weight").cols(head * D, (head + 1) * D);
    // The attention input is gamma * normalized(x) for pre/sandwich norm; fold gamma in.
    if (c.norm != "none" && c.norm_position != "post") {
        const Tensor& gamma = m.weight(lp(layer) + "ln1.weight").f32();
        for (Tensor* W : {&w.W_Q, &w.W_K, &w.W_V})
            for (int64_t r = 0; r < D; ++r)
                for (int64_t j = 0; j < c.d_model; ++j) W->data[r * c.d_model + j] *= gamma.data[j];
    }
    return w;
}

Tensor ov_matrix(const HeadWeights& h) { return matmul(h.W_O, h.W_V); }
Tensor qk_matrix(const HeadWeights& h) { return matmul(transpose2d(h.W_Q), h.W_K); }

namespace {
std::atomic<int64_t> g_circuits_run_bytes{0};
}  // namespace

void detail::set_circuits_run_bytes(int64_t bytes) { g_circuits_run_bytes = bytes; }

#ifdef TFI_USE_ACCELERATE
namespace {

int64_t circuits_run_bytes() {
    const int64_t b = g_circuits_run_bytes.load();
    return b > 0 ? b : int64_t(512) << 20;
}


// How a GPU circuits computation went: too little work (the CPU does it), done, or a value too
// large for fp16 inputs (it runs again in float32).
enum class GpuRun { NotWorth, Done, Overflow };

// Writes n float32 values into a batch buffer at `offset`: as they are, or into an fp16 buffer
// rounded, counting into `bad` the values that don't fit fp16.
void put(detail::Gpu::Batch& batch, int buf, int64_t offset, const float* src, int64_t n, std::atomic<int64_t>& bad) {
    if (batch.is_fp16(buf)) bad += detail::floats_to_halves(src, batch.halves(buf) + offset, n);
    else std::copy_n(src, n, batch.floats(buf) + offset);
}

// F with F F^T = G, for a symmetric positive semi-definite (r, r) G: its eigenvectors times the
// square roots of its eigenvalues (rounding's tiny negative ones clamped to 0).
Tensor gram_factor(const Tensor& G) {
    const int64_t r = G.shape[0];
    std::vector<float> a(G.data), w(static_cast<size_t>(r));  // symmetric: row-major = column-major
    __LAPACK_int n = static_cast<__LAPACK_int>(r), lda = n, info = 0, lwork = -1, liwork = -1, iwork_size = 0;
    float work_size = 0;
    const char jobz = 'V', uplo = 'U';
    ssyevd_(&jobz, &uplo, &n, a.data(), &lda, w.data(), &work_size, &lwork, &iwork_size, &liwork, &info);
    lwork = static_cast<__LAPACK_int>(work_size);
    liwork = iwork_size;
    std::vector<float> work(static_cast<size_t>(std::max<__LAPACK_int>(lwork, 1)));
    std::vector<__LAPACK_int> iwork(static_cast<size_t>(std::max<__LAPACK_int>(liwork, 1)));
    ssyevd_(&jobz, &uplo, &n, a.data(), &lda, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0) throw std::runtime_error("eigen-decomposition of a head's Gram matrix failed");
    Tensor F({r, r});
    for (int64_t j = 0; j < r; ++j) {  // column j of a (column-major) is eigenvector j
        const float s = std::sqrt(std::max(w[j], 0.f));
        for (int64_t i = 0; i < r; ++i) F.data[i * r + j] = a[j * r + i] * s;
    }
    return F;
}

// composition_scores' pairs on the GPU. A pair's score is sqrt(tr(C^T G_L C G_R)) over a norm, for
// its block C of W_reader W_O,writer (see compose). With F F^T = G for each Gram matrix that's the
// Frobenius norm of (F_L^T W_reader) (W_O,writer F_R): the factors fold into the weights once per
// head, one multiply per pair of layers gives every head pair's block, and the GPU sums each block's
// squares. With fp16 the factored weights go to the GPU as fp16 (its fast multiplies), the sums
// staying float32.
GpuRun composition_gpu(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& ids,
                       const std::vector<HeadCache>& heads, Composition& c, bool fp16) {
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model, N = std::ssize(ids);
    auto H = [&](int64_t l) { return m.layer_cfg(l).n_heads; };
    auto D = [&](int64_t l) { return m.layer_cfg(l).head_dim(); };
    std::vector<int64_t> first(L + 1, 0);  // layer l's heads are ids[first[l] .. first[l + 1])
    for (int64_t l = 0; l < L; ++l) first[l + 1] = first[l] + H(l);
    double work = 0;
    int64_t scratch = 0;
    for (int64_t lb = 1; lb < L; ++lb)
        for (int64_t la = 0; la < lb; ++la) {
            work += 3.0 * double(H(lb) * D(lb)) * double(d) * double(H(la) * D(la));
            scratch = std::max(scratch, 3 * H(lb) * D(lb) * H(la) * D(la));
        }
    detail::Gpu* gpu = detail::gpu_for_work(m, work);
    if (!gpu) return GpuRun::NotWorth;
    detail::Gpu::Batch batch(*gpu);
    using Mat = detail::Gpu::Batch::Mat;
    std::atomic<int64_t> bad{0};  // values that don't fit fp16
    // Writer side of every layer: W_O F(G_V) per head, side by side: (d, H D).
    std::vector<int> wb(L);
    for (int64_t l = 0; l < L; ++l) wb[l] = batch.buffer(d * H(l) * D(l), fp16);
    parallel_for(N, [&](int64_t a) {
        const int64_t l = ids[a].first, h = ids[a].second, Dl = D(l), cols = H(l) * Dl;
        const Tensor B = matmul(heads[a].w.W_O, gram_factor(heads[a].GV));  // (d, D)
        for (int64_t r = 0; r < d; ++r) put(batch, wb[l], r * cols + h * Dl, B.ptr() + r * Dl, Dl, bad);
    });
    if (bad) return GpuRun::Overflow;
    const int prod = batch.buffer(scratch);
    // Reader side, as many layers per run as fit circuits_run_bytes(): rows [Q, K, V][head][D] of
    // F^T W for F of G_K, G_Q and G_OV^T, (3 H D, d).
    for (int64_t lb = 1; lb < L;) {
        int64_t end = lb, bytes = 0;
        do {
            bytes += 3 * H(end) * D(end) * d * 4;
            ++end;
        } while (end < L && bytes + 3 * H(end) * D(end) * d * 4 <= circuits_run_bytes());
        std::vector<int> rb(L, -1), out(L, -1);
        for (int64_t l = lb; l < end; ++l) {
            rb[l] = batch.buffer(3 * H(l) * D(l) * d, fp16);
            out[l] = batch.buffer(3 * H(l) * first[l]);
        }
        parallel_for(first[end] - first[lb], [&](int64_t j) {
            const int64_t b = first[lb] + j, l = ids[b].first, h = ids[b].second, Dl = D(l);
            const HeadCache& r = heads[b];
            const Tensor* G[3] = {&r.GK, &r.GQ, &r.GOt};
            const Tensor* W[3] = {&r.w.W_Q, &r.w.W_K, &r.w.W_V};
            for (int kind = 0; kind < 3; ++kind) {
                const Tensor A = matmul(transpose2d(gram_factor(*G[kind])), *W[kind]);  // (D, d)
                put(batch, rb[l], ((kind * H(l) + h) * Dl) * d, A.ptr(), A.numel(), bad);
            }
        });
        if (bad) return GpuRun::Overflow;
        for (int64_t l = lb; l < end; ++l)
            for (int64_t la = 0; la < l; ++la) {
                const int64_t R = 3 * H(l) * D(l), C = H(la) * D(la);
                batch.matmul(Mat{rb[l], 0, d}, Mat{wb[la], 0, C}, Mat{prod, 0, C}, R, C, d, false);
                batch.block_sumsq(Mat{prod, 0, C}, R, C, D(l), D(la), out[l], 3 * H(l) * first[la]);
                batch.end_group();
            }
        batch.run();
        for (int64_t l = lb; l < end; ++l) {
            const float* o = batch.floats(out[l]);
            for (int64_t la = 0; la < l; ++la)
                for (int64_t i = 0; i < H(l); ++i)
                    for (int64_t j = 0; j < H(la); ++j) {
                        const int64_t a = first[la] + j, b = first[l] + i;
                        const HeadCache &w = heads[a], &r = heads[b];
                        auto score = [&](int kind) {
                            return std::sqrt(std::max(0.f, o[3 * H(l) * first[la] + (kind * H(l) + i) * H(la) + j]));
                        };
                        c.q.data[a * N + b] = score(0) / (r.qk_norm * w.ov_norm);
                        c.k.data[a * N + b] = score(1) / (r.qk_norm * w.ov_norm);
                        c.v.data[a * N + b] = score(2) / (r.ov_norm * w.ov_norm);
                    }
            batch.release(rb[l]);
            batch.release(out[l]);
        }
        lb = end;
    }
    return GpuRun::Done;
}

}  // namespace
#endif

// The heads' Gram matrices are computed in parallel, then the pairs. Each pair's arithmetic is the
// same as on one thread, so the scores don't depend on the thread count.
Composition composition_scores(const Model& m) {
    std::vector<std::pair<int64_t, int64_t>> ids;  // (layer, head)
    Composition c;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) {
            ids.emplace_back(i, h);
            c.labels.push_back(head_label(i, h));
        }
    const int64_t N = std::ssize(ids);
    std::vector<HeadCache> heads(N);
    parallel_for(N, [&](int64_t a) { heads[a] = cache_head(head_weights(m, ids[a].first, ids[a].second)); });
    c.q = Tensor({N, N}, kNaN);
    c.k = Tensor({N, N}, kNaN);
    c.v = Tensor({N, N}, kNaN);
    std::vector<std::pair<int64_t, int64_t>> pairs;  // writer in an earlier layer than the reader
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b)
            if (ids[a].first < ids[b].first) pairs.emplace_back(a, b);
#ifdef TFI_USE_ACCELERATE
    try {
        GpuRun r = composition_gpu(m, ids, heads, c, m.weight_format() == WeightFormat::fp16);
        if (r == GpuRun::Overflow) r = composition_gpu(m, ids, heads, c, false);
        if (r == GpuRun::Done) return c;
    } catch (const detail::GpuFailure&) {  // still failing after retries: the CPU computes it below
        if (detail::gpu_only()) throw;     // ... except in GPU-only mode
    }
#endif
    parallel_for(std::ssize(pairs), [&](int64_t n) {
        const auto [a, b] = pairs[n];
        const CompScores s = compose(heads[a], heads[b]);
        c.q.data[a * N + b] = s.q;
        c.k.data[a * N + b] = s.k;
        c.v.data[a * N + b] = s.v;
    });
    return c;
}

// Each random pair draws from its own stream (seeded by seed and the pair's index), so the pairs run
// in parallel and the result doesn't depend on the thread count.
float composition_random_baseline(const Model& m, int64_t n_pairs, uint64_t seed) {
    const int64_t d = m.cfg().d_model, D = m.cfg().head_dim();
    std::vector<double> score(n_pairs);
    parallel_for(n_pairs, [&](int64_t n) {
        std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), static_cast<uint32_t>(n),
                          static_cast<uint32_t>(static_cast<uint64_t>(n) >> 32)};
        std::mt19937_64 rng(seq);
        std::normal_distribution<float> normal(0.f, 1.f);
        auto random = [&](Shape sh) {
            Tensor t(sh);
            for (float& v : t.data) v = normal(rng);
            return t;
        };
        auto random_head = [&]() {
            return cache_head({random({D, d}), random({D, d}), random({D, d}), random({d, D})});
        };
        const HeadCache writer = random_head(), reader = random_head();
        const CompScores s = compose(writer, reader);
        score[n] = (s.q + s.k + s.v) / 3.0;
    });
    double total = 0;
    for (double v : score) total += v;  // in pair order: deterministic
    return static_cast<float>(total / n_pairs);
}

namespace {

// OV paths through the sampled tokens. A head's OV circuit W_O W_V has rank D (head size) < d_model,
// so embed -> OV(a) -> ... -> OV(z) -> unembed is evaluated in factored form:
//   (X W_V(a)^T) (W_V(b) W_O(a))^T ... (U W_O(z))^T
// which is the same product as X W_V(a)^T W_O(a)^T ... W_O(z)^T U^T, with far fewer operations.
struct OvPaths {
    const Model& m;
    Tokens toks;  // the sampled tokens
    Tensor X;     // their embeddings (n, d)
    Tensor U;     // unembedding with the final norm's weight folded in (V, d)

    OvPaths(const Model& model, int64_t max_tokens) : m(model) {
        const ModelConfig& cfg = m.cfg();
        const int64_t V = cfg.vocab_size, d = cfg.d_model, n = std::min(V, max_tokens);
        toks.resize(n);
        for (int64_t i = 0; i < n; ++i) toks[i] = i * V / n;
        const Weight& E = m.weight("embed.weight");
        X = Tensor({n, d});
        for (int64_t i = 0; i < n; ++i) E.read_rows(toks[i], toks[i] + 1, X.ptr() + i * d);
        U = m.unembed().to_tensor();
        if (cfg.norm != "none" && cfg.norm_position != "post") {
            const Tensor& gamma = m.weight("ln_final.weight").f32();
            for (int64_t v = 0; v < V; ++v)
                for (int64_t j = 0; j < d; ++j) U.data[v * d + j] *= gamma.data[j];
        }
    }
    // The sampled tokens after the first head's W_V: (n, D).
    Tensor start(const HeadWeights& w) const { return matmul(X, w.W_V, true); }
    // Carry A (after `from`'s W_V) through from's W_O and `to`'s W_V: (n, D).
    static Tensor step(const Tensor& A, const HeadWeights& from, const HeadWeights& to) {
        return matmul(A, matmul(to.W_V, from.W_O), true);
    }
    // The unembedding seen through the last head's W_O: (V, D).
    Tensor finish(const HeadWeights& w) const { return matmul(U, w.W_O); }
    // Fraction of sampled tokens whose logits A UO^T rank the token itself first.
    float hits(const Tensor& A, const Tensor& UO) const {
        const int64_t n = A.shape[0], chunk = 256;
        int64_t hit = 0;
        for (int64_t s = 0; s < n; s += chunk) {
            const int64_t e = std::min(n, s + chunk);
            const Tensor logits = matmul(rows(A, s, e), UO, true);  // (chunk, V)
            for (int64_t r = 0; r < e - s; ++r) hit += topk(logits.row_view(r), 1)[0] == toks[s + r];
        }
        return float(hit) / float(n);
    }
};

}  // namespace

#ifdef TFI_USE_ACCELERATE
namespace {

// What OvPaths reads, without its float32 copy of the unembedding: the GPU uses the unembedding in
// place and folds the final norm's scale into each head's W_O instead.
struct OvInputs {
    Tokens toks;
    Tensor X;                   // the sampled tokens' embeddings (n, d)
    Tensor final_gamma;         // the final norm's scale, or empty
    std::vector<Tensor> gamma;  // each layer's ln1 scale (folded into W_V by head_weights), or empty
    OvInputs(const Model& m, int64_t max_tokens) {
        const ModelConfig& cfg = m.cfg();
        const int64_t V = cfg.vocab_size, d = cfg.d_model, n = std::min(V, max_tokens);
        toks.resize(n);
        for (int64_t i = 0; i < n; ++i) toks[i] = i * V / n;  // as OvPaths
        const Weight& E = m.weight("embed.weight");
        X = Tensor({n, d});
        for (int64_t i = 0; i < n; ++i) E.read_rows(toks[i], toks[i] + 1, X.ptr() + i * d);
        if (cfg.norm != "none" && cfg.norm_position != "post") final_gamma = m.weight("ln_final.weight").f32();
        for (int64_t l = 0; l < cfg.n_layers; ++l) {
            const ModelConfig& c = m.layer_cfg(l);
            gamma.push_back(c.norm != "none" && c.norm_position != "post" ? m.weight(lp(l) + "ln1.weight").f32()
                                                                          : Tensor());
        }
    }
    // X with layer l's ln1 scale on each column: X W_V^T for the scaled W_V.
    Tensor x(int64_t l) const {
        Tensor out = X;
        const int64_t n = X.shape[0], d = X.shape[1];
        if (!gamma[l].empty())
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < d; ++j) out.data[i * d + j] *= gamma[l].data[j];
        return out;
    }
    // Layer l's W_O transposed, (H D, d), with the final norm's scale on each column.
    Tensor wo_t(const Model& m, int64_t l) const {
        const Tensor O = m.weight(lp(l) + "attn.o_proj.weight").to_tensor();  // (d, H D)
        const int64_t d = O.shape[0], HD = O.shape[1];
        Tensor out({HD, d});
        for (int64_t c = 0; c < HD; ++c)
            for (int64_t j = 0; j < d; ++j)
                out.data[c * d + j] = O.data[j * HD + c] * (final_gamma.empty() ? 1.f : final_gamma.data[j]);
        return out;
    }
};

using Mat = detail::Gpu::Batch::Mat;

// ov_copying_scores on the GPU, a layer per group: (U W_O)^T for all its heads in one multiply, X W_V^T
// likewise, then each head's logits (n, V) and their largest entries. With fp16 every multiply takes
// fp16 inputs: the products that feed another multiply are converted on the GPU, which reports a
// value too large (Overflow).
GpuRun ov_copying_gpu(const Model& m, int64_t max_tokens, Tensor& out, bool fp16) {
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model, V = m.cfg().vocab_size, H = m.max_heads();
    const int64_t n = std::min(V, max_tokens);
    double work = 0;
    int64_t max_hd = 0, max_kv = 0;
    for (int64_t l = 0; l < L; ++l) {
        const ModelConfig& c = m.layer_cfg(l);
        const int64_t HD = c.n_heads * c.head_dim(), KV = c.kv_heads() * c.head_dim();
        work += double(HD) * d * V + double(n) * HD * V + double(n) * d * KV;
        max_hd = std::max(max_hd, HD);
        max_kv = std::max(max_kv, KV);
    }
    detail::Gpu* gpu = detail::gpu_for_work(m, work);
    if (!gpu) return GpuRun::NotWorth;
    const OvInputs in(m, max_tokens);
    detail::Gpu::Batch batch(*gpu);
    std::atomic<int64_t> bad{0};
    const int U = batch.weight(m.unembed(), fp16);
    if (fp16 && !batch.is_fp16(U)) return GpuRun::Overflow;
    const int uo_t = batch.buffer(max_hd * V), xv = batch.buffer(n * max_kv), logits = batch.buffer(n * V);
    // fp16 copies of the products that feed the next multiply (the float32 ones without fp16).
    const int uo_h = fp16 ? batch.buffer(max_hd * V, true) : uo_t, xv_h = fp16 ? batch.buffer(n * max_kv, true) : xv;
    for (int64_t l0 = 0; l0 < L;) {  // as many layers per run as fit circuits_run_bytes()
        int64_t l1 = l0, bytes = 0;
        do {
            bytes += (m.layer_cfg(l1).n_heads * m.layer_cfg(l1).head_dim() * d + n * d) * 4;
            ++l1;
        } while (l1 < L && bytes + (m.layer_cfg(l1).n_heads * m.layer_cfg(l1).head_dim() * d + n * d) * 4 <=
                               circuits_run_bytes());
        std::vector<int> wo(L, -1), x(L, -1), idx(L, -1);
        for (int64_t l = l0; l < l1; ++l) {
            const ModelConfig& c = m.layer_cfg(l);
            const int64_t Hl = c.n_heads, D = c.head_dim(), HD = Hl * D, KV = c.kv_heads() * D, rep = Hl / c.kv_heads();
            wo[l] = batch.buffer(HD * d, fp16);
            x[l] = batch.buffer(n * d, fp16);
            idx[l] = batch.buffer(Hl * n);
            const Tensor wo_t = in.wo_t(m, l), xl = in.x(l);
            put(batch, wo[l], 0, wo_t.ptr(), wo_t.numel(), bad);
            put(batch, x[l], 0, xl.ptr(), xl.numel(), bad);
            const int Wv = batch.weight(m.weight(lp(l) + "attn.v_proj.weight"), fp16);
            if (bad || (fp16 && !batch.is_fp16(Wv))) return GpuRun::Overflow;
            batch.matmul(Mat{wo[l], 0, d}, Mat{U, 0, d}, Mat{uo_t, 0, V}, HD, V, d, true);  // (H D, V)
            batch.matmul(Mat{x[l], 0, d}, Mat{Wv, 0, d}, Mat{xv, 0, KV}, n, KV, d, true);   // (n, Hkv D)
            if (fp16) {
                batch.to_half(uo_t, uo_h, HD * V);
                batch.to_half(xv, xv_h, n * KV);
            }
            for (int64_t h = 0; h < Hl; ++h) {
                batch.matmul(Mat{xv_h, (h / rep) * D, KV}, Mat{uo_h, h * D * V, V}, Mat{logits, 0, V}, n, V, D, false);
                batch.row_argmax(Mat{logits, 0, V}, n, V, idx[l], h * n);
            }
            batch.end_group();
        }
        batch.run();
        if (batch.overflowed()) return GpuRun::Overflow;
        for (int64_t l = l0; l < l1; ++l) {
            const uint32_t* top = batch.uints(idx[l]);
            for (int64_t h = 0; h < m.layer_cfg(l).n_heads; ++h) {
                int64_t hit = 0;
                for (int64_t r = 0; r < n; ++r) hit += static_cast<int64_t>(top[h * n + r]) == in.toks[r];
                out.data[l * H + h] = float(hit) / float(n);
            }
            batch.release(wo[l]);
            batch.release(x[l]);
            batch.release(idx[l]);
        }
        l0 = l1;
    }
    return GpuRun::Done;
}

// ov_pair_copying_scores on the GPU: a group per (reader layer, writer layer), each recomputing what
// it needs, so groups share no scratch. fp16 as in ov_copying_gpu.
GpuRun ov_pair_copying_gpu(const Model& m, int64_t max_tokens, const std::vector<std::pair<int64_t, int64_t>>& ids,
                           Tensor& out, bool fp16) {
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model, V = m.cfg().vocab_size, N = std::ssize(ids);
    const int64_t n = std::min(V, max_tokens);
    auto H = [&](int64_t l) { return m.layer_cfg(l).n_heads; };
    auto D = [&](int64_t l) { return m.layer_cfg(l).head_dim(); };
    auto KV = [&](int64_t l) { return m.layer_cfg(l).kv_heads() * D(l); };
    std::vector<int64_t> first(L + 1, 0);
    for (int64_t l = 0; l < L; ++l) first[l + 1] = first[l] + H(l);
    double work = 0;
    int64_t max_hd = 0, max_kv = 0, max_m = 0, max_d = 0;
    for (int64_t lb = 0; lb < L; ++lb) {
        max_hd = std::max(max_hd, H(lb) * D(lb));
        max_kv = std::max(max_kv, KV(lb));
        max_d = std::max(max_d, D(lb));
        for (int64_t la = 0; la < lb; ++la) {
            work += double(H(la) * H(lb)) * n * D(lb) * V + double(H(lb) * D(lb)) * d * V;
            max_m = std::max(max_m, KV(lb) * H(la) * D(la));
        }
    }
    detail::Gpu* gpu = detail::gpu_for_work(m, work);
    if (!gpu) return GpuRun::NotWorth;
    const OvInputs in(m, max_tokens);
    detail::Gpu::Batch batch(*gpu);
    std::atomic<int64_t> bad{0};
    const int U = batch.weight(m.unembed(), fp16);
    const int uo_t = batch.buffer(max_hd * V), xv = batch.buffer(n * max_kv),
              M = batch.buffer(std::max<int64_t>(max_m, 1));
    const int a2 = batch.buffer(n * max_d), logits = batch.buffer(n * V);
    const int uo_h = fp16 ? batch.buffer(max_hd * V, true) : uo_t, xv_h = fp16 ? batch.buffer(n * max_kv, true) : xv;
    const int M_h = fp16 ? batch.buffer(std::max<int64_t>(max_m, 1), true) : M,
              a2_h = fp16 ? batch.buffer(n * max_d, true) : a2;
    std::vector<int> wo(L), x(L), vs(L), wv(L), o(L);
    for (int64_t l = 0; l < L; ++l) {
        wo[l] = batch.buffer(H(l) * D(l) * d, fp16);
        const Tensor wo_t = in.wo_t(m, l), xl = in.x(l);
        put(batch, wo[l], 0, wo_t.ptr(), wo_t.numel(), bad);
        x[l] = batch.buffer(n * d, fp16);
        put(batch, x[l], 0, xl.ptr(), xl.numel(), bad);
        // W_V with ln1's scale on each column, (Hkv D, d): the reader side of W_V(b) W_O(a).
        vs[l] = batch.buffer(KV(l) * d, fp16);
        Tensor Wv = m.weight(lp(l) + "attn.v_proj.weight").to_tensor();
        if (!in.gamma[l].empty())
            for (int64_t r = 0; r < KV(l); ++r)
                for (int64_t j = 0; j < d; ++j) Wv.data[r * d + j] *= in.gamma[l].data[j];
        put(batch, vs[l], 0, Wv.ptr(), Wv.numel(), bad);
        wv[l] = batch.weight(m.weight(lp(l) + "attn.v_proj.weight"), fp16);
        o[l] = batch.weight(m.weight(lp(l) + "attn.o_proj.weight"), fp16);
        if (fp16 && !(batch.is_fp16(wv[l]) && batch.is_fp16(o[l]))) return GpuRun::Overflow;
    }
    if (bad || (fp16 && !batch.is_fp16(U))) return GpuRun::Overflow;
    // Each (writer, reader) pair's top tokens, at pair_index * n.
    std::vector<int64_t> slot(static_cast<size_t>(N * N), -1);
    int64_t pairs = 0;
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b)
            if (ids[a].first < ids[b].first) slot[a * N + b] = pairs++;
    const int idx = batch.buffer(std::max<int64_t>(pairs, 1) * n);
    for (int64_t lb = 1; lb < L; ++lb)
        for (int64_t la = 0; la < lb; ++la) {
            const int64_t Db = D(lb), Da = D(la), rep_b = H(lb) / m.layer_cfg(lb).kv_heads(),
                          rep_a = H(la) / m.layer_cfg(la).kv_heads(), ldm = H(la) * Da;
            batch.matmul(Mat{wo[lb], 0, d}, Mat{U, 0, d}, Mat{uo_t, 0, V}, H(lb) * Db, V, d,
                         true);  // reader's (U W_O)^T
            batch.matmul(Mat{x[la], 0, d}, Mat{wv[la], 0, d}, Mat{xv, 0, KV(la)}, n, KV(la), d,
                         true);  // writer's X W_V^T
            batch.matmul(Mat{vs[lb], 0, d}, Mat{o[la], 0, ldm}, Mat{M, 0, ldm}, KV(lb), ldm, d,
                         false);  // W_V(b) W_O(a)
            if (fp16) {
                batch.to_half(uo_t, uo_h, H(lb) * Db * V);
                batch.to_half(xv, xv_h, n * KV(la));
                batch.to_half(M, M_h, KV(lb) * ldm);
            }
            for (int64_t i = 0; i < H(la); ++i)
                for (int64_t j = 0; j < H(lb); ++j) {
                    const int64_t a = first[la] + i, b = first[lb] + j;
                    // A' = (X W_V(a)^T) (W_V(b) W_O(a))^T, (n, D_b); then its logits through b's (U W_O)^T.
                    batch.matmul(Mat{xv_h, (i / rep_a) * Da, KV(la)}, Mat{M_h, (j / rep_b) * Db * ldm + i * Da, ldm},
                                 Mat{a2, 0, Db}, n, Db, Da, true);
                    if (fp16) batch.to_half(a2, a2_h, n * Db);
                    batch.matmul(Mat{a2_h, 0, Db}, Mat{uo_h, j * Db * V, V}, Mat{logits, 0, V}, n, V, Db, false);
                    batch.row_argmax(Mat{logits, 0, V}, n, V, idx, slot[a * N + b] * n);
                }
            batch.end_group();
        }
    batch.run();
    if (batch.overflowed()) return GpuRun::Overflow;
    const uint32_t* top = batch.uints(idx);
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b) {
            if (slot[a * N + b] < 0) continue;
            int64_t hit = 0;
            for (int64_t r = 0; r < n; ++r) hit += static_cast<int64_t>(top[slot[a * N + b] * n + r]) == in.toks[r];
            out.data[a * N + b] = float(hit) / float(n);
        }
    return GpuRun::Done;
}

}  // namespace
#endif

float ov_path_copying(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& path, int64_t max_tokens) {
    if (path.empty()) throw std::invalid_argument("OV path needs at least one head");
    for (size_t i = 1; i < path.size(); ++i)
        if (path[i].first <= path[i - 1].first)
            throw std::invalid_argument("OV path heads must be in increasing layers");
    const OvPaths ov(m, max_tokens);
    std::vector<HeadWeights> w;
    for (const auto& [layer, head] : path) w.push_back(head_weights(m, layer, head));
    Tensor A = ov.start(w[0]);
    for (size_t i = 1; i < w.size(); ++i) A = OvPaths::step(A, w[i - 1], w[i]);
    return ov.hits(A, ov.finish(w.back()));
}

Tensor ov_copying_scores(const Model& m, int64_t max_tokens) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
#ifdef TFI_USE_ACCELERATE
    try {
        Tensor out({L, H}, kNaN);
        GpuRun r = ov_copying_gpu(m, max_tokens, out, m.weight_format() == WeightFormat::fp16);
        if (r == GpuRun::Overflow) r = ov_copying_gpu(m, max_tokens, out, false);
        if (r == GpuRun::Done) return out;
    } catch (const detail::GpuFailure&) {  // still failing after retries: the CPU computes it below
        if (detail::gpu_only()) throw;     // ... except in GPU-only mode
    }
#endif
    const OvPaths ov(m, max_tokens);
    std::vector<std::pair<int64_t, int64_t>> ids;
    for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) ids.emplace_back(i, h);
    Tensor out({L, H}, kNaN);
    parallel_for(std::ssize(ids), [&](int64_t j) {
        const auto [i, h] = ids[j];
        const HeadWeights w = head_weights(m, i, h);
        out.data[i * H + h] = ov.hits(ov.start(w), ov.finish(w));
    });
    return out;
}

Tensor ov_pair_copying_scores(const Model& m, int64_t max_tokens) {
    std::vector<std::pair<int64_t, int64_t>> ids;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) ids.emplace_back(i, h);
    const int64_t N = std::ssize(ids);
#ifdef TFI_USE_ACCELERATE
    try {
        Tensor out({N, N}, kNaN);
        GpuRun r = ov_pair_copying_gpu(m, max_tokens, ids, out, m.weight_format() == WeightFormat::fp16);
        if (r == GpuRun::Overflow) r = ov_pair_copying_gpu(m, max_tokens, ids, out, false);
        if (r == GpuRun::Done) return out;
    } catch (const detail::GpuFailure&) {  // still failing after retries: the CPU computes it below
        if (detail::gpu_only()) throw;     // ... except in GPU-only mode
    }
#endif
    const OvPaths ov(m, max_tokens);
    std::vector<HeadWeights> w(N);
    std::vector<Tensor> A(N);  // each head as the writer: (n, D)
    parallel_for(N, [&](int64_t a) {
        w[a] = head_weights(m, ids[a].first, ids[a].second);
        A[a] = ov.start(w[a]);
    });
    Tensor out({N, N}, kNaN);
    // One reader per job: its unembedding side (V, D) is computed once for all its writers.
    parallel_for(N, [&](int64_t b) {
        Tensor UO;
        for (int64_t a = 0; a < N; ++a) {
            if (ids[a].first >= ids[b].first) continue;
            if (UO.data.empty()) UO = ov.finish(w[b]);
            out.data[a * N + b] = ov.hits(OvPaths::step(A[a], w[a], w[b]), UO);
        }
    });
    return out;
}

Tensor positional_qk(const Model& m, int64_t layer, int64_t head, int64_t T) {
    if (m.cfg().positional != "learned") throw std::runtime_error("positional QK needs learned positional embeddings");
    if (T > m.cfg().max_seq_len) throw std::out_of_range("T exceeds max_seq_len");
    const HeadWeights w = head_weights(m, layer, head);
    const Tensor P = m.weight("pos_embed.weight").rows(0, T);
    const Tensor S = matmul(matmul(P, w.W_Q, true), matmul(P, w.W_K, true), true);  // (T, T)
    const ModelConfig& c = m.layer_cfg(layer);
    const float scale = static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(c.head_dim())));
    Tensor pattern({T, T});
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t j = 0; j < T; ++j) pattern.data[i * T + j] = j <= i ? S.data[i * T + j] * scale : -INFINITY;
        softmax_(pattern.row_view(i));
    }
    return pattern;
}

}  // namespace tfi
