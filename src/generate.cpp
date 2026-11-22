#include "tfi/generate.h"

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tfi {

namespace {
void check(const GenerateOptions& o) {
    if (o.max_new_tokens < 0) throw std::invalid_argument("max_new_tokens must be 0 or more");
    if (!(o.temperature >= 0) || std::isinf(o.temperature)) throw std::invalid_argument("temperature must be 0 or more");
    if (o.top_k < 0) throw std::invalid_argument("top_k must be 0 or more");
    if (!(o.top_p > 0 && o.top_p <= 1)) throw std::invalid_argument("top_p must be in (0, 1]");
    if (o.speculative.draft_tokens < 0) throw std::invalid_argument("speculative.draft_tokens must be 0 or more");
    if (o.speculative.ngram < 1) throw std::invalid_argument("speculative.ngram must be 1 or more");
}
}  // namespace

Sampler::Sampler(const GenerateOptions& o) : temperature_(o.temperature), top_p_(o.top_p), top_k_(o.top_k), rng_(o.seed) {
    check(o);
}

// exp(x) for the sampler's weights, Cephes' expf: 2^n e^r with |r| <= ln(2)/2, a degree-5 polynomial for
// e^r, relative error about 1e-7. Written as separate IEEE multiplies and adds (no fused multiply-adds, no
// library calls), the same in the vector loop as in the scalar one, so the weights, and with them the
// tokens a seed picks, are the same bits on every platform.
namespace {
constexpr float kLog2e = 1.44269504088896341f, kLn2Hi = 0.693359375f, kLn2Lo = -2.12194440e-4f;
constexpr float kP0 = 1.9875691500e-4f, kP1 = 1.3981999507e-3f, kP2 = 8.3334519073e-3f, kP3 = 4.1665795894e-2f,
                kP4 = 1.6666665459e-1f, kP5 = 5.0000001201e-1f;
constexpr float kMinExp = -87.f;  // below it exp(x) is under 1.7e-38 (fp32's smallest normal): taken as 0
}  // namespace

namespace detail {

float exp_weight(float x) {
    if (!(x >= kMinExp)) return 0.f;  // also NaN
    float fx = x * kLog2e;
    fx = fx + 0.5f;
    fx = std::floor(fx);  // n, from -126 (x = -87) to 0
    float t = fx * kLn2Hi;
    float r = x - t;
    t = fx * kLn2Lo;
    r = r - t;
    const float z = r * r;
    float y = kP0 * r;
    y = y + kP1;
    y = y * r;
    y = y + kP2;
    y = y * r;
    y = y + kP3;
    y = y * r;
    y = y + kP4;
    y = y * r;
    y = y + kP5;
    y = y * z;
    y = y + r;
    y = y + 1.f;
    const uint32_t bits = static_cast<uint32_t>(static_cast<int32_t>(fx) + 127) << 23;  // 2^n
    float scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return y * scale;
}

void exp_weights(const float* logits, int64_t n, float top, float scale, float* w) {
    int64_t i = 0;
#if defined(__aarch64__)
    const float32x4_t vtop = vdupq_n_f32(top), vscale = vdupq_n_f32(scale), vmin = vdupq_n_f32(kMinExp);
    const float32x4_t zero = vdupq_n_f32(0.f), half = vdupq_n_f32(0.5f), one = vdupq_n_f32(1.f);
    const float32x4_t log2e = vdupq_n_f32(kLog2e), hi = vdupq_n_f32(kLn2Hi), lo = vdupq_n_f32(kLn2Lo);
    const float32x4_t p0 = vdupq_n_f32(kP0), p1 = vdupq_n_f32(kP1), p2 = vdupq_n_f32(kP2), p3 = vdupq_n_f32(kP3),
                      p4 = vdupq_n_f32(kP4), p5 = vdupq_n_f32(kP5);
    const int32x4_t bias = vdupq_n_s32(127);
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vsubq_f32(vld1q_f32(logits + i), vtop);
        x = vmulq_f32(x, vscale);
        const uint32x4_t ok = vcgeq_f32(x, vmin);  // false for NaN and -inf too
        x = vbslq_f32(ok, x, zero);                // keep the other lanes' arithmetic finite
        float32x4_t fx = vmulq_f32(x, log2e);
        fx = vaddq_f32(fx, half);
        fx = vrndmq_f32(fx);
        float32x4_t r = vsubq_f32(x, vmulq_f32(fx, hi));
        r = vsubq_f32(r, vmulq_f32(fx, lo));
        const float32x4_t z = vmulq_f32(r, r);
        float32x4_t y = vmulq_f32(p0, r);
        y = vaddq_f32(y, p1);
        y = vaddq_f32(vmulq_f32(y, r), p2);
        y = vaddq_f32(vmulq_f32(y, r), p3);
        y = vaddq_f32(vmulq_f32(y, r), p4);
        y = vaddq_f32(vmulq_f32(y, r), p5);
        y = vmulq_f32(y, z);
        y = vaddq_f32(y, r);
        y = vaddq_f32(y, one);
        const uint32x4_t bits = vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(vcvtq_s32_f32(fx), bias)), 23);
        y = vmulq_f32(y, vreinterpretq_f32_u32(bits));
        vst1q_f32(w + i, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(y), ok)));
    }
#endif
    for (; i < n; ++i) {
        const float d = logits[i] - top;
        w[i] = exp_weight(d * scale);
    }
}

}  // namespace detail

namespace {

// Sum of float weights in double, in a fixed order (four running sums), so the same on every platform.
double sum_weights(const float* w, size_t n) {
    double s[4] = {0, 0, 0, 0};
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        for (size_t k = 0; k < 4; ++k) s[k] += w[i + k];
    for (; i < n; ++i) s[0] += w[i];
    return (s[0] + s[1]) + (s[2] + s[3]);
}

// A logit's bits as an unsigned key in the logits' order (key 0 is left for "never").
uint32_t logit_key(float l) {
    const float v = l + 0.f;  // -0 as +0: equal logits, equal keys
    uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Among n items with keys key(i) (larger first; 0: never taken) and amounts amount(i), the fewest with the
// largest keys, lower index first among equal keys, whose amounts reach `target`: those with key > key_,
// and the first m with key == key_ (key_ 0: every item with a key). In linear time, without sorting: a
// histogram of amounts by the keys' top 12 bits finds the bucket where the running sum reaches the target;
// its items (one more pass) are narrowed by the next 12 bits, then the last 8, to items of equal keys.
struct Cut {
    uint32_t key = 0;
    int64_t m = 0;
    // Whether an item with key k is kept, items visited in index order, `taken` counting those with the cut's key.
    bool keeps(uint32_t k, int64_t& taken) const { return k > key || (k == key && k != 0 && taken++ < m); }
};

template <typename Key, typename Amount>
Cut cut_at(int64_t n, Key key, Amount amount, double target, std::vector<double>& hist, std::vector<int64_t>& sel) {
    hist.assign(4096, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t k = key(i);
        if (k) hist[k >> 20] += amount(i);
    }
    double acc = 0;
    auto boundary = [&](int64_t buckets) {  // the bucket, from the top, where acc reaches the target (-1: none)
        int64_t b = buckets - 1;
        for (; b >= 0 && acc + hist[static_cast<size_t>(b)] < target; --b) acc += hist[static_cast<size_t>(b)];
        return b;
    };
    const int64_t b1 = boundary(4096);
    if (b1 < 0) return {};  // rounding left the target out of reach: everything
    sel.clear();
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t k = key(i);
        if (k && (k >> 20) == static_cast<uint32_t>(b1)) sel.push_back(i);
    }
    uint32_t prefix = static_cast<uint32_t>(b1);
    for (const int shift : {8, 0}) {  // the next 12 bits, then the last 8
        const int bits = shift ? 12 : 8;
        hist.assign(size_t(1) << bits, 0.0);
        for (int64_t i : sel) hist[(key(i) >> shift) & ((1u << bits) - 1)] += amount(i);
        const int64_t b = boundary(int64_t(1) << bits);
        if (b < 0) return {prefix << (shift + bits), std::numeric_limits<int64_t>::max()};  // rounding: all of the bucket
        prefix = (prefix << bits) | static_cast<uint32_t>(b);
        size_t kept = 0;
        for (int64_t i : sel)
            if ((key(i) >> shift) == prefix) sel[kept++] = i;
        sel.resize(kept);
    }
    Cut c{prefix, 0};
    for (size_t j = 0; j < sel.size() && acc < target; ++j) {  // equal keys: lower index first
        ++c.m;
        acc += amount(sel[j]);
    }
    return c;
}

// The largest logit (NaN ignored), -inf if none; four lanes at a time on ARM (max is exact, so the same).
float max_logit(const float* l, int64_t n) {
    float top = -INFINITY;
    int64_t i = 0;
#if defined(__aarch64__)
    float32x4_t m = vdupq_n_f32(-INFINITY);
    for (; i + 4 <= n; i += 4) m = vmaxnmq_f32(m, vld1q_f32(l + i));
    top = vmaxnmvq_f32(m);
#endif
    for (; i < n; ++i)
        if (l[i] > top) top = l[i];
    return top;
}

}  // namespace

bool Sampler::select(const float* logits, int64_t n, int64_t& best, double& total) {
    // Greedy: the largest logit, the lowest id among equal ones.
    const float top = max_logit(logits, n);
    if (top == -INFINITY) throw std::runtime_error("generate: no logit to choose a token from (all NaN or -inf)");
    best = 0;
    while (!(logits[best] == top)) ++best;
    if (temperature_ == 0 || top_k_ == 1 || top == INFINITY) return false;
    const float scale = static_cast<float>(1.0 / temperature_);
    auto usable = [](float l) { return !std::isnan(l) && l != -INFINITY; };
    ids_.clear();
    if (top_k_ > 0) {  // the k largest logits (all if fewer), then only their weights
        keys_.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) keys_[static_cast<size_t>(i)] = usable(logits[i]) ? logit_key(logits[i]) : 0u;
        auto key = [&](int64_t i) { return keys_[static_cast<size_t>(i)]; };
        const Cut c = cut_at(n, key, [](int64_t) { return 1.0; }, static_cast<double>(top_k_), hist_, sel_);
        int64_t taken = 0;
        for (int64_t i = 0; i < n; ++i)
            if (c.keeps(key(i), taken)) ids_.push_back(i);
        gathered_.resize(ids_.size());
        for (size_t j = 0; j < ids_.size(); ++j) gathered_[j] = logits[ids_[j]];
        w_.resize(ids_.size());
        detail::exp_weights(gathered_.data(), static_cast<int64_t>(ids_.size()), top, scale, w_.data());
    } else {
        w_.resize(static_cast<size_t>(n));
        detail::exp_weights(logits, n, top, scale, w_.data());
    }
    if (top_p_ < 1) {  // the nucleus: a weight's bits order as its value (weights are >= 0; 0: never)
        const size_t m = w_.size();
        auto key = [&](int64_t j) {
            uint32_t k;
            std::memcpy(&k, &w_[static_cast<size_t>(j)], sizeof k);
            return k;
        };
        const Cut c = cut_at(static_cast<int64_t>(m), key, [&](int64_t j) { return static_cast<double>(w_[static_cast<size_t>(j)]); },
                             top_p_ * sum_weights(w_.data(), m), hist_, sel_);
        const bool dense = ids_.empty();
        size_t kept = 0;
        int64_t taken = 0;
        for (size_t j = 0; j < m; ++j) {
            if (!c.keeps(key(static_cast<int64_t>(j)), taken)) continue;
            if (dense) ids_.push_back(static_cast<int64_t>(j));
            else ids_[kept] = ids_[j];
            w_[kept++] = w_[j];
        }
        ids_.resize(kept);
        w_.resize(kept);
    }
    total = sum_weights(w_.data(), w_.size());
    return true;
}

int64_t Sampler::operator()(const float* logits, int64_t n) {
    int64_t best;
    double total;
    if (!select(logits, n, best, total)) return best;
    const double u = uniform() * total;
    // The token where the running sum passes u: whole blocks' sums first, then within the block.
    constexpr size_t kBlock = 64;
    double sum = 0;
    int64_t last = best;
    for (size_t b = 0; b < w_.size(); b += kBlock) {
        const size_t e = std::min(w_.size(), b + kBlock);
        const double block = sum_weights(w_.data() + b, e - b);
        if (u >= sum + block && e < w_.size()) {
            sum += block;
            continue;
        }
        for (size_t j = b; j < e; ++j) {
            if (!(w_[j] > 0)) continue;
            sum += w_[j];
            last = id(j);
            if (u < sum) return last;
        }
    }
    return last;  // rounding left u at the very end
}

void Sampler::distribution(const float* logits, int64_t n, std::vector<double>& p) {
    int64_t best;
    double total;
    const bool sampled = select(logits, n, best, total);
    p.assign(static_cast<size_t>(n), 0.0);
    if (!sampled) {
        p[static_cast<size_t>(best)] = 1;
        return;
    }
    for (size_t j = 0; j < w_.size(); ++j) p[static_cast<size_t>(id(j))] = w_[j] / total;
}

// The same on every standard library (mt19937_64's output is specified; its distributions are not).
double Sampler::uniform() { return static_cast<double>(rng_() >> 11) * 0x1.0p-53; }

int64_t Sampler::draw(const std::vector<double>& w) {
    double total = 0;
    for (double x : w) total += x;
    if (!(total > 0)) throw std::logic_error("Sampler::draw: no weight to draw from");
    const double u = uniform() * total;
    double sum = 0;
    int64_t last = -1;
    for (size_t i = 0; i < w.size(); ++i) {
        if (!(w[i] > 0)) continue;
        sum += w[i];
        last = static_cast<int64_t>(i);
        if (u < sum) return last;
    }
    return last;  // rounding left u at the very end
}

namespace {

// The last row of each activation the trace needs, from every pass (capture hooks).
struct TraceRecorder {
    const Model& m;
    const TraceOptions& opt;
    std::set<std::string> points;
    // name -> its last row: (1, ...); patterns: (H, 1, keys). With all_rows (speculative steps, a few rows
    // each), every row of the latest pass instead.
    ActivationCache last;
    bool all_rows = false;

    TraceRecorder(const Model& model, const TraceOptions& o) : m(model), opt(o) {
        if (o.top_k < 0 || o.attention_top < 0) throw std::invalid_argument("trace: top_k and attention_top must be 0 or more");
        if (o.lens)
            for (const std::string& n : lens_hook_points(m)) points.insert(n);
        if (o.dla) {
            direct_logit_attribution_from_check();
            for (const std::string& n : dla_hook_points(m)) points.insert(n);
        }
        if (o.attention_top > 0)
            for (int64_t l = 0; l < m.cfg().n_layers; ++l) points.insert("blocks." + std::to_string(l) + ".attn.hook_pattern");
    }
    void direct_logit_attribution_from_check() const {
        for (int64_t i = 0; i < m.cfg().n_layers; ++i)
            if (m.cfg().norm_position == "post" || m.layer_cfg(i).norm_position == "post")
                throw std::invalid_argument("trace: direct logit attribution needs an additive residual stream; post-norm "
                                            "models have none (set trace.dla = false)");
    }
    void capture(const std::string& name, const Tensor& a) {
        if (all_rows) {
            last[name] = a;
            return;
        }
        if (name.size() > 12 && name.compare(name.size() - 12, 12, "hook_pattern") == 0) {  // (H, T, keys): each head's last query
            const int64_t H = a.shape[0], T = a.shape[1], K = a.shape[2];
            Tensor t({H, 1, K});
            for (int64_t h = 0; h < H; ++h) std::copy(a.ptr() + (h * T + T - 1) * K, a.ptr() + (h * T + T) * K, t.ptr() + h * K);
            last[name] = std::move(t);
            return;
        }
        const int64_t rows = a.shape[0], width = a.numel() / std::max<int64_t>(rows, 1);
        Shape shape = a.shape;
        shape[0] = 1;
        last[name] = Tensor(shape, std::vector<float>(a.ptr() + (rows - 1) * width, a.ptr() + rows * width));
    }
    // The step for `token`, chosen from `logits`: those of row `row` of the latest pass (0 without all_rows).
    TraceStep step(const float* logits, int64_t V, int64_t token, int64_t position, int64_t row = 0) const {
        TraceStep st;
        st.token = token;
        st.position = position;
        std::vector<float> p(logits, logits + V);
        softmax_(p.data(), V);
        st.prob = p[token];
        st.logprob = std::log(std::max(p[token], std::numeric_limits<float>::min()));
        st.top_tokens = topk(p.data(), V, opt.top_k);
        for (int64_t t : st.top_tokens) st.top_probs.push_back(p[t]);
        if (opt.lens) st.lens = logit_lens_from(m, last, row, token, opt.top_k);
        if (opt.dla) st.dla = direct_logit_attribution_from(m, last, row, token);
        for (int64_t l = 0; opt.attention_top > 0 && l < m.cfg().n_layers; ++l) {
            const Tensor& P = last.at("blocks." + std::to_string(l) + ".attn.hook_pattern");
            const int64_t H = P.shape[0], T = P.shape[1], K = P.shape[2];
            const int64_t keys = K - (T - 1 - row);  // the keys row `row` sees: those up to its position
            for (int64_t h = 0; h < H; ++h) {
                TraceStep::Attention a;
                a.head = "L" + std::to_string(l) + "H" + std::to_string(h);
                const float* pr = P.ptr() + (h * T + row) * K;
                a.positions = topk(pr, keys, std::min(opt.attention_top, keys));
                for (int64_t j : a.positions) a.weights.push_back(pr[j]);
                st.attention.push_back(std::move(a));
            }
        }
        return st;
    }
};

// The positions `m` can run (0: no limit): learned positional embeddings' max_seq_len, or a RoPE scaling's
// original_max_position_embeddings, beyond which "dynamic" and "longrope" frequencies change.
int64_t position_limit(const Model& m) {
    int64_t lim = m.cfg().positional == "learned" ? m.cfg().max_seq_len : 0;
    for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
        const ModelConfig& c = m.layer_cfg(l);
        if (c.positional != "rope") continue;
        const int64_t r = detail::rope_table(c).max_len;
        if (r > 0) lim = lim > 0 ? std::min(lim, r) : r;
    }
    return lim;
}

// Prompt lookup: the (at most k) tokens that followed the most recent earlier occurrence of ctx's last n
// tokens, for n = max_n down to 1; none if no suffix occurs earlier.
Tokens lookup_draft(const Tokens& ctx, int64_t max_n, int64_t k) {
    const int64_t len = static_cast<int64_t>(ctx.size());
    for (int64_t n = std::min(max_n, len - 1); n >= 1 && k > 0; --n) {
        const int64_t s = len - n;  // the suffix: ctx[s, len)
        for (int64_t i = s - 1; i >= 0; --i) {
            if (!std::equal(ctx.begin() + s, ctx.end(), ctx.begin() + i)) continue;
            const int64_t from = i + n, to = std::min(len, from + k);
            return Tokens(ctx.begin() + from, ctx.begin() + to);
        }
    }
    return {};
}

// generate() with speculative steps (options.speculative.draft_tokens > 0). `cache` holds every token of
// ctx but the last, which each step runs with the proposed tokens: row i of the step's logits is the
// model's distribution after its i-th token, so for proposal i (accepted with probability min(1, p/q), else
// replaced by a draw from max(0, p - q)) and, when all are kept, for one more token of the model's own.
Tokens generate_speculative(const Model& model, const Tokens& prompt, const GenerateOptions& o, const Hooks* h,
                            TraceRecorder* rec, std::vector<TraceStep>* trace, Sampler& sample, const std::set<int64_t>& stop) {
    const SpeculativeOptions& so = o.speculative;
    const Model* draft = so.draft;
    SpeculativeStats stats;
    std::optional<Sampler> draft_sample;
    if (draft) {  // its own generator, so the model's draws don't depend on the draft's
        GenerateOptions d = o;
        d.seed = o.seed ^ 0x9E3779B97F4A7C15ull;
        draft_sample.emplace(d);
    }
    const int64_t limit = position_limit(model), draft_limit = draft ? position_limit(*draft) : 0;
    Tokens ctx = prompt, out;
    KVCache cache, draft_cache;
    if (prompt.size() > 1) model.forward_cached(Tokens(prompt.begin(), prompt.end() - 1), cache, h, /*last_only=*/true);
    if (rec) rec->all_rows = true;
    std::vector<double> p, w;
    std::vector<std::vector<double>> q;  // the draft's distribution for each proposal
    // Proposals per step: fewer after a rejection, more after a step that kept them all (as Hugging Face's
    // assisted generation does), so text the draft can't predict costs little more than plain steps.
    int64_t want = so.draft_tokens;
    auto finish = [&]() {
        if (so.stats) *so.stats = stats;
        return out;
    };
    for (;;) {
        const int64_t pos = cache.size();  // ctx.back()'s position
        // Proposals: no more than the tokens still wanted after this step's own, nor past the positions
        // either model can run (where generating without speculation would stop at the same token).
        int64_t k = std::min(want, o.max_new_tokens - static_cast<int64_t>(out.size()) - 1);
        if (limit > 0) k = std::min(k, limit - 1 - pos);
        if (draft && draft_limit > 0) k = std::min(k, draft_limit + 1 - static_cast<int64_t>(ctx.size()));
        Tokens d;
        if (k > 0 && draft) {
            if (q.size() < static_cast<size_t>(k)) q.resize(static_cast<size_t>(k));
            const int64_t dv = draft->cfg().vocab_size;
            Tensor dl = draft->forward_cached(Tokens(ctx.begin() + draft_cache.size(), ctx.end()), draft_cache, nullptr, true);
            for (int64_t i = 0; i < k; ++i) {
                draft_sample->distribution(dl.ptr(), dv, q[static_cast<size_t>(i)]);
                d.push_back(draft_sample->draw(q[static_cast<size_t>(i)]));
                // A token past the model's vocabulary (a draft with more rows) ends the proposals: the model
                // can't run it, and it is rejected below, as its probability under the model is 0.
                if (d.back() >= model.cfg().vocab_size) break;
                if (i + 1 < k) dl = draft->forward_cached({d.back()}, draft_cache, nullptr, true);
            }
        } else if (k > 0) {
            d = lookup_draft(ctx, so.ngram, k);
        }
        Tokens in{ctx.back()};
        for (int64_t t : d)
            if (t < model.cfg().vocab_size) in.push_back(t);
        const Tensor L = model.forward_cached(in, cache, h, /*last_only=*/false);
        const int64_t V = L.shape[1], n_ctx = static_cast<int64_t>(ctx.size());
        ++stats.steps;
        stats.drafted += static_cast<int64_t>(d.size());
        auto emit = [&](int64_t t, int64_t row) {
            const float* lg = L.ptr() + row * V;
            if (trace) trace->push_back(rec->step(lg, V, t, pos + row, row));
            out.push_back(t);
            ctx.push_back(t);
            bool more = !stop.count(t) && static_cast<int64_t>(out.size()) < o.max_new_tokens;
            if (o.on_token && !o.on_token(t)) more = false;
            return more;
        };
        size_t kept = 0;
        bool more = true;
        for (; kept < d.size() && more; ++kept) {
            const int64_t t = d[kept], row = static_cast<int64_t>(kept);
            sample.distribution(L.ptr() + row * V, V, p);
            const std::vector<double>* qi = draft ? &q[kept] : nullptr;
            const double pt = t < V ? p[static_cast<size_t>(t)] : 0, qt = qi ? (*qi)[static_cast<size_t>(t)] : 1;
            if (pt > 0 && (pt >= qt || sample.uniform() * qt < pt)) {
                ++stats.accepted;
                more = emit(t, row);
                continue;
            }
            w = p;  // rejected: draw from max(0, p - q)
            if (qi)
                for (size_t j = 0; j < w.size() && j < qi->size(); ++j) w[j] = std::max(0.0, w[j] - (*qi)[j]);
            else if (t < V)
                w[static_cast<size_t>(t)] = 0;
            double total = 0;
            for (double x : w) total += x;
            more = emit(sample.draw(total > 0 ? w : p), row);
            break;
        }
        if (!more) return finish();
        if (kept < d.size()) want = std::max<int64_t>(1, want - 1);
        else if (!d.empty()) want = std::min(so.draft_tokens, want + 2);
        if (kept == d.size()) {  // every proposal kept: one more token, from the last row
            const int64_t row = static_cast<int64_t>(d.size());
            if (!emit(sample(L.ptr() + row * V, V), row)) return finish();
        }
        // The cache keeps ctx but its new last token: the step's first token and the proposals kept.
        cache.truncate(pos + 1 + static_cast<int64_t>(kept));
        if (draft) draft_cache.truncate(std::min(draft_cache.size(), n_ctx + static_cast<int64_t>(kept)));
    }
}

}  // namespace

Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options, const Hooks* hooks,
                std::vector<TraceStep>* trace) {
    Sampler sample(options);
    if (prompt.empty()) throw std::invalid_argument("generate needs a prompt of at least one token");
    std::set<int64_t> stop(options.stop_tokens.begin(), options.stop_tokens.end());
    if (options.stop_at_eos) stop.insert(model.eos_tokens().begin(), model.eos_tokens().end());
    Tokens out;
    if (options.speculative.stats) *options.speculative.stats = {};
    if (options.max_new_tokens == 0) return out;
    std::optional<TraceRecorder> rec;
    Hooks all;  // the caller's hooks, then the trace's captures (so they see what the hooks left)
    if (hooks) all.extend(*hooks);
    if (trace) {
        rec.emplace(model, options.trace);
        TraceRecorder* r = &*rec;
        all.add_capture([r](const std::string& n) { return r->points.count(n) > 0; },
                        [r](const std::string& n, Tensor& a) { r->capture(n, a); });
    }
    const Hooks* h = all.empty() ? nullptr : &all;
    if (options.speculative.draft_tokens > 0)
        return generate_speculative(model, prompt, options, h, rec ? &*rec : nullptr, trace, sample, stop);
    KVCache cache;
    Tensor logits = model.forward_cached(prompt, cache, h, /*last_only=*/true);
    for (;;) {
        const int64_t t = sample(logits.ptr(), logits.shape[1]);
        if (trace) trace->push_back(rec->step(logits.ptr(), logits.shape[1], t, cache.size() - 1));
        out.push_back(t);
        bool more = !stop.count(t) && static_cast<int64_t>(out.size()) < options.max_new_tokens;
        if (options.on_token && !options.on_token(t)) more = false;
        if (!more) return out;
        logits = model.forward_cached({t}, cache, h, /*last_only=*/true);
    }
}

namespace {

// Hooks for a batched pass that give each sequence its own hooks: every hook point's stacked activation
// split into each sequence's rows (dimension 1 for hook_q / hook_k / hook_v, (H, rows, D); dimension 0
// otherwise), that sequence's hooks run on its slice, and what they leave written back. A hook point that
// only captures stays a capture (a GPU pass doesn't stop for it).
struct PerSequenceHooks {
    std::vector<Hooks> of;  // per sequence
    std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> segs;  // this pass: (sequence, (first row, rows))
    Hooks all;

    static bool heads_first(const std::string& n) {
        auto ends = [&](const char* e) { const size_t l = std::strlen(e); return n.size() >= l && n.compare(n.size() - l, l, e) == 0; };
        return ends("attn.hook_q") || ends("attn.hook_k") || ends("attn.hook_v");
    }
    void each(const std::string& name, Tensor& act, bool write) {
        const int dim = heads_first(name) ? 1 : 0;
        const int64_t outer = dim ? act.shape[0] : 1, rows = act.shape[dim];
        int64_t inner = 1;
        for (int64_t k = dim + 1; k < act.dim(); ++k) inner *= act.shape[k];
        for (const auto& sg : segs) {
            const Hooks& h = of[sg.first];
            if (!h.matches(name)) continue;
            const int64_t r0 = sg.second.first, n = sg.second.second;
            Shape shape = act.shape;
            shape[dim] = n;
            Tensor slice(shape);
            for (int64_t o = 0; o < outer; ++o)
                std::copy(act.ptr() + (o * rows + r0) * inner, act.ptr() + (o * rows + r0 + n) * inner, slice.ptr() + o * n * inner);
            h.run(name, slice);
            if (!write) continue;
            if (slice.shape != shape) throw std::runtime_error(name + ": a hook changed the activation's shape");
            for (int64_t o = 0; o < outer; ++o)
                std::copy(slice.ptr() + o * n * inner, slice.ptr() + (o + 1) * n * inner, act.ptr() + (o * rows + r0) * inner);
        }
    }
    explicit PerSequenceHooks(size_t n) : of(n) {
        all.add_filter([this](const std::string& name) {
                           for (const Hooks& h : of)
                               if (h.changes(name)) return true;
                           return false;
                       },
                       [this](const std::string& name, Tensor& act) { each(name, act, true); });
        all.add_capture([this](const std::string& name) {
                            bool any = false;
                            for (const Hooks& h : of) {
                                if (h.changes(name)) return false;  // handled above
                                any = any || h.matches(name);
                            }
                            return any;
                        },
                        [this](const std::string& name, Tensor& act) { each(name, act, false); });
    }
};

}  // namespace

std::vector<Tokens> generate_batch(const Model& model, const std::vector<Tokens>& prompts, const GenerateOptions& options,
                                   const std::vector<const Hooks*>& hooks, std::vector<std::vector<TraceStep>>* traces) {
    const size_t n = prompts.size();
    if (!hooks.empty() && hooks.size() != n) throw std::invalid_argument("generate_batch: hooks must be empty or one per prompt");
    if (options.on_token) throw std::invalid_argument("generate_batch: on_token is not used here; give generate() one prompt to follow it");
    if (options.speculative.draft_tokens > 0)
        throw std::invalid_argument("generate_batch: speculative decoding is not supported here; give generate() one prompt for it");
    if (traces && options.trace.attention_top > 0)
        throw std::invalid_argument("generate_batch: a trace's attention needs each sequence's pattern; use generate() for that");
    for (const Tokens& p : prompts)
        if (p.empty()) throw std::invalid_argument("generate_batch needs every prompt to have at least one token");
    std::vector<Sampler> samplers;
    for (size_t i = 0; i < n; ++i) {
        GenerateOptions o = options;
        o.seed = options.seed + i;
        samplers.emplace_back(o);
    }
    std::set<int64_t> stop(options.stop_tokens.begin(), options.stop_tokens.end());
    if (options.stop_at_eos) stop.insert(model.eos_tokens().begin(), model.eos_tokens().end());
    std::vector<Tokens> out(n);
    if (n == 0 || options.max_new_tokens == 0) return out;

    PerSequenceHooks ph(n);
    std::vector<std::optional<TraceRecorder>> recs(n);
    bool any_hooks = false;
    for (size_t i = 0; i < n; ++i) {
        if (!hooks.empty() && hooks[i]) {
            ph.of[i].extend(*hooks[i]);
            any_hooks = true;
        }
        if (traces) {
            recs[i].emplace(model, options.trace);
            TraceRecorder* r = &*recs[i];
            ph.of[i].add_capture([r](const std::string& nm) { return r->points.count(nm) > 0; },
                                 [r](const std::string& nm, Tensor& a) { r->capture(nm, a); });
            any_hooks = true;
        }
    }
    if (traces) traces->assign(n, {});

    KVBatch batch(static_cast<int64_t>(n));
    std::vector<Tokens> step = prompts;
    for (;;) {
        ph.segs.clear();
        int64_t row = 0;
        for (size_t i = 0; i < n; ++i)
            if (!step[i].empty()) {
                ph.segs.push_back({static_cast<int64_t>(i), {row, static_cast<int64_t>(step[i].size())}});
                row += static_cast<int64_t>(step[i].size());
            }
        if (ph.segs.empty()) return out;
        const std::vector<Tensor> logits = model.forward_cached(step, batch, any_hooks ? &ph.all : nullptr, /*last_only=*/true);
        for (size_t i = 0; i < n; ++i) {
            if (step[i].empty()) continue;
            const Tensor& lg = logits[i];
            const int64_t t = samplers[i](lg.ptr(), lg.shape[1]);
            if (traces) (*traces)[i].push_back(recs[i]->step(lg.ptr(), lg.shape[1], t, batch[static_cast<int64_t>(i)].size() - 1));
            out[i].push_back(t);
            const bool more = !stop.count(t) && static_cast<int64_t>(out[i].size()) < options.max_new_tokens;
            step[i] = more ? Tokens{t} : Tokens{};
        }
    }
}

}  // namespace tfi
