#include "tfi/generate.h"

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
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
    // (A temperature whose reciprocal overflows a float leaves the largest logit alone, as greedy does.)
    if (temperature_ == 0 || top_k_ == 1 || top == INFINITY || 1.0 / temperature_ > std::numeric_limits<float>::max())
        return false;
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
// tokens, for n = max_n down to 1; none if no suffix occurs earlier. Copying goes on past the end of ctx
// into the proposals themselves: a match d tokens back proposes the text repeating with period d (X Y X Y
// ... from a match 2 back), not just the d tokens there are.
Tokens lookup_draft(const Tokens& ctx, int64_t max_n, int64_t k) {
    const int64_t len = static_cast<int64_t>(ctx.size());
    for (int64_t n = std::min(max_n, len - 1); n >= 1 && k > 0; --n) {
        const int64_t s = len - n;  // the suffix: ctx[s, len)
        for (int64_t i = s - 1; i >= 0; --i) {
            if (!std::equal(ctx.begin() + s, ctx.end(), ctx.begin() + i)) continue;
            Tokens out;
            for (int64_t j = i + n; static_cast<int64_t>(out.size()) < k; ++j)
                out.push_back(j < len ? ctx[static_cast<size_t>(j)] : out[static_cast<size_t>(j - len)]);
            return out;
        }
    }
    return {};
}

// Whether a and b hold the same values, up to the rounding of redoing an edit on the input x (adding a
// vector twice, say, can differ in the last bits).
bool same_values(const Tensor& a, const Tensor& b, const Tensor& x) {
    if (a.shape != b.shape) return false;
    const bool has_x = x.shape == a.shape;
    for (int64_t i = 0; i < a.numel(); ++i) {
        const float u = a.ptr()[i], v = b.ptr()[i];
        if (u == v) continue;
        const float w = has_x ? x.ptr()[i] : 0.f;
        if (!(std::fabs(u - v) <= 1e-6f * (std::fabs(u) + std::fabs(v) + std::fabs(w)) + 1e-9f)) return false;
    }
    return true;
}

// What a hook point's hooks will do to a token's row, predicted from what they did to earlier tokens' rows.
// Each distinct edit gets an id (the same row, some values overwritten, a scale or an offset; further
// calls rule out the explanations that stop fitting), and the next id comes from the sequence of ids: the
// period that has held longest, or else what followed the same last ids. Every observed call scores the
// prediction, so it keeps learning while nothing is proposed and is only used once it has been right.
class EditPredictor {
  public:
    // The ids of the next n calls, -1 from the first that can't be predicted.
    std::vector<int> plan(size_t n) const {
        std::vector<int> h = history_, out;
        const int p = period(h);
        while (out.size() < n) {
            const int j = p ? h[h.size() - static_cast<size_t>(p)] : after(h);
            if (j < 0) break;
            out.push_back(j);
            h.push_back(j);
        }
        out.resize(n, -1);
        return out;
    }
    Tensor predict(int id, const Tensor& x) const {
        if (id < 0) return x;
        const Edit& e = edits_[static_cast<size_t>(id)];
        return apply(e, e.kinds & (~e.kinds + 1), x);  // the first explanation left
    }
    // How many of the next n calls can be predicted, if predictions have been earning trust.
    int64_t confident(size_t n) const {
        if (score_ < 1) return 0;
        int64_t c = 0;
        for (int j : plan(n)) {
            if (j < 0) break;
            ++c;
        }
        return c;
    }
    void observe(const Tensor& x, const Tensor& y) {
        const int guess = plan(1)[0], id = match(x, y);
        if (guess >= 0) score_ = guess == id ? std::min(score_ + 1, 8) : std::max(score_ - 2, -8);
        for (size_t o = 1; o <= std::min(kOrder, history_.size()); ++o) {
            const std::vector<int> context(history_.end() - static_cast<std::ptrdiff_t>(o), history_.end());
            auto it = followers_.find(context);
            if (it == followers_.end()) followers_[context] = {id, 1};
            else if (it->second.first == id) it->second.second = std::min(it->second.second + 1, 3);
            else if (--it->second.second <= 0) it->second = {id, 1};
        }
        if (followers_.size() > kFollowers) followers_.clear();
        history_.push_back(id);
        if (history_.size() > 2 * kHistory) history_.erase(history_.begin(), history_.end() - static_cast<std::ptrdiff_t>(kHistory));
    }

  private:
    enum Kind : unsigned { kSame = 1, kOverwrite = 2, kScale = 4, kOffset = 8 };
    struct Edit {
        Tensor in, out;
        unsigned kinds;
    };
    static constexpr size_t kEdits = 16, kOrder = 8, kHistory = 256, kFollowers = 4096;
    static constexpr int kMaxPeriod = 32;

    static Tensor apply(const Edit& e, unsigned kind, const Tensor& x) {
        if (kind == kSame || e.in.shape != x.shape) return x;
        Tensor y = x;
        const float* in = e.in.ptr();
        const float* out = e.out.ptr();
        for (int64_t i = 0; i < y.numel(); ++i) {
            if (kind == kOverwrite) {
                if (out[i] != in[i]) y.ptr()[i] = out[i];
            } else if (kind == kScale) {
                if (in[i] != 0) y.ptr()[i] *= out[i] / in[i];
            } else {
                y.ptr()[i] += out[i] - in[i];
            }
        }
        return y;
    }
    int match(const Tensor& x, const Tensor& y) {
        for (size_t j = 0; j < edits_.size(); ++j) {
            Edit& e = edits_[j];
            if (e.in.shape != x.shape && !(e.kinds & kSame)) continue;
            unsigned left = 0;
            for (unsigned kind : {kSame, kOverwrite, kScale, kOffset})
                if ((e.kinds & kind) && same_values(apply(e, kind, x), y, x)) left |= kind;
            if (left) {
                e.kinds = left;
                return static_cast<int>(j);
            }
        }
        if (edits_.size() == kEdits) return -1;
        edits_.push_back({x, y, same_values(x, y, x) ? unsigned(kSame) : unsigned(kOverwrite | kScale | kOffset)});
        return static_cast<int>(edits_.size()) - 1;
    }
    // The period p that has repeated for the longest (at least one whole period, and 3 calls); 0: none.
    static int period(const std::vector<int>& h) {
        const int64_t n = static_cast<int64_t>(h.size());
        int best = 0;
        int64_t best_run = 0;
        for (int p = 1; p <= kMaxPeriod && p < n; ++p) {
            int64_t run = 0;
            for (int64_t i = n - 1; i - p >= 0 && h[i] >= 0 && h[i] == h[i - p]; --i) ++run;
            if (run >= std::max<int64_t>(3, p) && run > best_run) {
                best = p;
                best_run = run;
            }
        }
        return best;
    }
    // What has reliably followed the longest suffix of h; -1: nothing has.
    int after(const std::vector<int>& h) const {
        for (size_t o = std::min(kOrder, h.size()); o >= 1; --o) {
            const auto it = followers_.find(std::vector<int>(h.end() - static_cast<std::ptrdiff_t>(o), h.end()));
            if (it != followers_.end() && it->second.second >= 2) return it->second.first;
        }
        return -1;
    }

    std::vector<Edit> edits_;
    std::vector<int> history_;
    std::map<std::vector<int>, std::pair<int, int>> followers_;  // last ids -> (the id after them, confidence)
    int score_ = 0;
};

}  // namespace

// One decoding, advanced a token at a time. Speculation happens here and stays here: a pass computes the
// current token with guesses after it, but what the outside sees (delivered tokens, hook calls, the trace)
// is only ever confirmed tokens, in order. Hooks don't run on a guess. In a speculative pass the current
// token's real hooks run, and each guess's row gets the predicted edit; when a guess is confirmed its
// recorded inputs go through the real hooks, in the order the pass reached them. If they do what was predicted,
// its row stands. If not, the guess is computed again with the real outputs from the block where it went
// wrong (the blocks below are right, and so are those of the later guesses, which stay: each, once
// confirmed, is computed again from that block too, as the guesses before it changed there).
struct DecodeRuntime::Impl {
    enum class PassKind { Plain, Speculative, Rerun };
    struct Point {
        std::string name;
        bool edits = false;  // else only captures
        int row_dim = 0;     // 1 for hook_q / hook_k / hook_v / hook_pattern
        bool pattern = false;
        int64_t block = -1;   // its block (-1: before the first, hook_embed and hook_pos_embed; the last for hook_final)
        int64_t resume = -1;  // the block to compute a token again from when this point's output was mispredicted
        EditPredictor predictor;
        std::vector<Tensor> in, predicted;  // this pass's guesses (rows 1, 2, ...)
        bool done = false;                  // called for the token being confirmed
        Tensor done_out;
    };

    const Model& model;
    const Tokens prompt;
    const GenerateOptions o;
    const Hooks* user;
    std::vector<TraceStep>* trace;
    Sampler sample;
    std::set<int64_t> stop;
    std::optional<TraceRecorder> rec;
    Hooks plain, routed;  // the caller's hooks and the trace as they are; or through this runtime
    const Hooks* plain_hooks = nullptr;
    std::map<std::string, Point> points;  // with speculation: the hook points the caller's hooks are on
    std::vector<Point*> fired;            // in the order this pass reached them
    PassKind kind = PassKind::Plain;
    // Recomputing a token from a block needs its input to the block (hook_resid_pre): from the point there if
    // the caller's hooks are on it, else recorded from each speculative pass.
    bool resume = false;                   // tokens are computed again from a block rather than from the start
    std::map<std::string, int64_t> record;  // hook_resid_pre name -> block
    std::map<int64_t, Tensor> recorded;     // block -> this pass's hook_resid_pre
    int64_t n_layers = 0;

    const int64_t K;
    const bool speculate;
    const Model* draft;
    std::optional<Sampler> draft_sample;
    int64_t limit = 0, draft_limit = 0;
    KVCache cache, draft_cache;
    Tokens ctx, proposals;
    std::vector<double> p, w;
    std::vector<std::vector<double>> q;  // the draft's distribution for each proposal
    SpeculativeStats stats;
    int64_t emitted = 0;
    bool started = false, finished = false;

    // The pass in progress.
    Tensor L;
    int64_t V = 0, pos = 0, row = 0, kept = 0, proposed = 0, step_ctx = 0;
    bool pass_open = false;  // a proposal was accepted: its row is next
    int64_t stale_from = 0;  // the lowest block where the pass's later rows used something that changed

    // How much to propose, from outcomes alone (the number of proposals affects which sampled tokens come out, so
    // neither a clock nor the device may decide it).
    int64_t want = 0, misses = 0, pause = 0, cooldown = 1;

    Impl(const Model& m, const Tokens& t, const GenerateOptions& options, const Hooks* hooks, std::vector<TraceStep>* tr)
        : model(m), prompt(t), o(options), user(hooks), trace(tr), sample(o), K(o.speculative.draft_tokens), speculate(K > 0),
          draft(o.speculative.draft), ctx(t), want(K) {
        if (prompt.empty()) throw std::invalid_argument("generate needs a prompt of at least one token");
        stop.insert(o.stop_tokens.begin(), o.stop_tokens.end());
        if (o.stop_at_eos) stop.insert(model.eos_tokens().begin(), model.eos_tokens().end());
        if (o.speculative.stats) *o.speculative.stats = {};
        if (o.max_new_tokens == 0) {
            finished = true;
            return;
        }
        auto add_trace = [this](Hooks& h) {
            TraceRecorder* r = &*rec;
            h.add_capture([r](const std::string& n) { return r->points.count(n) > 0; },
                          [r](const std::string& n, Tensor& a) { r->capture(n, a); });
        };
        if (user) plain.extend(*user);
        if (trace) {
            rec.emplace(model, o.trace);
            add_trace(plain);
        }
        plain_hooks = plain.empty() ? nullptr : &plain;
        if (!speculate) return;
        limit = position_limit(model);
        if (draft) {
            draft_limit = position_limit(*draft);
            GenerateOptions d = o;  // its own generator, so the model's draws don't depend on the draft's
            d.seed = o.seed ^ 0x9E3779B97F4A7C15ull;
            draft_sample.emplace(d);
        }
        if (!user) return;
        n_layers = model.cfg().n_layers;
        for (const std::string& n : model.hook_names()) {
            if (!user->matches(n)) continue;
            Point s;
            s.name = n;
            s.edits = user->changes(n);
            auto ends = [&](const char* e) { const size_t l = std::strlen(e); return n.size() >= l && n.compare(n.size() - l, l, e) == 0; };
            s.pattern = ends("attn.hook_pattern");
            s.row_dim = s.pattern || ends("attn.hook_q") || ends("attn.hook_k") || ends("attn.hook_v") ? 1 : 0;
            if (n.compare(0, 7, "blocks.") == 0) s.block = std::stoll(n.substr(7));
            else if (n == "hook_final") s.block = n_layers - 1;
            s.resume = ends(".hook_resid_post") && s.block + 1 < n_layers ? s.block + 1 : s.block;
            points.emplace(n, std::move(s));
        }
        if (points.empty()) return;
        // (A trace keeps each pass's activations whole: its tokens are computed again from the start.)
        resume = !trace;
        for (const auto& kv : points)
            for (int64_t b : {kv.second.block, kv.second.resume}) {
                const std::string n = "blocks." + std::to_string(b) + ".hook_resid_pre";
                if (resume && b >= 0 && !points.count(n)) record[n] = b;
            }
        routed.add_filter([this](const std::string& n) { const auto it = points.find(n); return it != points.end() && it->second.edits; },
                          [this](const std::string& n, Tensor& a) { at_point(n, a); });
        routed.add_capture([this](const std::string& n) { const auto it = points.find(n); return it != points.end() && !it->second.edits; },
                           [this](const std::string& n, Tensor& a) { at_point(n, a); });
        if (!record.empty())
            routed.add_capture([this](const std::string& n) { return record.count(n) > 0; },
                               [this](const std::string& n, Tensor& a) {
                                   if (kind == PassKind::Speculative) recorded[record.at(n)] = a;
                               });
        if (trace) add_trace(routed);
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    std::optional<int64_t> next() {
        if (finished) return std::nullopt;
        try {
            const int64_t t = started ? advance() : first_token();
            started = true;
            ctx.push_back(t);
            ++emitted;
            if (stop.count(t) || emitted >= o.max_new_tokens) finished = true;
            if (speculate && o.speculative.stats) *o.speculative.stats = stats;
            return t;
        } catch (...) {
            finished = true;  // a hook or the model failed partway: nothing to continue from
            throw;
        }
    }

    // The prompt's pass, the same with and without speculation.
    int64_t first_token() {
        L = model.forward_cached(prompt, cache, plain_hooks, /*last_only=*/true);
        V = L.shape[1];
        if (speculate) stats.steps = 1;
        const int64_t t = sample(L.ptr(), V);
        if (trace) trace->push_back(rec->step(L.ptr(), V, t, cache.size() - 1));
        if (rec && speculate) rec->all_rows = true;
        return t;
    }

    // Everything up to the next token. The last token delivered is ctx.back(); only now do its hooks run.
    int64_t advance() {
        if (pass_open) {
            ++row;
            settle_row();
        } else {
            begin_pass();
        }
        return choose();
    }

    void begin_pass() {
        pos = cache.size();
        step_ctx = static_cast<int64_t>(ctx.size());
        row = kept = proposed = 0;
        stale_from = n_layers;
        proposals.clear();
        if (!speculate) {
            L = model.forward_cached({ctx.back()}, cache, plain_hooks, /*last_only=*/true);
            return;
        }
        // A token the draft has no row for (a smaller vocabulary): it can't read this text from here on, so
        // the model continues alone.
        if (draft && std::any_of(ctx.begin() + draft_cache.size(), ctx.end(),
                                 [&](int64_t t) { return t >= draft->cfg().vocab_size; }))
            draft = nullptr;
        propose();
        Tokens in{ctx.back()};
        for (int64_t t : proposals)
            if (t < model.cfg().vocab_size) in.push_back(t);
        fired.clear();
        recorded.clear();
        kind = in.size() > 1 ? PassKind::Speculative : PassKind::Plain;
        L = model.forward_cached(in, cache, points.empty() ? plain_hooks : &routed, /*last_only=*/false);
        ++stats.steps;
        proposed = static_cast<int64_t>(proposals.size());
        stats.drafted += proposed;
    }

    // Proposals: no more than the tokens still wanted after this step's own, nor past the positions either
    // model can run (where generating without speculation would stop at the same token), nor than the hooks'
    // edits can be predicted for, nor past a stop token.
    void propose() {
        int64_t k = std::min(want, o.max_new_tokens - emitted - 1);
        if (pause > 0) {
            --pause;
            k = 0;
        }
        if (limit > 0) k = std::min(k, limit - 1 - pos);
        if (draft && draft_limit > 0) k = std::min(k, draft_limit + 1 - static_cast<int64_t>(ctx.size()));
        // An edit that can't be predicted yet still allows proposals if a token it gets wrong is computed again
        // from the upper half of the blocks only.
        for (const auto& kv : points)
            if (k > 0 && kv.second.edits && !(resume && 2 * kv.second.resume >= n_layers))
                k = std::min(k, kv.second.predictor.confident(static_cast<size_t>(k) + 1) - 1);
        if (k <= 0) return;
        if (draft) {
            if (q.size() < static_cast<size_t>(k)) q.resize(static_cast<size_t>(k));
            const int64_t dv = draft->cfg().vocab_size;
            Tensor dl = draft->forward_cached(Tokens(ctx.begin() + draft_cache.size(), ctx.end()), draft_cache, nullptr, true);
            for (int64_t i = 0; i < k; ++i) {
                draft_sample->distribution(dl.ptr(), dv, q[static_cast<size_t>(i)]);
                proposals.push_back(draft_sample->draw(q[static_cast<size_t>(i)]));
                // A token past the model's vocabulary (a draft with more rows) ends the proposals: the model
                // can't run it, and it is rejected below, as its probability under the model is 0.
                if (proposals.back() >= model.cfg().vocab_size || stop.count(proposals.back())) break;
                if (i + 1 < k) dl = draft->forward_cached({proposals.back()}, draft_cache, nullptr, true);
            }
        } else if (!o.speculative.draft) {
            proposals = lookup_draft(ctx, o.speculative.ngram, k);
            const auto s = std::find_if(proposals.begin(), proposals.end(), [&](int64_t t) { return stop.count(t) > 0; });
            if (s != proposals.end()) proposals.erase(s + 1, proposals.end());
        }
    }

    // The token after row `row`: the proposal there if the acceptance rule keeps it (min(1, p/q)), else a draw
    // from max(0, p - q); past the last proposal, the model's own.
    int64_t choose() {
        V = L.shape[1];
        const float* lg = L.ptr() + row * V;
        int64_t t;
        bool accepted = false;
        if (row < static_cast<int64_t>(proposals.size())) {
            const size_t r = static_cast<size_t>(row);
            t = proposals[r];
            sample.distribution(lg, V, p);
            const std::vector<double>* qi = draft ? &q[r] : nullptr;
            const double pt = t < V ? p[static_cast<size_t>(t)] : 0, qt = qi ? (*qi)[static_cast<size_t>(t)] : 1;
            accepted = pt > 0 && (pt >= qt || sample.uniform() * qt < pt);
            if (!accepted) {
                w = p;
                if (qi)
                    for (size_t j = 0; j < w.size() && j < qi->size(); ++j) w[j] = std::max(0.0, w[j] - (*qi)[j]);
                else if (t < V)
                    w[static_cast<size_t>(t)] = 0;
                double total = 0;
                for (double x : w) total += x;
                t = sample.draw(total > 0 ? w : p);
            }
        } else {
            t = sample(lg, V);
        }
        if (trace) trace->push_back(rec->step(lg, V, t, pos + row, row));
        if (accepted) {
            ++stats.accepted;
            ++kept;
            pass_open = true;
        } else {
            end_pass();
        }
        return t;
    }

    // The cache keeps the rows that were used, so it holds the text but its new last token.
    void end_pass() {
        pass_open = false;
        if (!speculate) return;
        cache.truncate(pos + row + 1);
        if (draft) draft_cache.truncate(std::min(draft_cache.size(), step_ctx + kept));
        if (proposed == 0) return;
        want = kept < proposed ? std::max<int64_t>(1, want - 1) : std::min(K, want + 2);
        if (kept > 0) {
            misses = 0;
            cooldown = 1;
        } else if (++misses >= 2) {  // proposals aren't paying for themselves: stop for a while, longer each time
            pause = cooldown;
            cooldown = std::min<int64_t>(32, cooldown * 2);
            misses = 0;
            want = 1;
        }
    }

    // Row `row` is now a confirmed token. Its recorded activations go through the real hooks, in order, up to
    // the first that doesn't do what was predicted; those from stale blocks don't (the row is computed again
    // from there, and its hooks run then). It stands if nothing was mispredicted or stale; else it is
    // computed again from the lowest block that was (or the start, without its input there).
    void settle_row() {
        const Point* wrong = confirm_row();
        if (!wrong && stale_from >= n_layers) return;
        int64_t b = stale_from;
        for (const Point* s : fired)
            if (!s->done) b = std::min(b, s->block);
        Tensor x;
        if (wrong && wrong->resume > wrong->block && wrong->resume <= b) {  // hook_resid_post: its output is the next block's input
            b = wrong->resume;
            x = wrong->done_out;
        } else {
            if (wrong) b = std::min(b, wrong->block);
            x = input_to(b);
        }
        if (!resume || b < 0 || x.numel() == 0) return recompute_row();
        ++stats.reruns;
        kind = PassKind::Rerun;
        const Tensor lg = model.rerun_cached(b, pos + row, std::move(x), cache, &routed);
        std::copy(lg.ptr(), lg.ptr() + lg.numel(), L.ptr() + row * L.shape[1]);
        stale_from = std::min(stale_from, b);
    }

    // Row `row`'s input to block b, (1, d_model), from this pass; empty if not kept.
    Tensor input_to(int64_t b) const {
        if (b < 0 || b >= n_layers) return {};
        const std::string n = "blocks." + std::to_string(b) + ".hook_resid_pre";
        const auto p = points.find(n);
        if (p != points.end()) return static_cast<size_t>(row) <= p->second.in.size() ? p->second.in[static_cast<size_t>(row) - 1] : Tensor();
        const auto it = recorded.find(b);
        if (it == recorded.end() || it->second.dim() != 2 || it->second.shape[0] <= row) return {};
        const int64_t d = it->second.shape[1];
        return Tensor({1, d}, std::vector<float>(it->second.ptr() + row * d, it->second.ptr() + (row + 1) * d));
    }

    // The first point whose real hooks didn't do what was predicted, or nullptr.
    const Point* confirm_row() {
        const size_t r = static_cast<size_t>(row) - 1;
        for (Point* s : fired) s->done = false;
        for (Point* s : fired) {
            if (s->block >= stale_from) continue;
            const Tensor& x = s->in[r];
            Tensor y = x;
            user->run(s->name, y);
            s->done = true;
            if (!s->edits) continue;
            if (y.shape != x.shape) throw std::runtime_error(s->name + ": a hook changed the activation's shape");
            s->predictor.observe(x, y);
            const bool right = same_values(y, s->predicted[r], x);
            s->done_out = std::move(y);
            if (!right) return s;
        }
        return nullptr;
    }

    // Row `row` alone, with the hook outputs confirm_row got: the guesses after it depended on the wrong value.
    void recompute_row() {
        ++stats.reruns;
        cache.truncate(pos + row);
        pos += row;
        row = 0;
        stale_from = n_layers;
        proposals.clear();
        kind = PassKind::Rerun;
        L = model.forward_cached({ctx.back()}, cache, &routed, /*last_only=*/false);
    }

    void at_point(const std::string& name, Tensor& act) {
        Point& s = points.at(name);
        if (kind == PassKind::Plain) {
            call(s, act);
        } else if (kind == PassKind::Rerun) {
            if (!s.done) call(s, act);
            else if (s.edits) act = s.done_out;
        } else {
            speculative_point(s, act);
        }
    }

    // The real hooks on one confirmed row, in place.
    void call(Point& s, Tensor& row_act) {
        if (!s.edits) {
            user->run(s.name, row_act);
            return;
        }
        const Tensor x = row_act;
        user->run(s.name, row_act);
        if (row_act.shape != x.shape) throw std::runtime_error(s.name + ": a hook changed the activation's shape");
        s.predictor.observe(x, row_act);
    }

    void speculative_point(Point& s, Tensor& act) {
        const size_t dim = static_cast<size_t>(s.row_dim);
        const int64_t outer = s.row_dim ? act.shape[0] : 1, n = act.shape[dim];
        int64_t inner = 1;
        for (size_t k = dim + 1; k < act.shape.size(); ++k) inner *= act.shape[k];
        auto width = [&](int64_t r) { return s.pattern ? pos + r + 1 : inner; };  // a pattern's row: the keys up to its token
        auto get = [&](int64_t r) {
            Shape shape = act.shape;
            shape[dim] = 1;
            if (s.pattern) shape[2] = width(r);
            Tensor t(shape);
            for (int64_t o2 = 0; o2 < outer; ++o2)
                std::copy(act.ptr() + (o2 * n + r) * inner, act.ptr() + (o2 * n + r) * inner + width(r), t.ptr() + o2 * width(r));
            return t;
        };
        auto put = [&](int64_t r, const Tensor& t) {
            for (int64_t o2 = 0; o2 < outer; ++o2)
                std::copy(t.ptr() + o2 * width(r), t.ptr() + (o2 + 1) * width(r), act.ptr() + (o2 * n + r) * inner);
        };
        fired.push_back(&s);
        s.in.clear();
        s.predicted.clear();
        Tensor current = get(0);
        call(s, current);
        if (s.edits) put(0, current);
        const std::vector<int> plan = s.edits ? s.predictor.plan(static_cast<size_t>(n - 1)) : std::vector<int>{};
        for (int64_t r = 1; r < n; ++r) {
            Tensor x = get(r);
            if (s.edits) {
                Tensor y = s.predictor.predict(plan[static_cast<size_t>(r - 1)], x);
                put(r, y);
                s.predicted.push_back(std::move(y));
            }
            s.in.push_back(std::move(x));
        }
    }
};

DecodeRuntime::DecodeRuntime(const Model& model, const Tokens& prompt, const GenerateOptions& options, const Hooks* hooks,
                             std::vector<TraceStep>* trace)
    : impl_(std::make_unique<Impl>(model, prompt, options, hooks, trace)) {}
DecodeRuntime::~DecodeRuntime() = default;
std::optional<int64_t> DecodeRuntime::next() { return impl_->next(); }

Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options, const Hooks* hooks,
                std::vector<TraceStep>* trace) {
    DecodeRuntime runtime(model, prompt, options, hooks, trace);
    Tokens out;
    while (const std::optional<int64_t> t = runtime.next()) {
        out.push_back(*t);
        if (options.on_token && !options.on_token(*t)) break;
    }
    return out;
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
