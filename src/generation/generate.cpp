#include "tfi/generate.h"

#include "../model/ops.h"
#include "tfi/runtime.h"
#include "trace.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
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
    if (!(o.temperature >= 0) || std::isinf(o.temperature))
        throw std::invalid_argument("temperature must be 0 or more");
    if (o.top_k < 0) throw std::invalid_argument("top_k must be 0 or more");
    if (!(o.top_p > 0 && o.top_p <= 1)) throw std::invalid_argument("top_p must be in (0, 1]");
    if (o.speculative.draft_tokens < 0) throw std::invalid_argument("speculative.draft_tokens must be 0 or more");
    if (o.speculative.ngram < 1) throw std::invalid_argument("speculative.ngram must be 1 or more");
}
}  // namespace

Sampler::Sampler(const GenerateOptions& o)
    : temperature_(o.temperature), top_p_(o.top_p), top_k_(o.top_k), rng_(o.seed) {
    check(o);
}

bool Sampler::greedy() const {
    return temperature_ == 0 || top_k_ == 1 || 1.0 / temperature_ > std::numeric_limits<float>::max();
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
    return y * std::bit_cast<float>(bits);
}

void exp_weights(std::span<const float> l, float top, float scale, std::span<float> out) {
    const float* logits = l.data();
    float* w = out.data();
    const int64_t n = std::ssize(l);
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
double sum_weights(std::span<const float> weights) {
    const float* w = weights.data();
    const size_t n = weights.size();
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
    const auto u = std::bit_cast<uint32_t>(v);
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

template <std::invocable<int64_t> Key, std::invocable<int64_t> Amount>
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
        if (b < 0)
            return {prefix << (shift + bits), std::numeric_limits<int64_t>::max()};  // rounding: all of the bucket
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
float max_logit(std::span<const float> logits) {
    const float* l = logits.data();
    const int64_t n = std::ssize(logits);
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

bool Sampler::select(std::span<const float> logits, int64_t& best, double& total) {
    const int64_t n = std::ssize(logits);
    // Greedy: the largest logit, the lowest id among equal ones.
    const float top = max_logit(logits);
    if (top == -INFINITY) throw std::runtime_error("generate: no logit to choose a token from (all NaN or -inf)");
    best = 0;
    while (!(logits[best] == top)) ++best;
    // (A temperature whose reciprocal overflows a float leaves the largest logit alone, as greedy does.)
    if (greedy() || top == INFINITY) return false;
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
        detail::exp_weights(gathered_, top, scale, w_);
    } else {
        w_.resize(static_cast<size_t>(n));
        detail::exp_weights(logits, top, scale, w_);
    }
    if (top_p_ < 1) {  // the nucleus: a weight's bits order as its value (weights are >= 0; 0: never)
        const size_t m = w_.size();
        auto key = [&](int64_t j) { return std::bit_cast<uint32_t>(w_[static_cast<size_t>(j)]); };
        const Cut c = cut_at(
            static_cast<int64_t>(m), key, [&](int64_t j) { return static_cast<double>(w_[static_cast<size_t>(j)]); },
            top_p_ * sum_weights(w_), hist_, sel_);
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
    total = sum_weights(w_);
    return true;
}

int64_t Sampler::operator()(std::span<const float> logits) {
    int64_t best;
    double total;
    if (!select(logits, best, total)) return best;
    const double u = uniform() * total;
    // The token where the running sum passes u: whole blocks' sums first, then within the block.
    constexpr size_t kBlock = 64;
    double sum = 0;
    int64_t last = best;
    for (size_t b = 0; b < w_.size(); b += kBlock) {
        const size_t e = std::min(w_.size(), b + kBlock);
        const double block = sum_weights(std::span(w_).subspan(b, e - b));
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

void Sampler::distribution(std::span<const float> logits, std::vector<double>& p) {
    int64_t best;
    double total;
    const bool sampled = select(logits, best, total);
    p.assign(logits.size(), 0.0);
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

// Through a DecodeRuntime (runtime.cpp), which also speculates.
Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options, const Hooks* hooks,
                std::vector<TraceStep>* trace) {
    DecodeRuntime runtime(model, prompt, options, hooks, trace);
    Tokens out;
    while (const std::optional<int64_t> t = runtime.next()) {
        out.push_back(*t);
        if (options.on_token && !options.on_token(*t)) break;
    }
    if (options.kv_clamped) *options.kv_clamped = runtime.kv_clamped();
    return out;
}

namespace {

// Hooks for a batched pass that give each sequence its own hooks: every hook point's stacked activation
// split into each sequence's rows (dimension 1 for hook_q / hook_k / hook_v, (H, rows, D); dimension 0
// otherwise), that sequence's hooks run on its slice, and what they leave written back. A hook point that
// only captures stays a capture (a GPU pass doesn't stop for it).
struct PerSequenceHooks {
    std::vector<Hooks> of;                                              // per sequence
    std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> segs;  // this pass: (sequence, (first row, rows))
    Hooks all;

    static bool heads_first(const std::string& n) {
        return n.ends_with("attn.hook_q") || n.ends_with("attn.hook_k") || n.ends_with("attn.hook_v");
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
                std::copy(act.ptr() + (o * rows + r0) * inner, act.ptr() + (o * rows + r0 + n) * inner,
                          slice.ptr() + o * n * inner);
            h.run(name, slice);
            if (!write) continue;
            if (slice.shape != shape) throw std::runtime_error(name + ": a hook changed the activation's shape");
            for (int64_t o = 0; o < outer; ++o)
                std::copy(slice.ptr() + o * n * inner, slice.ptr() + (o + 1) * n * inner,
                          act.ptr() + (o * rows + r0) * inner);
        }
    }
    explicit PerSequenceHooks(size_t n) : of(n) {
        all.add_filter(
            [this](const std::string& name) {
                for (const Hooks& h : of)
                    if (h.changes(name)) return true;
                return false;
            },
            [this](const std::string& name, Tensor& act) { each(name, act, true); });
        all.add_capture(
            [this](const std::string& name) {
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

std::vector<Tokens> generate_batch(const Model& model, const std::vector<Tokens>& prompts,
                                   const GenerateOptions& options, const std::vector<const Hooks*>& hooks,
                                   std::vector<std::vector<TraceStep>>* traces) {
    const size_t n = prompts.size();
    if (!hooks.empty() && hooks.size() != n)
        throw std::invalid_argument("generate_batch: hooks must be empty or one per prompt");
    if (options.on_token)
        throw std::invalid_argument(
            "generate_batch: on_token is not used here; give generate() one prompt to follow it");
    if (options.speculative.draft_tokens > 0)
        throw std::invalid_argument(
            "generate_batch: speculative decoding is not supported here; give generate() one prompt for it");
    if (traces && options.trace.attention_top > 0)
        throw std::invalid_argument(
            "generate_batch: a trace's attention needs each sequence's pattern; use generate() for that");
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
    if (options.kv_clamped) *options.kv_clamped = 0;
    if (n == 0 || options.max_new_tokens == 0) return out;

    PerSequenceHooks ph(n);
    std::vector<std::optional<detail::TraceRecorder>> recs(n);
    bool any_hooks = false;
    for (size_t i = 0; i < n; ++i) {
        if (!hooks.empty() && hooks[i]) {
            ph.of[i].extend(*hooks[i]);
            any_hooks = true;
        }
        if (traces) {
            recs[i].emplace(model, options.trace);
            detail::TraceRecorder* r = &*recs[i];
            ph.of[i].add_capture([r](const std::string& nm) { return r->points.contains(nm); },
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
                ph.segs.push_back({static_cast<int64_t>(i), {row, std::ssize(step[i])}});
                row += std::ssize(step[i]);
            }
        if (ph.segs.empty()) {
            if (options.kv_clamped)
                for (int64_t i = 0; i < batch.size(); ++i) *options.kv_clamped += batch[i].clamped();
            return out;
        }
        const std::vector<Tensor> logits =
            model.forward_cached(step, batch, any_hooks ? &ph.all : nullptr, /*last_only=*/true);
        for (size_t i = 0; i < n; ++i) {
            if (step[i].empty()) continue;
            const Tensor& lg = logits[i];
            const int64_t t = samplers[i](lg.data);
            if (traces) (*traces)[i].push_back(recs[i]->step(lg.data, t, batch[static_cast<int64_t>(i)].size() - 1));
            out[i].push_back(t);
            const bool more = !stop.contains(t) && std::ssize(out[i]) < options.max_new_tokens;
            step[i] = more ? Tokens{t} : Tokens{};
        }
    }
}

}  // namespace tfi
