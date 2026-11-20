#include "tfi/generate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace tfi {

namespace {
void check(const GenerateOptions& o) {
    if (o.max_new_tokens < 0) throw std::invalid_argument("max_new_tokens must be 0 or more");
    if (!(o.temperature >= 0) || std::isinf(o.temperature)) throw std::invalid_argument("temperature must be 0 or more");
    if (o.top_k < 0) throw std::invalid_argument("top_k must be 0 or more");
    if (!(o.top_p > 0 && o.top_p <= 1)) throw std::invalid_argument("top_p must be in (0, 1]");
}
}  // namespace

Sampler::Sampler(const GenerateOptions& o) : temperature_(o.temperature), top_p_(o.top_p), top_k_(o.top_k), rng_(o.seed) {
    check(o);
}

int64_t Sampler::operator()(const float* logits, int64_t n) {
    // Greedy: the largest logit, the lowest id among equal ones.
    int64_t best = -1;
    for (int64_t i = 0; i < n; ++i)
        if (!std::isnan(logits[i]) && logits[i] != -INFINITY && (best < 0 || logits[i] > logits[best])) best = i;
    if (best < 0) throw std::runtime_error("generate: no logit to choose a token from (all NaN or -inf)");
    if (temperature_ == 0 || top_k_ == 1) return best;
    if (logits[best] == INFINITY) return best;

    // Probabilities up to a constant, exp((logit - max) / temperature), most likely first.
    candidates_.clear();
    const double top = logits[best];
    for (int64_t i = 0; i < n; ++i)
        if (!std::isnan(logits[i]) && logits[i] != -INFINITY)
            candidates_.push_back({std::exp((static_cast<double>(logits[i]) - top) / temperature_), i});
    auto more_likely = [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    size_t keep = candidates_.size();
    double mass = 0;  // what top_p is a fraction of: every candidate's probability, or (after top_k) the top k's
    if (top_k_ > 0 && static_cast<size_t>(top_k_) < keep) {
        keep = static_cast<size_t>(top_k_);
        std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(keep), candidates_.end(), more_likely);
        for (size_t i = 0; i < keep; ++i) mass += candidates_[i].first;
    } else if (top_p_ < 1) {
        // Only tokens that can be in the nucleus need sorting. Those below `floor` hold together less
        // than (1 - top_p) of the mass, so the most likely ones above it already reach top_p: the
        // nucleus (and so the pick) is the same as after sorting all of them.
        for (const auto& c : candidates_) mass += c.first;
        const double floor = 0.5 * (1 - top_p_) * mass / static_cast<double>(candidates_.size());
        const auto end = std::partition(candidates_.begin(), candidates_.end(), [&](const auto& c) { return c.first >= floor; });
        keep = static_cast<size_t>(end - candidates_.begin());
        std::sort(candidates_.begin(), end, more_likely);
    }
    if (top_p_ < 1) {  // the fewest most likely tokens whose probabilities reach top_p
        double sum = 0;
        size_t i = 0;
        while (i < keep) {
            sum += candidates_[i++].first;
            if (sum >= top_p_ * mass) break;
        }
        keep = i;
    }
    double total = 0;
    for (size_t i = 0; i < keep; ++i) total += candidates_[i].first;
    // A uniform number in [0, total) from the generator's top 53 bits: the same on every standard library.
    const double u = static_cast<double>(rng_() >> 11) * 0x1.0p-53 * total;
    double sum = 0;
    for (size_t i = 0; i < keep; ++i) {
        sum += candidates_[i].first;
        if (u < sum) return candidates_[i].second;
    }
    return candidates_[keep - 1].second;  // rounding left u at the very end
}

namespace {

// The last row of each activation the trace needs, from every pass (capture hooks).
struct TraceRecorder {
    const Model& m;
    const TraceOptions& opt;
    std::set<std::string> points;
    ActivationCache last;  // name -> its last row: (1, ...); patterns: (H, 1, keys)

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
    TraceStep step(const float* logits, int64_t V, int64_t token, int64_t position) const {
        TraceStep st;
        st.token = token;
        st.position = position;
        std::vector<float> p(logits, logits + V);
        softmax_(p.data(), V);
        st.prob = p[token];
        st.logprob = std::log(std::max(p[token], std::numeric_limits<float>::min()));
        st.top_tokens = topk(p.data(), V, opt.top_k);
        for (int64_t t : st.top_tokens) st.top_probs.push_back(p[t]);
        if (opt.lens) st.lens = logit_lens_from(m, last, 0, token, opt.top_k);
        if (opt.dla) st.dla = direct_logit_attribution_from(m, last, 0, token);
        for (int64_t l = 0; opt.attention_top > 0 && l < m.cfg().n_layers; ++l) {
            const Tensor& P = last.at("blocks." + std::to_string(l) + ".attn.hook_pattern");
            const int64_t H = P.shape[0], K = P.shape[2];
            for (int64_t h = 0; h < H; ++h) {
                TraceStep::Attention a;
                a.head = "L" + std::to_string(l) + "H" + std::to_string(h);
                a.positions = topk(P.ptr() + h * K, K, std::min(opt.attention_top, K));
                for (int64_t j : a.positions) a.weights.push_back(P.data[h * K + j]);
                st.attention.push_back(std::move(a));
            }
        }
        return st;
    }
};

}  // namespace

Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options, const Hooks* hooks,
                std::vector<TraceStep>* trace) {
    Sampler sample(options);
    if (prompt.empty()) throw std::invalid_argument("generate needs a prompt of at least one token");
    std::set<int64_t> stop(options.stop_tokens.begin(), options.stop_tokens.end());
    if (options.stop_at_eos) stop.insert(model.eos_tokens().begin(), model.eos_tokens().end());
    Tokens out;
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
