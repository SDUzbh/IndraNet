// Internal: what generate()'s trace records from each pass, shared by DecodeRuntime (runtime.cpp) and
// generate_batch (generate.cpp). Not part of the public API.
#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include "tfi/generate.h"

namespace tfi {
namespace detail {

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
        if (o.top_k < 0 || o.attention_top < 0)
            throw std::invalid_argument("trace: top_k and attention_top must be 0 or more");
        if (o.lens)
            for (const std::string& n : lens_hook_points(m)) points.insert(n);
        if (o.dla) {
            direct_logit_attribution_from_check();
            for (const std::string& n : dla_hook_points(m)) points.insert(n);
        }
        if (o.attention_top > 0)
            for (int64_t l = 0; l < m.cfg().n_layers; ++l) points.insert(std::format("blocks.{}.attn.hook_pattern", l));
    }
    void direct_logit_attribution_from_check() const {
        for (int64_t i = 0; i < m.cfg().n_layers; ++i)
            if (m.cfg().norm_position == "post" || m.layer_cfg(i).norm_position == "post")
                throw std::invalid_argument(
                    "trace: direct logit attribution needs an additive residual stream; post-norm "
                    "models have none (set trace.dla = false)");
    }
    void capture(const std::string& name, const Tensor& a) {
        if (all_rows) {
            last[name] = a;
            return;
        }
        if (name.ends_with("hook_pattern")) {  // (H, T, keys): each head's last query
            const int64_t H = a.shape[0], T = a.shape[1], K = a.shape[2];
            Tensor t({H, 1, K});
            for (int64_t h = 0; h < H; ++h)
                std::copy(a.ptr() + (h * T + T - 1) * K, a.ptr() + (h * T + T) * K, t.ptr() + h * K);
            last[name] = std::move(t);
            return;
        }
        const int64_t rows = a.shape[0], width = a.numel() / std::max<int64_t>(rows, 1);
        Shape shape = a.shape;
        shape[0] = 1;
        last[name] = Tensor(shape, std::vector<float>(a.ptr() + (rows - 1) * width, a.ptr() + rows * width));
    }
    // The step for `token`, chosen from `logits`: those of row `row` of the latest pass (0 without all_rows).
    TraceStep step(std::span<const float> logits, int64_t token, int64_t position, int64_t row = 0) const {
        TraceStep st;
        st.token = token;
        st.position = position;
        std::vector<float> p(logits.begin(), logits.end());
        softmax_(p);
        st.prob = p[token];
        st.logprob = std::log(std::max(p[token], std::numeric_limits<float>::min()));
        st.top_tokens = topk(p, opt.top_k);
        for (int64_t t : st.top_tokens) st.top_probs.push_back(p[t]);
        if (opt.lens) st.lens = logit_lens_from(m, last, row, token, opt.top_k);
        if (opt.dla) st.dla = direct_logit_attribution_from(m, last, row, token);
        for (int64_t l = 0; opt.attention_top > 0 && l < m.cfg().n_layers; ++l) {
            const Tensor& P = last.at(std::format("blocks.{}.attn.hook_pattern", l));
            const int64_t H = P.shape[0], T = P.shape[1], K = P.shape[2];
            const int64_t keys = K - (T - 1 - row);  // the keys row `row` sees: those up to its position
            for (int64_t h = 0; h < H; ++h) {
                TraceStep::Attention a;
                a.head = std::format("L{}H{}", l, h);
                const std::span<const float> pr = P.row_view(h * T + row).first(static_cast<size_t>(keys));
                a.positions = topk(pr, std::min(opt.attention_top, keys));
                for (int64_t j : a.positions) a.weights.push_back(pr[j]);
                st.attention.push_back(std::move(a));
            }
        }
        return st;
    }
};

}  // namespace detail
}  // namespace tfi
