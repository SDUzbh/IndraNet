#include "tfi/generate.h"

#include <algorithm>
#include <cmath>
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
    if (top_k_ > 0 && static_cast<size_t>(top_k_) < keep) {
        keep = static_cast<size_t>(top_k_);
        std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(keep), candidates_.end(), more_likely);
    } else if (top_p_ < 1) {
        std::sort(candidates_.begin(), candidates_.end(), more_likely);
    }
    if (top_p_ < 1) {  // the fewest most likely tokens whose probabilities reach top_p
        double total = 0;
        for (size_t i = 0; i < keep; ++i) total += candidates_[i].first;
        double sum = 0;
        size_t i = 0;
        while (i < keep) {
            sum += candidates_[i++].first;
            if (sum >= top_p_ * total) break;
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

Tokens generate(const Model& model, const Tokens& prompt, const GenerateOptions& options, const Hooks* hooks) {
    Sampler sample(options);
    if (prompt.empty()) throw std::invalid_argument("generate needs a prompt of at least one token");
    std::set<int64_t> stop(options.stop_tokens.begin(), options.stop_tokens.end());
    if (options.stop_at_eos) stop.insert(model.eos_tokens().begin(), model.eos_tokens().end());
    Tokens out;
    if (options.max_new_tokens == 0) return out;
    KVCache cache;
    Tensor logits = model.forward_cached(prompt, cache, hooks, /*last_only=*/true);
    for (;;) {
        const int64_t t = sample(logits.ptr(), logits.shape[1]);
        out.push_back(t);
        bool more = !stop.count(t) && static_cast<int64_t>(out.size()) < options.max_new_tokens;
        if (options.on_token && !options.on_token(t)) more = false;
        if (!more) return out;
        logits = model.forward_cached({t}, cache, hooks, /*last_only=*/true);
    }
}

}  // namespace tfi
