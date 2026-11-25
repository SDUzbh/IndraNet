#include "tfi/runtime.h"

#include "../model/ops.h"
#include "trace.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>

namespace tfi {

using detail::TraceRecorder;

namespace {

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
    // How edit `id` is predicted now: the first explanation left (-1: the row unchanged).
    unsigned kind(int id) const {
        if (id < 0) return kSame;
        const unsigned k = edits_[static_cast<size_t>(id)].kinds;
        return k & (~k + 1);
    }
    Tensor predict(int id, unsigned kind, const Tensor& x) const {
        return id < 0 ? x : apply(edits_[static_cast<size_t>(id)], kind, x);
    }
    // Whether the hooks' output y is `predicted`, what predict(id, kind, x) gave.
    bool agrees(int id, unsigned kind, const Tensor& x, const Tensor& predicted, const Tensor& y) const {
        return close(id < 0 ? nullptr : &edits_[static_cast<size_t>(id)], kind, x, predicted, y);
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
        if (history_.size() > 2 * kHistory)
            history_.erase(history_.begin(), history_.end() - static_cast<std::ptrdiff_t>(kHistory));
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
    // Whether b holds a's values, a being edit e of kind `kind` redone on x. Exactly for a row left alone or values
    // overwritten (redoing those is exact, so any difference is a real edit, however small); a scale or an offset
    // redone can differ in the last bits (adding y - x to another row is not adding the vector the hook added), so
    // there, up to that rounding: a relative difference of about 1e-6 of the values involved.
    static bool close(const Edit* e, unsigned kind, const Tensor& x, const Tensor& a, const Tensor& b) {
        if (a.shape != b.shape) return false;
        const bool exact = !e || kind == kSame || kind == kOverwrite || e->in.shape != x.shape;
        for (int64_t i = 0; i < a.numel(); ++i) {
            const float u = a.ptr()[i], v = b.ptr()[i];
            if (u == v) continue;
            if (exact) return false;
            float scale = std::fabs(u) + std::fabs(v);
            if (kind == kOffset) scale += std::fabs(e->in.ptr()[i]) + std::fabs(e->out.ptr()[i]);
            if (!(std::fabs(u - v) <= 1e-6f * scale)) return false;
        }
        return true;
    }
    int match(const Tensor& x, const Tensor& y) {
        for (size_t j = 0; j < edits_.size(); ++j) {
            Edit& e = edits_[j];
            if (e.in.shape != x.shape && !(e.kinds & kSame)) continue;
            unsigned left = 0;
            for (unsigned kind : {kSame, kOverwrite, kScale, kOffset})
                if ((e.kinds & kind) && close(&e, kind, x, apply(e, kind, x), y)) left |= kind;
            if (left) {
                e.kinds = left;
                return static_cast<int>(j);
            }
        }
        if (edits_.size() == kEdits) return -1;
        edits_.push_back(
            {x, y, close(nullptr, kSame, x, x, y) ? unsigned(kSame) : unsigned(kOverwrite | kScale | kOffset)});
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
        int64_t block = -1;  // its block (-1: before the first, hook_embed and hook_pos_embed; the last for hook_final)
        int64_t resume = -1;  // the block to compute a token again from when this point's output was mispredicted
        EditPredictor predictor;
        std::vector<Tensor> in, predicted;          // this pass's guesses (rows 1, 2, ...)
        std::vector<std::pair<int, unsigned>> how;  // ... the edit and kind each prediction used
        bool done = false;                          // called for the token being confirmed
        Tensor done_out;
        size_t index = 0;   // in names
        char expected = 0;  // what the filters said before this pass: 0 not on it, 1 captures only, 2 edits
    };

    const Model& model;
    const Tokens prompt;
    const GenerateOptions o;
    const Hooks* user;
    std::vector<TraceStep>* trace;
    Sampler sample;
    const bool greedy;  // every pick the most likely token: proposals are checked by id, without distributions
    std::set<int64_t> stop;
    std::optional<TraceRecorder> rec;
    // A token's hooks run on a copy of the caller's (cur), made when they are about to run, as each pass of
    // Model::forward_cached runs a copy: what a hook keeps in itself (a mutable lambda's state) lasts one token, and
    // the caller's Hooks is left as it was. Filters are asked then too, as a pass asks them.
    Hooks plain, routed, cur;  // a pass without speculation: the caller's hooks and the trace; or through this runtime
    std::vector<std::string> names;  // with speculation and hooks: every hook point
    std::map<std::string, size_t> index_of;
    std::map<std::string, Point> points;  // the hook points the caller's hooks have been on
    std::vector<Point*> fired;            // in the order this pass reached them
    std::vector<char> seen;               // per name, what the hooks of this pass's first row were (as Point::expected)
    PassKind kind = PassKind::Plain;
    // Recomputing a token from a block needs its input to the block (hook_resid_pre): from the point there if
    // the caller's hooks are on it, else recorded from each speculative pass.
    bool resume = false;                    // tokens are computed again from a block rather than from the start
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

    Impl(const Model& m, const Tokens& t, const GenerateOptions& options, const Hooks* hooks,
         std::vector<TraceStep>* tr)
        : model(m), prompt(t), o(options), user(hooks), trace(tr), sample(o), greedy(sample.greedy()),
          K(o.speculative.draft_tokens), speculate(K > 0), draft(o.speculative.draft), ctx(t), want(K) {
        if (prompt.empty()) throw std::invalid_argument("generate needs a prompt of at least one token");
        stop.insert(o.stop_tokens.begin(), o.stop_tokens.end());
        if (o.stop_at_eos) stop.insert(model.eos_tokens().begin(), model.eos_tokens().end());
        if (o.speculative.stats) *o.speculative.stats = {};
        if (o.max_new_tokens == 0) {
            finished = true;
            return;
        }
        if (trace) rec.emplace(model, o.trace);
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
        names = model.hook_names();
        for (size_t i = 0; i < names.size(); ++i) index_of[names[i]] = i;
        // (A trace keeps each pass's activations whole: its tokens are computed again from the start.)
        resume = !trace;
        routed.add_filter([this](const std::string& n) { return index_of.count(n) && cur.changes(n); },
                          [this](const std::string& n, Tensor& a) { at_point(n, a); });
        routed.add_capture(
            [this](const std::string& n) { return index_of.count(n) && !cur.changes(n) && cur.matches(n); },
            [this](const std::string& n, Tensor& a) { at_point(n, a); });
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

    void add_trace(Hooks& h) {
        TraceRecorder* r = &*rec;
        h.add_capture([r](const std::string& n) { return r->points.count(n) > 0; },
                      [r](const std::string& n, Tensor& a) { r->capture(n, a); });
    }

    // The hooks of a pass without speculation: the caller's as they are now, and the trace.
    const Hooks* plain_pass() {
        plain = Hooks();
        if (user) plain.extend(*user);
        if (trace) add_trace(plain);
        return plain.empty() ? nullptr : &plain;
    }

    Point& point(size_t i) {
        const std::string& n = names[i];
        const auto it = points.find(n);
        if (it != points.end()) return it->second;
        Point s;
        s.name = n;
        s.index = i;
        auto ends = [&](const char* e) {
            const size_t l = std::strlen(e);
            return n.size() >= l && n.compare(n.size() - l, l, e) == 0;
        };
        s.pattern = ends("attn.hook_pattern");
        s.row_dim = s.pattern || ends("attn.hook_q") || ends("attn.hook_k") || ends("attn.hook_v") ? 1 : 0;
        if (n.compare(0, 7, "blocks.") == 0) s.block = std::stoll(n.substr(7));
        else if (n == "hook_final") s.block = n_layers - 1;
        s.resume = ends(".hook_resid_post") && s.block + 1 < n_layers ? s.block + 1 : s.block;
        return points.emplace(n, std::move(s)).first->second;
    }

    // What cur's hooks are on hook point n: 0 nothing, 1 captures only, 2 edits.
    char mode(const std::string& n) const { return cur.changes(n) ? 2 : cur.matches(n) ? 1 : 0; }

    // The prompt's pass, the same with and without speculation.
    int64_t first_token() {
        L = model.forward_cached(prompt, cache, plain_pass(), /*last_only=*/true);
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
            L = model.forward_cached({ctx.back()}, cache, plain_pass(), /*last_only=*/true);
            return;
        }
        // A token the draft has no row for (a smaller vocabulary): it can't read this text from here on, so
        // the model continues alone.
        if (draft && std::any_of(ctx.begin() + draft_cache.size(), ctx.end(),
                                 [&](int64_t t) { return t >= draft->cfg().vocab_size; }))
            draft = nullptr;
        if (user) {
            cur = *user;  // the current token's hooks
            expect();
        }
        propose();
        Tokens in{ctx.back()};
        for (int64_t t : proposals)
            if (t < model.cfg().vocab_size) in.push_back(t);
        fired.clear();
        recorded.clear();
        seen.assign(names.size(), 0);
        kind = in.size() > 1 ? PassKind::Speculative : PassKind::Plain;
        L = model.forward_cached(in, cache, user ? &routed : plain_pass(), /*last_only=*/false);
        ++stats.steps;
        proposed = static_cast<int64_t>(proposals.size());
        stats.drafted += proposed;
    }

    // The hook points this pass's hooks are expected on, as the filters say now: for how much to propose, and which
    // blocks' inputs to record. (The pass asks the filters again as it reaches each point.)
    void expect() {
        for (auto& kv : points) kv.second.expected = 0;
        for (size_t i = 0; i < names.size(); ++i)
            if (const char md = mode(names[i])) point(i).expected = md;
        record.clear();
        if (!resume) return;
        for (const auto& kv : points) {
            if (!kv.second.expected) continue;
            for (int64_t b : {kv.second.block, kv.second.resume}) {
                if (b < 0) continue;
                const std::string n = "blocks." + std::to_string(b) + ".hook_resid_pre";
                const auto p = points.find(n);
                if (p == points.end() || !p->second.expected) record[n] = b;
            }
        }
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
            if (k > 0 && kv.second.expected == 2 && !(resume && 2 * kv.second.resume >= n_layers))
                k = std::min(k, kv.second.predictor.confident(static_cast<size_t>(k) + 1) - 1);
        if (k <= 0) return;
        if (draft) {
            if (!greedy && q.size() < static_cast<size_t>(k)) q.resize(static_cast<size_t>(k));
            const int64_t dv = draft->cfg().vocab_size;
            Tensor dl =
                draft->forward_cached(Tokens(ctx.begin() + draft_cache.size(), ctx.end()), draft_cache, nullptr, true);
            for (int64_t i = 0; i < k; ++i) {
                if (greedy) {  // its distribution would be one-hot at this token
                    proposals.push_back((*draft_sample)(dl.ptr(), dv));
                } else {
                    draft_sample->distribution(dl.ptr(), dv, q[static_cast<size_t>(i)]);
                    proposals.push_back(draft_sample->draw(q[static_cast<size_t>(i)]));
                }
                // A token past the model's vocabulary (a draft with more rows) ends the proposals: the model
                // can't run it, and it is rejected below, as its probability under the model is 0.
                if (proposals.back() >= model.cfg().vocab_size || stop.count(proposals.back())) break;
                if (i + 1 < k) dl = draft->forward_cached({proposals.back()}, draft_cache, nullptr, true);
            }
        } else if (!o.speculative.draft) {
            proposals = lookup_draft(ctx, o.speculative.ngram, k);
            const auto s =
                std::find_if(proposals.begin(), proposals.end(), [&](int64_t t) { return stop.count(t) > 0; });
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
            if (greedy) {  // p and q are one-hot: the proposal is kept if it is the model's pick, which replaces it if
                           // not
                const int64_t best = sample(lg, V);
                accepted = best == t;
                t = best;
            } else {
                sample.distribution(lg, V, p);
                const std::vector<double>* qi = draft ? &q[r] : nullptr;
                const double pt = t < V ? p[static_cast<size_t>(t)] : 0, qt = qi ? (*qi)[static_cast<size_t>(t)] : 1;
                accepted = pt > 0 && (pt >= qt || sample.uniform() * qt < pt);
                if (!accepted) {
                    w = p;
                    if (qi)
                        for (size_t j = 0; j < w.size() && j < qi->size(); ++j) w[j] = std::max(0.0, w[j] - (*qi)[j]);
                    else if (t < V) w[static_cast<size_t>(t)] = 0;
                    double total = 0;
                    for (double x : w) total += x;
                    t = sample.draw(total > 0 ? w : p);
                }
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
    // computed again from the lowest block that was (or the start, without its input there). If the filters
    // now put the hooks on other points than the pass did, it is computed again from the start.
    void settle_row() {
        if (user) cur = *user;  // this token's hooks
        for (auto& kv : points) kv.second.done = false;
        if (!same_routing()) return recompute_row();
        const Point* wrong = confirm_row();
        if (!wrong && stale_from >= n_layers) return;
        // hook_final mispredicted: it is the last point of a pass, confirmed only when nothing is stale, so everything
        // before it stands. Its output makes this row's logits and nothing else (no key or value, no other row), so
        // they alone are computed again from it, and the later rows still stand. (A trace keeps the pass's
        // activations, which this would leave behind: it computes the row again.)
        if (wrong && wrong->name == "hook_final" && resume) {
            ++stats.reruns;
            const Tensor lg = model.rerun_final(pos + row, wrong->done_out, cache);
            std::copy(lg.ptr(), lg.ptr() + lg.numel(), L.ptr() + row * L.shape[1]);
            return;
        }
        int64_t b = stale_from;
        for (const Point* s : fired)
            if (!s->done) b = std::min(b, s->block);
        Tensor x;
        if (wrong && wrong->resume > wrong->block &&
            wrong->resume <= b) {  // hook_resid_post: its output is the next block's input
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

    // Whether cur's hooks are on the points, and as edits or captures, that the hooks of this pass's first row were.
    bool same_routing() const {
        for (size_t i = 0; i < names.size(); ++i)
            if (mode(names[i]) != seen[i]) return false;
        return true;
    }

    // Row `row`'s input to block b, (1, d_model), from this pass; empty if not kept.
    Tensor input_to(int64_t b) const {
        if (b < 0 || b >= n_layers) return {};
        const std::string n = "blocks." + std::to_string(b) + ".hook_resid_pre";
        const auto p = points.find(n);
        if (p != points.end() && seen[p->second.index])
            return static_cast<size_t>(row) <= p->second.in.size() ? p->second.in[static_cast<size_t>(row) - 1]
                                                                   : Tensor();
        const auto it = recorded.find(b);
        if (it == recorded.end() || it->second.dim() != 2 || it->second.shape[0] <= row) return {};
        const int64_t d = it->second.shape[1];
        return Tensor({1, d}, std::vector<float>(it->second.ptr() + row * d, it->second.ptr() + (row + 1) * d));
    }

    // The first point whose real hooks didn't do what was predicted, or nullptr.
    const Point* confirm_row() {
        const size_t r = static_cast<size_t>(row) - 1;
        for (Point* s : fired) {
            if (s->block >= stale_from) continue;
            const Tensor& x = s->in[r];
            Tensor y = x;
            cur.run(s->name, y);
            s->done = true;
            if (!s->edits) continue;
            if (y.shape != x.shape) throw std::runtime_error(s->name + ": a hook changed the activation's shape");
            const bool right = s->predictor.agrees(s->how[r].first, s->how[r].second, x, s->predicted[r], y);
            s->predictor.observe(x, y);
            s->done_out = std::move(y);
            if (!right) return s;
        }
        return nullptr;
    }

    // Row `row` alone, with the hook outputs confirm_row got, if any: the guesses after it depended on the wrong value.
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
        Point& s = point(index_of.at(name));
        if (kind == PassKind::Rerun && s.done) {  // confirm_row ran its hooks
            if (s.edits) act = s.done_out;
            return;
        }
        s.edits = cur.changes(name);
        if (kind == PassKind::Speculative) {
            seen[s.index] = s.edits ? 2 : 1;
            speculative_point(s, act);
        } else {
            call(s, act);
        }
    }

    // The real hooks on one confirmed row, in place.
    void call(Point& s, Tensor& row_act) {
        if (!s.edits) {
            cur.run(s.name, row_act);
            return;
        }
        const Tensor x = row_act;
        cur.run(s.name, row_act);
        if (row_act.shape != x.shape) throw std::runtime_error(s.name + ": a hook changed the activation's shape");
        s.predictor.observe(x, row_act);
    }

    void speculative_point(Point& s, Tensor& act) {
        const size_t dim = static_cast<size_t>(s.row_dim);
        const int64_t outer = s.row_dim ? act.shape[0] : 1, n = act.shape[dim];
        int64_t inner = 1;
        for (size_t k = dim + 1; k < act.shape.size(); ++k) inner *= act.shape[k];
        auto width = [&](int64_t r) {
            return s.pattern ? pos + r + 1 : inner;
        };  // a pattern's row: the keys up to its token
        auto get = [&](int64_t r) {
            Shape shape = act.shape;
            shape[dim] = 1;
            if (s.pattern) shape[2] = width(r);
            Tensor t(shape);
            for (int64_t o2 = 0; o2 < outer; ++o2)
                std::copy(act.ptr() + (o2 * n + r) * inner, act.ptr() + (o2 * n + r) * inner + width(r),
                          t.ptr() + o2 * width(r));
            return t;
        };
        auto put = [&](int64_t r, const Tensor& t) {
            for (int64_t o2 = 0; o2 < outer; ++o2)
                std::copy(t.ptr() + o2 * width(r), t.ptr() + (o2 + 1) * width(r), act.ptr() + (o2 * n + r) * inner);
        };
        fired.push_back(&s);
        s.in.clear();
        s.predicted.clear();
        s.how.clear();
        Tensor current = get(0);
        call(s, current);
        if (s.edits) put(0, current);
        const std::vector<int> plan = s.edits ? s.predictor.plan(static_cast<size_t>(n - 1)) : std::vector<int>{};
        for (int64_t r = 1; r < n; ++r) {
            Tensor x = get(r);
            if (s.edits) {
                const int id = plan[static_cast<size_t>(r - 1)];
                const unsigned k = s.predictor.kind(id);
                Tensor y = s.predictor.predict(id, k, x);
                put(r, y);
                s.predicted.push_back(std::move(y));
                s.how.emplace_back(id, k);
            }
            s.in.push_back(std::move(x));
        }
    }
};

DecodeRuntime::DecodeRuntime(const Model& model, const Tokens& prompt, const GenerateOptions& options,
                             const Hooks* hooks, std::vector<TraceStep>* trace)
    : impl_(std::make_unique<Impl>(model, prompt, options, hooks, trace)) {}
DecodeRuntime::~DecodeRuntime() = default;
std::optional<int64_t> DecodeRuntime::next() { return impl_->next(); }

}  // namespace tfi
