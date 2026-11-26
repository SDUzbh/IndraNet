// Self-contained unit tests for tfi: invariants of the analysis tools on random models.
// The forward pass itself is checked against PyTorch by python/tests/test_golden.py (golden tests).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "../src/gpu/gpu.h"         // internal (macOS): GPU matrix multiplies
#include "../src/gpu/gpu_chunks.h"  // internal: chunked GPU work (portable half)
#include "../src/model/half.h"      // internal: fp16 conversion
#include "../src/model/ops.h"       // internal: patching batch sizes
#include "../src/model/vmath.h"     // internal: vectorizable exp / erf / tanh
#include "../src/text/regex.h"      // internal: tokenizer split patterns
#include "../src/text/unicode.h"    // internal: UTF-8, categories, NFC
#include "tfi/analysis.h"
#include "tfi/generate.h"
#include "tfi/json.h"
#include "tfi/model.h"
#include "tfi/runtime.h"
#include "tfi/safetensors.h"
#include "tfi/tokenizer.h"

using namespace tfi;

// Tensor::row_view borrows the tensor's data, so it is not callable on a temporary (the view would dangle).
template <class T>
concept HasRowView = requires(T&& t) { std::forward<T>(t).row_view(0); };
static_assert(HasRowView<Tensor&> && HasRowView<const Tensor&>);
static_assert(!HasRowView<Tensor> && !HasRowView<const Tensor>);

static int g_failures = 0, g_checks = 0;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        ++g_checks;                                                                                                    \
        if (!(cond)) {                                                                                                 \
            ++g_failures;                                                                                              \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
        }                                                                                                              \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(double(a) - double(b)) <= (tol))

#define CHECK_THROWS(expr)                                                                                             \
    do {                                                                                                               \
        bool threw = false;                                                                                            \
        try {                                                                                                          \
            (void)(expr);                                                                                              \
        } catch (...) { threw = true; }                                                                                \
        CHECK(threw);                                                                                                  \
    } while (0)

// With fp16 weights the GPU's multiplies take fp16 inputs (WeightFormat): a GPU result is then within about 1e-3
// relative of the CPU's per multiply, compounded through the layers. The tolerance of a comparison that may involve
// the GPU: fp32's own for a float32 model, at least 2e-2 for an fp16 one.
static double tol_for(const Model& m, double fp32) {
    return m.weight_format() == WeightFormat::fp16 ? std::max(fp32, 2e-2) : fp32;
}
// check_close's added tolerance while an fp16 model is tested (HalfTolerance); 0 for float32.
static double g_half_tol = 0;
struct HalfTolerance {
    double prev;
    explicit HalfTolerance(const Model& m) : prev(g_half_tol) { g_half_tol = std::max(prev, tol_for(m, 0)); }
    ~HalfTolerance() { g_half_tol = prev; }
};

// Random weights with exactly the tensor names and shapes indranet's state_dict uses.
static Model random_model(ModelConfig cfg, uint64_t seed = 0, WeightFormat format = WeightFormat::fp32) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::map<std::string, Tensor> w;
    auto randn = [&](const std::string& name, Shape s, float std) {
        Tensor t(s);
        for (float& v : t.data) v = normal(rng) * std;
        w[name] = t;
    };
    auto norm = [&](const std::string& prefix, const ModelConfig& c, int64_t dim) {
        if (c.norm == "none") return;
        Tensor g({dim});
        for (float& v : g.data) v = 1.f + 0.2f * normal(rng);
        w[prefix + ".weight"] = g;
        if (c.norm == "layernorm" && c.bias.has("norm")) randn(prefix + ".bias", {dim}, 0.1f);
    };
    auto lin = [&](const std::string& prefix, const ModelConfig& c, int64_t out, int64_t in, float std) {
        randn(prefix + ".weight", {out, in}, std);
        const std::string part = prefix.find("mlp.") != std::string::npos     ? "mlp"
                                 : prefix.find("o_proj") != std::string::npos ? "o"
                                                                              : "qkv";
        if (c.bias.has(part)) randn(prefix + ".bias", {out}, 0.1f);
    };
    const int64_t d = cfg.d_model;
    randn("embed.weight", {cfg.vocab_size, d}, 1.0f);
    if (cfg.positional == "learned") randn("pos_embed.weight", {cfg.max_seq_len, d}, 0.5f);
    if (!cfg.tie_embeddings) randn("unembed.weight", {cfg.vocab_size, d}, 0.2f);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const ModelConfig c = cfg.layer_config(i);
        const std::string p = std::format("blocks.{}.", i);
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        norm(p + "ln1", c, d);
        lin(p + "attn.q_proj", c, H * D, d, 0.5f);
        lin(p + "attn.k_proj", c, Hkv * D, d, 0.5f);
        lin(p + "attn.v_proj", c, Hkv * D, d, 0.3f);
        lin(p + "attn.o_proj", c, d, H * D, 0.3f);
        if (c.qk_norm) {
            ModelConfig qc = c;
            if (qc.norm == "none") qc.norm = "rmsnorm";
            norm(p + "attn.q_norm", qc, D);
            norm(p + "attn.k_norm", qc, D);
        }
        if (c.norm_position == "sandwich") norm(p + "ln1_post", c, d);
        if (c.has_mlp()) {
            const bool gated = c.mlp != "mlp";
            int64_t d_mlp = c.d_mlp ? *c.d_mlp : 4 * d;
            if (gated && !c.d_mlp) {
                const int64_t base = static_cast<int64_t>(8.0 * d / 3.0), mult = c.mlp_multiple_of;
                d_mlp = mult * ((base + mult - 1) / mult);
            }
            if (gated) lin(p + "mlp.gate", c, d_mlp, d, 0.3f);
            lin(p + "mlp.up", c, d_mlp, d, 0.3f);
            lin(p + "mlp.down", c, d, d_mlp, 0.2f);
            norm(p + "ln2", c, d);
            if (c.norm_position == "sandwich") norm(p + "ln2_post", c, d);
        }
    }
    if (cfg.norm_position != "post") norm("ln_final", cfg, d);
    return Model(cfg, w, format);
}

static ModelConfig tiny(std::function<void(ModelConfig&)> edit = nullptr) {
    ModelConfig c;
    c.vocab_size = 50;
    c.d_model = 16;
    c.n_layers = 2;
    c.n_heads = 4;
    c.max_seq_len = 32;
    c.mlp_multiple_of = 8;
    if (edit) edit(c);
    c.validate();
    return c;
}

static Tokens seq(int64_t T, uint64_t seed, int64_t V = 50) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, V - 1);
    Tokens t(T);
    for (auto& x : t) x = dist(rng);
    return t;
}

static std::vector<std::pair<std::string, ModelConfig>> configs() {
    return {
        {"default (rmsnorm, rope, swiglu)", tiny()},
        {"layernorm + learned + gelu mlp + bias", tiny([](ModelConfig& c) {
             c.norm = "layernorm";
             c.positional = "learned";
             c.mlp = "mlp";
             c.bias = true;
             c.tie_embeddings = false;
         })},
        {"alibi + GQA + window + q/k/v biases only", tiny([](ModelConfig& c) {
             c.positional = "alibi";
             c.n_kv_heads = 2;
             c.sliding_window = 3;
             c.bias = Biases::from_json(json::parse(R"(["qkv"])"));
         })},
        {"parallel residual", tiny([](ModelConfig& c) {
             c.parallel_residual = true;
             c.norm = "layernorm";
             c.bias = true;
         })},
        {"sandwich + softcaps", tiny([](ModelConfig& c) {
             c.norm_position = "sandwich";
             c.attn_logit_softcap = 5.0;
             c.final_logit_softcap = 10.0;
         })},
        {"attention-only, no norm", tiny([](ModelConfig& c) {
             c.mlp = "none";
             c.norm = "none";
         })},
        {"RoPE scaling: yarn, longrope in layer 1 (cos/sin factors)", tiny([](ModelConfig& c) {
             c.rope_scaling = RopeScaling::from_json(
                 json::parse(R"({"type": "yarn", "factor": 4, "original_max_position_embeddings": 8})"));
             c.layer_overrides[1] = json::parse(R"({"rope_scaling": {"type": "longrope", "short_factor": [1.5, 3.0],
                 "attention_factor": 1.25, "original_max_position_embeddings": 64}})");
         })},
    };
}

// RoPE scaling: what the config accepts, and the length limit of "dynamic" and "longrope", on the CPU
// and (macOS) the GPU. The frequencies themselves are checked against PyTorch and transformers by the
// golden tests (python/tests/test_golden.py, test_hf.py).
static void test_rope_scaling() {
    auto parse = [](const char* s) { return RopeScaling::from_json(json::parse(s)); };
    CHECK_THROWS(parse(R"({"type": "ntk", "factor": 2})"));
    CHECK_THROWS(parse(R"({"type": "linear"})"));
    CHECK_THROWS(parse(R"({"type": "linear", "factor": 0})"));
    CHECK_THROWS(parse(R"({"type": "linear", "factor": 2, "beta_fast": 3})"));
    CHECK_THROWS(parse(
        R"({"type": "llama3", "factor": 8, "low_freq_factor": 4, "high_freq_factor": 1, "original_max_position_embeddings": 8})"));
    const RopeScaling y =
        parse(R"({"type": "yarn", "factor": 4, "original_max_position_embeddings": 8, "beta_fast": null})");
    CHECK(y.beta_fast == 32.0 && y.truncate);

    const ModelConfig c = tiny([&](ModelConfig& m) {
        m.rope_scaling = parse(R"({"type": "dynamic", "factor": 2, "original_max_position_embeddings": 10})");
    });
    const detail::RopeTable plain = detail::rope_table(tiny()), dyn = detail::rope_table(c);
    CHECK(dyn.inv_freq == plain.inv_freq && dyn.factor == 1.f && dyn.max_len == 10);  // plain RoPE up to its limit
    const Model m = random_model(c, 7);
    ActivationCache cache;
    CHECK(m.run_with_cache(seq(10, 1), cache).shape == (Shape{10, 50}));
    CHECK_THROWS(m.run_with_cache(seq(11, 1), cache));
    CHECK_THROWS(m.forward(seq(11, 1)));
    const ModelConfig wrong = tiny([&](ModelConfig& m) {  // short factors for 4 frequency pairs, head size 4 has 2
        m.rope_scaling =
            parse(R"({"type": "longrope", "short_factor": [1, 2, 3, 4], "original_max_position_embeddings": 10})");
    });
    CHECK_THROWS(detail::rope_table(wrong));
#ifdef TFI_USE_ACCELERATE
    if (m.gpu()) {  // (a Mac can have no Metal device)
        const Device before = device();
        set_device(Device::Gpu);  // single passes on the GPU: the GPU path checks the length too
        CHECK(m.run_with_cache(seq(10, 1), cache).shape == (Shape{10, 50}));
        CHECK_THROWS(m.run_with_cache(seq(11, 1), cache));
        set_device(before);
    }
#endif
}

static void test_json() {
    const json::Value v = json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": null, "d": "x\"é"}, "e": true})");
    CHECK(v.at("a").array.size() == 3);
    CHECK_NEAR(v.at("a").array[2].as_number(), -300, 1e-9);
    CHECK(v.at("b").at("c").is_null());
    CHECK(v.at("b").at("d").as_string() == "x\"\xc3\xa9");
    CHECK(v.at("e").as_bool());
    CHECK_THROWS(json::parse("{\"a\": }"));
}

static void test_reshape_and_linear() {
    // reshape on an rvalue takes over the buffer; on an lvalue it copies and leaves the original alone.
    Tensor t({2, 3}, {0, 1, 2, 3, 4, 5});
    const float* buf = t.ptr();
    Tensor moved = std::move(t).reshape({3, -1});
    CHECK(moved.ptr() == buf);
    CHECK(moved.shape == (Shape{3, 2}));
    const Tensor copy = moved.reshape({6});
    CHECK(copy.ptr() != moved.ptr());
    CHECK(moved.shape == (Shape{3, 2}) && moved.data[5] == 5.f);
    CHECK(copy.shape == (Shape{6}) && copy.data[5] == 5.f);
    CHECK_THROWS(moved.reshape({4}));

    // linear on a 3-D input with bias, against a direct computation.
    std::mt19937_64 rng(2);
    std::normal_distribution<float> n(0, 1);
    Tensor x({2, 3, 4}), W({5, 4}), b({5});
    for (Tensor* p : {&x, &W, &b})
        for (float& v : p->data) v = n(rng);
    const Tensor y = linear(x, W, &b);
    CHECK(y.shape == (Shape{2, 3, 5}));
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t t2 = 0; t2 < 3; ++t2)
            for (int64_t o = 0; o < 5; ++o) {
                float s = b.data[o];
                for (int64_t k = 0; k < 4; ++k) s += x.at({i, t2, k}) * W.at({o, k});
                CHECK_NEAR(y.at({i, t2, o}), s, 1e-4);
            }
    CHECK(linear(x, W).shape == (Shape{2, 3, 5}));
    CHECK_THROWS(linear(x, Tensor({5, 3})));
}

static void test_matmul() {
    std::mt19937_64 rng(1);
    std::normal_distribution<float> n(0, 1);
    Tensor A({5, 7}), B({7, 3});
    for (float& v : A.data) v = n(rng);
    for (float& v : B.data) v = n(rng);
    const Tensor C = matmul(A, B), Ct = matmul(A, transpose2d(B), true);
    for (int64_t i = 0; i < 5; ++i)
        for (int64_t j = 0; j < 3; ++j) {
            float s = 0;
            for (int64_t k = 0; k < 7; ++k) s += A.at({i, k}) * B.at({k, j});
            CHECK_NEAR(C.at({i, j}), s, 1e-4);
            CHECK_NEAR(Ct.at({i, j}), s, 1e-4);
        }
    CHECK_THROWS(matmul(A, A));
}

static void test_cache_and_hooks(const std::string& name, const Model& m) {
    const Tokens t = seq(10, 3);
    ActivationCache cache;
    const Tensor logits = m.run_with_cache(t, cache);
    CHECK(cache.size() == m.hook_names().size());
    for (const auto& n : m.hook_names()) CHECK(cache.contains(n));
    CHECK(max_abs_diff(logits, m.forward(t)) == 0.f);

    // Causality: changing the last token leaves earlier logits untouched.
    Tokens t2 = t;
    t2.back() = (t2.back() + 1) % m.cfg().vocab_size;
    const Tensor l2 = m.forward(t2);
    const int64_t V = m.cfg().vocab_size;
    CHECK(max_abs_diff(rows(logits, 0, 9), rows(l2, 0, 9)) == 0.f);
    CHECK(max_abs_diff(rows(logits, 9, 10), rows(l2, 9, 10)) > 0.f);

    // A zero-ablation hook changes the output.
    Hooks zero;
    zero.add("blocks.1.hook_attn_out", [](const std::string&, Tensor& a) { std::ranges::fill(a.data, 0.f); });
    CHECK(max_abs_diff(m.forward(t, &zero), logits) > 0.f);
    (void)V;
    (void)name;
}

static void test_logit_lens(const Model& m) {
    const Tokens t = seq(9, 4);
    const auto lens = logit_lens(m, t, 8, 7, 3);
    CHECK(lens.front().stage == "embed");
    CHECK(lens.size() == 1 + 2 * 2 - (m.cfg().has_mlp() ? 0 : 2));
    // The last stage is the model's real output distribution.
    Tensor logits = m.forward(t);
    Tensor last = rows(logits, 8, 9);
    softmax_(last.data);
    CHECK_NEAR(lens.back().target_prob, last.data[7], 1e-5);
    CHECK(lens.back().top_tokens[0] == topk(last.data, 1)[0]);
}

static void test_dla(const Model& m) {
    const Tokens t = seq(11, 5);
    const DLAResult r = direct_logit_attribution(m, t, 10, 3, 4);
    CHECK_NEAR(r.sum, r.actual, 1e-3 * std::max(1.f, std::fabs(r.actual)));
    // The actual value is the pre-softcap logit difference; without softcap it equals the logits.
    if (!m.cfg().final_logit_softcap) {
        const Tensor logits = m.forward(t);
        CHECK_NEAR(r.actual, logit_metric(logits, 10, 3, 4), 1e-4);
    }
    const DLAResult single = direct_logit_attribution(m, t, 10, 3);
    CHECK_NEAR(single.sum, single.actual, 1e-3 * std::max(1.f, std::fabs(single.actual)));
}

static void test_patching(const Model& m) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const PatchResult r = patch_residual(m, clean, corrupt, "resid_pre", 7, 1, 2);
    // At layer 0 only position 3 differs, so patching it restores the clean run exactly.
    CHECK_NEAR(r.values.at({0, 3}), 1.0, 1e-4);
    for (int64_t p = 0; p < 8; ++p)
        if (p != 3) CHECK_NEAR(r.values.at({0, p}), 0.0, 1e-4);
    // Positions before the change are identical in both runs at every layer.
    for (int64_t l = 0; l < 2; ++l)
        for (int64_t p = 0; p < 3; ++p) CHECK_NEAR(r.values.at({l, p}), 0.0, 1e-4);
    const PatchResult heads = patch_heads(m, clean, corrupt, 7, 1, 2);
    CHECK(heads.values.shape == (Shape{2, 4}));
    CHECK_THROWS(patch_residual(m, clean, Tokens(7, 0), "resid_pre", 6, 1, 2));
    CHECK_THROWS(patch_residual(m, clean, corrupt, "nope", 7, 1, 2));

    // values = effect / gap wherever a site exists.
    CHECK(r.normalized());
    for (int64_t i = 0; i < r.values.numel(); ++i)
        if (!std::isnan(r.effect.data[i])) CHECK_NEAR(r.values.data[i], r.effect.data[i] / r.gap(), 1e-5);
}

static void test_patching_edge_cases(const Model& m) {
    const bool has_mlp = m.layer_cfg(0).has_mlp();
    // Identical inputs: zero gap. No NaN effects, values undefined (all NaN), nothing claims to restore.
    const Tokens t = seq(8, 11);
    const std::vector<std::string> sites = {"resid_pre", "mlp_out"};
    for (const std::string& site : sites) {
        const PatchResult r = patch_residual(m, t, t, site, 7, 1, 2);
        CHECK(!r.normalized());
        CHECK(r.gap() == 0.f);
        for (float v : r.values.data) CHECK(std::isnan(v));
        for (float e : r.effect.data) {
            if (site == "mlp_out" && !has_mlp) CHECK(std::isnan(e));  // no such site: NaN effect
            else CHECK(e == 0.f);
        }
    }
    const PatchResult heads = patch_heads(m, t, t, 7, 1, 2);
    CHECK(!heads.normalized());
    for (float e : heads.effect.data) CHECK(e == 0.f);

    // A change after `pos` cannot affect it (causality): also a zero gap, not NaN effects.
    Tokens later = t;
    later[7] = (later[7] + 1) % m.cfg().vocab_size;
    const PatchResult causal = patch_residual(m, t, later, "resid_pre", 5, 1, 2);
    CHECK(!causal.normalized());
    for (int64_t l = 0; l < m.cfg().n_layers; ++l) CHECK(causal.effect.at({l, 0}) == 0.f);

    // Missing sites stay NaN in both tensors when normalized.
    if (!has_mlp) {
        Tokens corrupt = t;
        corrupt[2] = (corrupt[2] + 3) % m.cfg().vocab_size;
        const PatchResult r = patch_residual(m, t, corrupt, "mlp_out", 7, 1, 2);
        for (int64_t i = 0; i < r.values.numel(); ++i)
            CHECK(std::isnan(r.values.data[i]) && std::isnan(r.effect.data[i]));
    }
}

// The straightforward versions the optimized tools must agree with: full sequences, full forward
// passes, one at a time.
static Tensor naive_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                          int64_t pos, int64_t target, int64_t other) {
    const int64_t L = m.cfg().n_layers, T = std::ssize(clean), d = m.cfg().d_model;
    ActivationCache cc;
    m.run_with_cache(clean, cc);
    const float base = logit_metric(m.forward(corrupt), pos, target, other);
    Tensor out({L, T}, NAN);
    for (int64_t i = 0; i < L; ++i) {
        const std::string name = std::format("blocks.{}.hook_{}", i, site);
        if (!cc.contains(name)) continue;
        const Tensor& src = cc.at(name);
        for (int64_t t = 0; t < T; ++t) {
            Hooks h;
            h.add(name, [&src, t, d](const std::string&, Tensor& a) {
                std::copy(src.ptr() + t * d, src.ptr() + (t + 1) * d, a.ptr() + t * d);
            });
            out.data[i * T + t] = logit_metric(m.forward(corrupt, &h), pos, target, other) - base;
        }
    }
    return out;
}

static Tensor naive_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                                int64_t other) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    ActivationCache cc;
    m.run_with_cache(clean, cc);
    const float base = logit_metric(m.forward(corrupt), pos, target, other);
    Tensor out({L, H}, NAN);
    for (int64_t i = 0; i < L; ++i) {
        const std::string name = std::format("blocks.{}.attn.hook_z", i);
        const Tensor& src = cc.at(name);
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        for (int64_t hd = 0; hd < Hl; ++hd) {
            Hooks h;
            h.add(name, [&src, hd, Hl, D](const std::string&, Tensor& a) {
                for (int64_t t = 0; t < a.shape[0]; ++t)
                    std::copy(src.ptr() + (t * Hl + hd) * D, src.ptr() + (t * Hl + hd + 1) * D,
                              a.ptr() + (t * Hl + hd) * D);
            });
            out.data[i * H + hd] = logit_metric(m.forward(corrupt, &h), pos, target, other) - base;
        }
    }
    return out;
}

// Bit for bit equal, NaN matching NaN (the sites or heads a layer doesn't have).
static bool same_values(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) return false;
    for (int64_t i = 0; i < a.numel(); ++i)
        if (!(a.data[i] == b.data[i] || (std::isnan(a.data[i]) && std::isnan(b.data[i])))) return false;
    return true;
}

// Two computations of the same thing that add numbers in different orders (batched vs per pass, GPU
// vs CPU): close relative to each value, plus a little relative to the tensor's largest value, since
// rounding errors scale with the numbers summed (models without norms reach ~100 in a small output).
// An fp16 model's GPU multiplies round their inputs to fp16: g_half_tol then.
static void check_close(const Tensor& got, const Tensor& want) {
    CHECK(got.shape == want.shape);
    float scale = 0;
    for (float v : want.data)
        if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
    for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i) {
        if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
        else
            CHECK_NEAR(got.data[i], want.data[i],
                       std::max(1e-4, g_half_tol) * std::max(1.f, std::fabs(want.data[i])) +
                           std::max(1e-5, g_half_tol) * scale);
    }
}

// Patching runs batched passes (src/analysis.cpp): stacked rows, recomputing only the rows from the
// patched position on, and on macOS GPU linear layers. Every batch size, with the GPU off and on,
// must agree with naive patching, and the thread count must never change a result.
// Batched reference means and ablation agree with one plain forward pass per reference and per head.
// The reference sets ablation is checked with: equal lengths (a mean per position), then mixed lengths
// (one pooled mean).
static std::vector<std::vector<Tokens>> ablation_reference_sets(int64_t T) {
    return {{seq(T, 61), seq(T, 62), seq(T, 63)}, {seq(T, 61), seq(4, 62), seq(T + 2, 63)}};
}

// Naive head means and ablation for one reference set: the means summed from each reference's own
// pass, and (per-position means only) each head's loss increase from a whole forward pass with it ablated.
struct NaiveAblation {
    HeadMeans means;
    std::map<int, Tensor> loss;  // per Ablation mode: (L, max heads); empty for pooled means
    float base = 0;
};

static std::vector<NaiveAblation> naive_ablation(const Model& m, const Tokens& t) {
    const int64_t T = std::ssize(t), L = m.cfg().n_layers;
    std::vector<NaiveAblation> out;
    for (const std::vector<Tokens>& refs : ablation_reference_sets(T)) {
        bool per_position = true;
        for (const Tokens& r : refs) per_position = per_position && std::ssize(r) == T;
        NaiveAblation na;
        double count = 0;
        for (const Tokens& r : refs) {
            ActivationCache c;
            m.run_with_cache(r, c, [](const std::string& n) { return n.find("attn.hook_z") != std::string::npos; });
            for (int64_t l = 0; l < L; ++l) {
                const Tensor& z = c.at(std::format("blocks.{}.attn.hook_z", l));
                const int64_t row = z.numel() / z.shape[0];
                if (std::ssize(na.means.layers) <= l) {  // hook_z's shape, one row or one per position
                    Shape shape = z.shape;
                    shape[0] = per_position ? T : 1;
                    na.means.layers.emplace_back(shape);
                }
                for (int64_t p = 0; p < z.shape[0]; ++p)
                    for (int64_t e = 0; e < row; ++e)
                        na.means.layers[l].data[(per_position ? p : 0) * row + e] += z.data[p * row + e];
            }
            count += per_position ? 1.0 : static_cast<double>(r.size());
        }
        for (Tensor& layer : na.means.layers)
            for (float& v : layer.data) v = static_cast<float>(v / count);
        if (per_position) {
            na.base = next_token_loss(m.forward(t), t);
            for (Ablation mode : {Ablation::Zero, Ablation::Mean}) {
                Tensor loss({L, m.max_heads()}, NAN);
                for (int64_t l = 0; l < L; ++l)
                    for (int64_t h = 0; h < m.layer_cfg(l).n_heads; ++h) {
                        const Hooks hk = ablate_head(m, l, h, mode, mode == Ablation::Mean ? &na.means : nullptr);
                        loss.at({l, h}) = next_token_loss(m.forward(t, &hk), t) - na.base;
                    }
                na.loss[static_cast<int>(mode)] = std::move(loss);
            }
        }
        out.push_back(std::move(na));
    }
    return out;
}

// head_means and head_ablation (batched, possibly on the GPU) against the naive results.
static void check_ablation_matches_naive(const Model& m, const Tokens& t, const std::vector<NaiveAblation>& naive) {
    const std::vector<std::vector<Tokens>> sets = ablation_reference_sets(std::ssize(t));
    for (size_t i = 0; i < sets.size() && i < naive.size(); ++i) {
        const HeadMeans means = head_means(m, sets[i], std::ssize(t));
        const NaiveAblation& na = naive[i];
        CHECK(means.layers.size() == na.means.layers.size());
        for (size_t l = 0; l < means.layers.size() && l < na.means.layers.size(); ++l)
            CHECK(max_abs_diff(means.layers[l], na.means.layers[l]) <=
                  tol_for(m, 1e-5) * std::max(1.f, max_abs(na.means.layers[l])));
        if (!means.per_position()) continue;
        for (Ablation mode : {Ablation::Zero, Ablation::Mean}) {
            const Tensor got = head_ablation(m, t, mode, sets[i]);
            const Tensor& want = na.loss.at(static_cast<int>(mode));
            for (int64_t l = 0; l < m.cfg().n_layers; ++l)
                for (int64_t h = 0; h < m.layer_cfg(l).n_heads; ++h)
                    // float rounding grows with the loss (models without norms reach losses in the hundreds)
                    CHECK_NEAR(got.at({l, h}), want.at({l, h}), tol_for(m, 1e-5) * std::max(1.f, std::fabs(na.base)));
        }
    }
}

// The naive results test_optimized_patching_matches_naive compares with, computed once per model on the
// CPU: they don't depend on the batch sizes, threads or device the optimized paths are tried with.
struct NaiveRefs {
    std::map<std::pair<int64_t, std::string>, Tensor> residual;  // (pos, site)
    std::map<std::pair<int64_t, int64_t>, Tensor> heads;         // (pos, other)
    std::vector<NaiveAblation> ablation;
};

static NaiveRefs naive_refs(const Model& m, const Tokens& t, const Tokens& corrupt) {
    const Device before = device();
    set_device(Device::Cpu);
    NaiveRefs r;
    for (int64_t pos : {8, 3}) {
        for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"})
            r.residual[{pos, site}] = naive_patch(m, t, corrupt, site, pos, 1, 2);
        for (int64_t other : {2, -1}) r.heads[{pos, other}] = naive_patch_heads(m, t, corrupt, pos, 1, other);
    }
    r.ablation = naive_ablation(m, t);
    set_device(before);
    return r;
}

// cpu_parts false leaves out what doesn't involve the GPU (forward_from, the CPU's batched passes), for
// callers that already ran it on the same model and only change a GPU setting.
static void test_optimized_patching_matches_naive(const Model& m, bool cpu_parts = true) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    const Tokens t = seq(9, 51);
    if (cpu_parts) {  // forward_from reproduces forward exactly from any layer's cached input
        ActivationCache cache;
        const Tensor full = m.run_with_cache(t, cache);
        for (int64_t l = 0; l <= m.cfg().n_layers; ++l) {
            const Tensor start = l < m.cfg().n_layers ? cache.at(std::format("blocks.{}.hook_resid_pre", l))
                                                      : cache.at(std::format("blocks.{}.hook_resid_post", l - 1));
            CHECK(max_abs_diff(m.forward_from(l, start), full) == 0.f);
        }
        CHECK_THROWS(m.forward_from(-1, cache.at("blocks.0.hook_resid_pre")));
        CHECK_THROWS(m.forward_from(0, Tensor({9, m.cfg().d_model + 1})));
    }

    Tokens corrupt = t;
    corrupt[2] = (corrupt[2] + 5) % m.cfg().vocab_size;
    corrupt[6] = (corrupt[6] + 9) % m.cfg().vocab_size;
    const int64_t last = m.cfg().n_layers - 1;
    const NaiveRefs naive = naive_refs(m, t, corrupt);
    std::vector<bool> gpu_modes;
    if (cpu_parts) gpu_modes.push_back(false);
#ifdef TFI_USE_ACCELERATE
    if (m.gpu()) gpu_modes.push_back(true);
#endif
    for (const bool gpu : gpu_modes) {
#ifdef TFI_USE_ACCELERATE
        detail::set_gpu_min_rows(gpu ? 1 : std::numeric_limits<int64_t>::max());
        detail::set_gpu_min_work(gpu ? 1 : 0);
#else
        (void)gpu;
#endif
        for (const auto& [residual, heads] : std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {5, 9}, {0, 0}}) {
            detail::set_patch_batch_rows(residual, heads);
            for (int64_t pos : {8, 3}) {
                for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"}) {
                    const PatchResult r = patch_residual(m, t, corrupt, site, pos, 1, 2);
                    check_close(r.effect, naive.residual.at({pos, site}));
                    // After the last layer's attention, earlier positions cannot reach `pos`: exactly 0.
                    if (site != "resid_pre" && !std::isnan(r.effect.at({last, 0})))
                        for (int64_t q = 0; q < pos; ++q) CHECK(r.effect.at({last, q}) == 0.f);
                }
                check_close(patch_heads(m, t, corrupt, pos, 1, 2).effect, naive.heads.at({pos, 2}));
                check_close(patch_heads(m, t, corrupt, pos, 1, -1).effect, naive.heads.at({pos, -1}));
            }
            check_ablation_matches_naive(m, t, naive.ablation);
        }
        // The thread count never changes a result.
        detail::set_patch_batch_rows(5, 9);
        set_num_threads(1);
        const Tensor p1 = patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect;
        const Tensor h1 = patch_heads(m, t, corrupt, 8, 1, 2).effect;
        set_num_threads(4);
        CHECK(num_threads() == 4);
        CHECK(same_values(patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect, p1));
        CHECK(same_values(patch_heads(m, t, corrupt, 8, 1, 2).effect, h1));
        set_num_threads(1);
        const std::vector<Tokens> refs = {seq(9, 61), seq(9, 62), seq(9, 63), seq(5, 64)};
        const Tensor a1 = head_ablation(m, t, Ablation::Mean, {refs.begin(), refs.begin() + 3});
        const Tensor pooled1 = head_means(m, refs, 9).layers.back();
        set_num_threads(4);
        CHECK(same_values(head_ablation(m, t, Ablation::Mean, {refs.begin(), refs.begin() + 3}), a1));
        CHECK(same_values(head_means(m, refs, 9).layers.back(), pooled1));
        set_num_threads(0);
    }
    detail::set_patch_batch_rows(0, 0);
#ifdef TFI_USE_ACCELERATE
    // A copy of the model gets its own GPU context (the context caches weights by address) and the
    // same results.
    if (m.gpu()) {
        detail::set_gpu_min_rows(1);
        detail::set_gpu_min_work(1);
        const Model copy = m;
        CHECK(copy.gpu() != nullptr && copy.gpu() != m.gpu());
        check_close(patch_residual(copy, t, corrupt, "resid_pre", 8, 1, 2).effect,
                    patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect);
        // set_use_gpu(false) gives the CPU result bit for bit, even where the GPU would be used.
        detail::set_gpu_min_rows(std::numeric_limits<int64_t>::max());
        const Tensor cpu = patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect;
        const Tensor cpu_heads = patch_heads(m, t, corrupt, 8, 1, 2).effect;
        detail::set_gpu_min_rows(1);
        detail::set_gpu_min_work(1);
        set_use_gpu(false);
        CHECK(same_values(patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect, cpu));
        CHECK(same_values(patch_heads(m, t, corrupt, 8, 1, 2).effect, cpu_heads));
        set_use_gpu(true);

        // A model with fp16 weights: its GPU multiplies take fp16 inputs, really (they differ from its CPU passes,
        // whose arithmetic is float32), close to naive patching (errors around 1e-3 relative), the same for any thread
        // count; with the GPU off, its CPU passes.
        std::map<std::string, Tensor> w;
        for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
        const Model m16(m.cfg(), w, WeightFormat::fp16);
        set_use_gpu(false);
        const Tensor cpu16 = patch_residual(m16, t, corrupt, "attn_out", 8, 1, 2).effect;
        set_use_gpu(true);
        CHECK(max_abs_diff(patch_residual(m16, t, corrupt, "attn_out", 8, 1, 2).effect, cpu16) > 0.f);
        auto check_fp16 = [](const Tensor& got, const Tensor& want) {
            CHECK(got.shape == want.shape);
            float scale = 1.f;
            for (float v : want.data)
                if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
            for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
                if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
                else CHECK_NEAR(got.data[i], want.data[i], 2e-2 * scale);
        };
        for (int64_t pos : {8, 3}) {
            for (const std::string site : {"resid_pre", "attn_out", "mlp_out"})
                check_fp16(patch_residual(m16, t, corrupt, site, pos, 1, 2).effect, naive.residual.at({pos, site}));
            check_fp16(patch_heads(m16, t, corrupt, pos, 1, 2).effect, naive.heads.at({pos, 2}));
        }
        set_num_threads(1);
        const Tensor f1 = patch_residual(m16, t, corrupt, "attn_out", 8, 1, 2).effect;
        set_num_threads(4);
        CHECK(same_values(patch_residual(m16, t, corrupt, "attn_out", 8, 1, 2).effect, f1));
        set_num_threads(0);
        set_use_gpu(false);
        CHECK(same_values(patch_residual(m16, t, corrupt, "attn_out", 8, 1, 2).effect, cpu16));
        set_use_gpu(true);
    }
    detail::set_gpu_min_rows(0);
    detail::set_gpu_min_work(0);
#endif
    set_use_gpu(false);
    CHECK(!use_gpu());
    set_use_gpu(true);
    CHECK(use_gpu());
}

static void test_patch_gap_threshold() {
    PatchResult r;
    r.clean_metric = 5.f, r.corrupt_metric = 5.f;
    CHECK(!r.normalized());
    r.corrupt_metric = 5.f - 1e-7f;  // float noise around a real metric
    CHECK(!r.normalized());
    r.corrupt_metric = 4.9f;  // small but genuine
    CHECK(r.normalized());
    r.clean_metric = 0.f, r.corrupt_metric = 1e-3f;
    CHECK(r.normalized());
    r.clean_metric = NAN;
    CHECK(!r.normalized());
}

static void test_mean_ablation(const Model& m) {
    const int64_t V = m.cfg().vocab_size;
    const Tokens t = seq(10, 21);
    const std::vector<Tokens> refs = {seq(10, 31), seq(10, 32), seq(10, 33)};
    const HeadMeans means = head_means(m, refs, 10);
    CHECK(means.per_position());

    // No future leak: editing only the input's last token never moves earlier logits.
    Tokens t2 = t;
    t2.back() = (t2.back() + 1) % V;
    for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
        const Hooks hk = ablate_head(m, l, 0, Ablation::Mean, &means);
        CHECK(max_abs_diff(rows(m.forward(t, &hk), 0, 9), rows(m.forward(t2, &hk), 0, 9)) == 0.f);
    }

    // Means come from the references: an input that is its own reference is left unchanged, exactly
    // (every head already equals its mean, so no pass runs).
    for (float v : head_ablation(m, t, Ablation::Mean, {t}).data)
        if (!std::isnan(v)) CHECK(v == 0.f);

    // References of other lengths give one pooled mean, written at every position.
    const std::vector<Tokens> other = {seq(6, 41), seq(9, 42)};
    const HeadMeans pooled = head_means(m, other, 10);
    CHECK(!pooled.per_position());
    const int64_t H = m.layer_cfg(0).n_heads, D = m.layer_cfg(0).head_dim();
    std::vector<double> expect(D, 0.0);
    for (const Tokens& r : other) {
        ActivationCache c;
        m.run_with_cache(r, c, [](const std::string& n) { return n == "blocks.0.attn.hook_z"; });
        const Tensor& z = c.at("blocks.0.attn.hook_z");
        for (int64_t p = 0; p < z.shape[0]; ++p)
            for (int64_t e = 0; e < D; ++e) expect[e] += z.data[(p * H + 1) * D + e] / 15.0;  // 6 + 9 positions
    }
    const Hooks hk = ablate_head(m, 0, 1, Ablation::Mean, &pooled);
    ActivationCache after;
    m.run_with_cache(t, after, [](const std::string& n) { return n == "blocks.0.attn.hook_z"; }, &hk);
    const Tensor& z = after.at("blocks.0.attn.hook_z");
    for (int64_t p = 0; p < 10; p += 3)
        for (int64_t e = 0; e < D; ++e) CHECK_NEAR(z.data[(p * H + 1) * D + e], expect[e], 1e-5);

    // Refusals: no references, or per-position means used on an input of another length.
    CHECK_THROWS(head_ablation(m, t, Ablation::Mean));
    CHECK_THROWS(ablate_head(m, 0, 0, Ablation::Mean));
    const Hooks wrong = ablate_head(m, 0, 0, Ablation::Mean, &means);
    CHECK_THROWS(m.forward(seq(7, 1), &wrong));
    CHECK_THROWS(ablate_head(m, 0, 99, Ablation::Zero));
}

static void test_ablation_and_scores(const Model& m) {
    const Tensor abl = head_ablation(m, seq(12, 7), Ablation::Mean, {seq(12, 8), seq(12, 9)});
    CHECK(abl.shape == (Shape{2, 4}));
    for (float v : abl.data) CHECK(std::isfinite(v));
    const HeadScores s = head_scores(m, 6, 2, 0);
    CHECK(s.induction.shape == (Shape{2, 4}));
    for (float v : s.previous_token.data) CHECK(v >= 0.f && v <= 1.f + 1e-5f);
    CHECK(s.top_two_hop.size() == 5);
}

static void test_composition_matches_direct(const Model& m) {
    // The low-rank formula must equal the direct d x d computation.
    const Composition c = composition_scores(m);
    const int64_t H = m.cfg().n_heads;
    for (int64_t a = 0; a < H; a += 3)
        for (int64_t b = 0; b < H; b += 2) {
            const HeadWeights w = head_weights(m, 0, a), r = head_weights(m, 1, b);
            const Tensor ov = ov_matrix(w), qk = qk_matrix(r), ov2 = ov_matrix(r);
            const float k = frobenius(matmul(qk, ov)) / (frobenius(qk) * frobenius(ov));
            const float q = frobenius(matmul(transpose2d(qk), ov)) / (frobenius(qk) * frobenius(ov));
            const float v = frobenius(matmul(ov2, ov)) / (frobenius(ov2) * frobenius(ov));
            const int64_t N = 2 * H, i = a, j = H + b;
            CHECK_NEAR(c.k.data[i * N + j], k, 1e-4);
            CHECK_NEAR(c.q.data[i * N + j], q, 1e-4);
            CHECK_NEAR(c.v.data[i * N + j], v, 1e-4);
        }
    CHECK(std::isnan(c.k.data[0]));  // same layer
    const float base = composition_random_baseline(m, 16);
    CHECK(base > 0.f && base < 1.f);
}

// The OV path through d_model, as before the factored form: embed -> W_V^T W_O^T per head -> unembed.
static float direct_ov_path(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& path, int64_t max_tokens) {
    const int64_t V = m.cfg().vocab_size, d = m.cfg().d_model, n = std::min(V, max_tokens);
    Tensor X({n, d});
    const Tensor E = m.weight("embed.weight").to_tensor();
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < d; ++j) X.data[i * d + j] = E.data[(i * V / n) * d + j];
    for (const auto& [layer, head] : path) {
        const HeadWeights w = head_weights(m, layer, head);
        X = matmul(matmul(X, w.W_V, true), w.W_O, true);
    }
    Tensor U = m.unembed().to_tensor();
    if (m.cfg().norm != "none" && m.cfg().norm_position != "post")
        for (int64_t v = 0; v < V; ++v)
            for (int64_t j = 0; j < d; ++j) U.data[v * d + j] *= m.weight("ln_final.weight").f32().data[j];
    const Tensor logits = matmul(X, U, true);
    int64_t hits = 0;
    for (int64_t i = 0; i < n; ++i) hits += topk(logits.row_view(i), 1)[0] == i * V / n;
    return float(hits) / float(n);
}

// circuits: factored OV paths = the direct product (up to rounding at near-ties); the all-pairs
// scores = one path at a time, bit for bit; nothing depends on the thread count.
static void test_circuits_fast_paths(const Model& m) {
    const int64_t n = 40, H = m.cfg().n_heads, N = 2 * H;
    const Tensor single = ov_copying_scores(m, n), pair = ov_pair_copying_scores(m, n);
    CHECK(pair.shape == (Shape{N, N}));
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b) {
            const int64_t la = a / H, ha = a % H, lb = b / H, hb = b % H;
            if (la >= lb) {
                CHECK(std::isnan(pair.data[a * N + b]));
                continue;
            }
            CHECK(pair.data[a * N + b] == ov_path_copying(m, {{la, ha}, {lb, hb}}, n));
            CHECK_NEAR(pair.data[a * N + b], direct_ov_path(m, {{la, ha}, {lb, hb}}, n),
                       1.5 / n);  // a near-tie may flip
        }
    for (int64_t l = 0; l < 2; ++l)
        for (int64_t h = 0; h < H; ++h) {
            CHECK(single.at({l, h}) == ov_path_copying(m, {{l, h}}, n));
            CHECK_NEAR(single.at({l, h}), direct_ov_path(m, {{l, h}}, n), 1.5 / n);
        }
    auto same = [](const Tensor& x, const Tensor& y) {  // bit for bit, NaN where the other is NaN
        if (x.shape != y.shape) return false;
        for (size_t i = 0; i < x.data.size(); ++i)
            if (!(x.data[i] == y.data[i] || (std::isnan(x.data[i]) && std::isnan(y.data[i])))) return false;
        return true;
    };
    set_num_threads(1);
    const Composition c1 = composition_scores(m);
    const float base1 = composition_random_baseline(m, 8);
    const Tensor single1 = ov_copying_scores(m, n), pair1 = ov_pair_copying_scores(m, n);
    set_num_threads(4);
    const Composition c4 = composition_scores(m);
    CHECK(same(c1.q, c4.q) && same(c1.k, c4.k) && same(c1.v, c4.v));
    CHECK(composition_random_baseline(m, 8) == base1);
    CHECK(same(single1, single) && same(pair1, pair));
    set_num_threads(0);
    CHECK(composition_random_baseline(m, 8, 1) != base1);  // the seed matters
}

static void test_copying_and_positional() {
    // A head whose OV circuit is the identity copies every token.
    Model m = random_model(tiny([](ModelConfig& c) {
        c.mlp = "none";
        c.norm = "none";
        c.positional = "learned";
        c.n_heads = 1;
        c.d_head = 16;
    }));
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    for (int64_t l = 0; l < 2; ++l) {
        const std::string p = std::format("blocks.{}.attn.", l);
        Tensor I({16, 16});
        for (int64_t i = 0; i < 16; ++i) I.at({i, i}) = 1.f;
        w[p + "v_proj.weight"] = I;
        w[p + "o_proj.weight"] = I;
    }
    Model id(m.cfg(), w);
    const Tensor copy = ov_copying_scores(id);
    CHECK(copy.data[0] > 0.9f);
    CHECK_NEAR(ov_path_copying(id, {{0, 0}, {1, 0}}), copy.data[0], 1e-6);  // identity twice is still identity
    CHECK_THROWS(ov_path_copying(id, {{1, 0}, {0, 0}}));
    CHECK(ov_path_copying(random_model(tiny()), {{0, 1}}) < 0.5f);
    const Tensor pqk = positional_qk(id, 0, 0, 8);
    CHECK(pqk.shape == (Shape{8, 8}));
    CHECK_NEAR(pqk.at({0, 0}), 1.0, 1e-6);
    CHECK(pqk.at({2, 5}) == 0.f);
    CHECK_THROWS(positional_qk(random_model(tiny()), 0, 0, 8));
}

static void test_post_norm_dla_refused() {
    Model m = random_model(tiny([](ModelConfig& c) {
        c.norm_position = "post";
        c.norm = "layernorm";
    }));
    CHECK_THROWS(direct_logit_attribution(m, seq(5, 1), 4, 1));
    // A single post-norm layer in a pre-norm model breaks additivity just the same.
    Model mixed = random_model(tiny([](ModelConfig& c) {
        c.norm = "layernorm";
        c.layer_overrides[1] = json::parse(R"({"norm_position": "post"})");
    }));
    CHECK_THROWS(direct_logit_attribution(mixed, seq(5, 1), 4, 1));
}

// The vectorizable exp, erf and tanh (src/model/vmath.h) against double precision, at every 16th float
// of their working range, plus special values.
// --- fp16 weights ----------------------------------------------------------------------------

// fp16 <-> float32: every fp16 value survives the round trip; rounding is to nearest, ties to even;
// the bulk (vectorized) conversions match the scalar ones, and on arm64 the hardware's.
static void test_half_conversion() {
    using detail::float_to_half;
    using detail::half_to_float;
    int64_t bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = half_to_float(static_cast<uint16_t>(h));
        if (std::isnan(f)) bad += !std::isnan(half_to_float(float_to_half(f)));
        else bad += float_to_half(f) != h;
    }
    CHECK(bad == 0);
    CHECK(float_to_half(1.f + std::ldexp(1.f, -11)) == 0x3C00);      // tie: to even (down)
    CHECK(float_to_half(1.f + 3 * std::ldexp(1.f, -11)) == 0x3C02);  // tie: to even (up)
    CHECK(float_to_half(65504.f) == 0x7BFF);
    CHECK(float_to_half(65519.f) == 0x7BFF);
    CHECK(float_to_half(65520.f) == 0x7C00);  // rounds past the largest value: infinity
    CHECK(float_to_half(-1e9f) == 0xFC00);
    CHECK(float_to_half(std::ldexp(1.f, -24)) == 0x0001);  // smallest subnormal
    CHECK(float_to_half(std::ldexp(1.f, -25)) == 0x0000);  // half of it: tie to even (zero)
    CHECK(float_to_half(std::ldexp(1.0001f, -25)) == 0x0001);
    CHECK(float_to_half(std::ldexp(1.f, -14)) == 0x0400);  // smallest normal
    CHECK(detail::fits_half(65504.f) && !detail::fits_half(65504.5f) && !detail::fits_half(NAN) &&
          !detail::fits_half(INFINITY));

    std::mt19937_64 rng(3);
    std::uniform_real_distribution<float> expo(-30.f, 18.f);
    std::vector<float> f(1003);
    for (float& v : f) v = std::exp2(expo(rng)) * (rng() & 1 ? 1.f : -1.f);
    f[5] = NAN;
    f[6] = INFINITY;
    f[7] = 7e4f;
    f[8] = 0.f;
    f[9] = -0.f;
    std::vector<uint16_t> h(f.size());
    int64_t over = 0;
    for (float v : f) over += !detail::fits_half(v);
    CHECK(detail::floats_to_halves(f.data(), h.data(), std::ssize(f)) == over);
    bool same = true, hw = true;
    for (size_t i = 0; i < f.size(); ++i) {
        if (i != 5) same &= h[i] == float_to_half(f[i]);
#if defined(__aarch64__)
        const __fp16 x = static_cast<__fp16>(f[i]);
        if (i != 5) hw &= std::bit_cast<uint16_t>(x) == float_to_half(f[i]);
#endif
    }
    CHECK(same);
    CHECK(hw);
    std::vector<float> back(h.size());
    detail::halves_to_floats(h.data(), back.data(), std::ssize(h));
    bool same_back = true;
    for (size_t i = 0; i < h.size(); ++i)
        same_back &= std::isnan(back[i]) ? std::isnan(half_to_float(h[i])) : back[i] == half_to_float(h[i]);
    CHECK(same_back);
}

// Weight in fp16: reads give the rounded values; linear() and matmul() with an fp16 weight equal
// the float32 versions on the rounded values (also across the blocks they convert in).
static void test_fp16_weight() {
    std::mt19937_64 rng(4);
    std::normal_distribution<float> nd;
    Tensor W({3000, 200}), b({3000}), x({7, 200}), A({5, 3000});
    for (Tensor* t : {&W, &b, &x, &A})
        for (float& v : t->data) v = nd(rng);
    const Weight w = Weight::fp16(W);
    CHECK(w.is_fp16() && w.shape == W.shape && w.bytes() == 2 * W.numel());
    const Tensor R = w.to_tensor();
    bool rounded = true;
    for (int64_t i = 0; i < W.numel(); ++i)
        rounded &= R.data[i] == detail::half_to_float(detail::float_to_half(W.data[i]));
    CHECK(rounded);
    CHECK(max_abs_diff(R, W) > 0.f);
    CHECK(max_abs_diff(R, W) <= std::ldexp(1.f, -11) * max_abs(W));
    CHECK(w.rows(10, 13).data == rows(R, 10, 13).data);
    CHECK(w.cols(150, 200).data == cols(R, 150, 200).data);
    CHECK(w.row(2999).data == row(R, 2999).data);
    CHECK_THROWS(w.f32());
    CHECK_THROWS(w.rows(5, 3001));
    const long page = sysconf(_SC_PAGESIZE);
    CHECK(reinterpret_cast<uintptr_t>(w.f16()) % static_cast<uintptr_t>(page) == 0);
    CHECK(w.capacity() % static_cast<size_t>(page) == 0 && w.capacity() >= 2 * static_cast<size_t>(W.numel()));
    CHECK(w.f32_data() == nullptr && w.data() == w.f16());
    const Weight bw(b);
    CHECK(!bw.is_fp16() && bw.f32().data == b.data && bw.data() == nullptr);
    // float32 matrices are page-aligned too (the GPU uses them in place), and copies share them.
    const Weight w32(W);
    CHECK(!w32.is_fp16() && w32.f16() == nullptr && w32.f32_data() == w32.data());
    CHECK(reinterpret_cast<uintptr_t>(w32.data()) % static_cast<uintptr_t>(page) == 0);
    CHECK(w32.capacity() % static_cast<size_t>(page) == 0 && w32.capacity() >= 4 * static_cast<size_t>(W.numel()));
    CHECK(w32.to_tensor().data == W.data && w32.rows(10, 13).data == rows(W, 10, 13).data);
    CHECK(w32.cols(150, 200).data == cols(W, 150, 200).data && w32.row(5).data == row(W, 5).data);
    CHECK_THROWS(w32.f32());  // 2-D: read with rows(), cols(), to_tensor()
    const Weight w32copy = w32;
    CHECK(w32copy.data() == w32.data());
    {  // the library's multiply from 17 rows (fewer: direct_rows, checked by test_linear_few_rows)
        Tensor x17({17, 200});
        for (int64_t i = 0; i < x17.numel(); ++i)
            x17.data[i] = x.data[i % x.numel()] * (1 + 0.01f * float(i / x.numel()));
        CHECK(linear(x17, w32, &bw).data == linear(x17, W, &b).data);
    }
    CHECK(matmul(A, w32).data == matmul(A, W).data);
    CHECK(Weight(Tensor({0, 4})).to_tensor().shape == Shape({0, 4}));
    CHECK_THROWS(Weight::fp16(b));  // only 2-D weights are stored in fp16
    // From fp16 bit patterns, as an F16 tensor in a file: taken as they are.
    std::vector<uint16_t> bits(static_cast<size_t>(W.numel()));
    detail::floats_to_halves(W.ptr(), bits.data(), W.numel());
    const Weight wbits = Weight::fp16_bits(W.shape, bits.data());
    CHECK(wbits.is_fp16() && wbits.to_tensor().data == R.data);
    CHECK(reinterpret_cast<uintptr_t>(wbits.data()) % static_cast<uintptr_t>(page) == 0);
    CHECK_THROWS(Weight::fp16_bits({W.numel()}, bits.data()));  // 1-D
    bits[3] = 0x7C00;                                           // infinity
    CHECK_THROWS(Weight::fp16_bits(W.shape, bits.data()));
    bits[3] = 0x7E00;  // NaN
    CHECK_THROWS(Weight::fp16_bits(W.shape, bits.data()));
    const Tensor want = linear(x, R, &b), got = linear(x, w, &bw);
    CHECK(got.shape == want.shape);
    CHECK(max_abs_diff(got, want) <= 1e-5f * std::max(1.f, max_abs(want)));
    const Tensor want_m = matmul(A, R), got_m = matmul(A, w);
    CHECK(got_m.shape == want_m.shape);
    CHECK(max_abs_diff(got_m, want_m) <= 1e-5f * std::max(1.f, max_abs(want_m)));
    CHECK_THROWS(linear(Tensor({2, 3}), w));
    CHECK_THROWS(matmul(Tensor({2, 3}), w));
    Tensor big = W;
    big.data[7] = -7e4f;
    CHECK(!Weight::fits_fp16(big) && Weight::fits_fp16(W));
    CHECK_THROWS(Weight::fp16(big));
    big.data[7] = NAN;
    CHECK(!Weight::fits_fp16(big));
}

// The same model in float32, with every weight rounded to fp16 values.
static Model rounded_copy(const Model& m) {
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    return Model(m.cfg(), w);
}

// A model with fp16 weights computes what the float32 model with the rounded weights computes.
static void test_fp16_model(const Model& m16) {
    CHECK(m16.weight_format() == WeightFormat::fp16);
    int64_t bytes = 0, bytes32 = 0;
    for (const auto& [name, w] : m16.weights()) {
        CHECK(w.is_fp16() == (w.dim() == 2));
        bytes += w.numel() * (w.dim() == 2 ? 2 : 4);
        bytes32 += w.numel() * 4;
    }
    CHECK(m16.weight_bytes() == bytes && bytes < bytes32);
    CHECK(m16.fp32_weights().empty());
    const Model r = rounded_copy(m16);
    const Tokens t = seq(12, 7, m16.cfg().vocab_size);
    const Tensor a = m16.forward(t), want = r.forward(t);
    CHECK(max_abs_diff(a, want) <= 1e-5f * std::max(1.f, max_abs(want)));
    auto dlogits = [](const Tensor& logits) {
        Tensor g(logits.shape);
        for (int64_t i = 0; i < g.numel(); ++i) g.data[i] = std::sin(0.37f * float(i));
        return g;
    };
    ActivationCache g16, gr;
    m16.run_with_gradients(t, dlogits, g16);
    r.run_with_gradients(t, dlogits, gr);
    CHECK(g16.size() == gr.size());
    for (const auto& [name, t] : gr)
        CHECK(g16.contains(name) && max_abs_diff(g16.at(name), t) <= 1e-4f * std::max(1.f, max_abs(t)));
    const Model copy = m16;  // copies share the fp16 values
    CHECK(copy.forward(t).data == a.data);
    CHECK(copy.weight("embed.weight").f16() == m16.weight("embed.weight").f16() && copy.weight("embed.weight").f16());
    // A weight with a value outside fp16's range stays float32.
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m16.weights()) w[name] = weight.to_tensor();
    w["embed.weight"].data[3] = 1e5f;
    const Model big(m16.cfg(), w, WeightFormat::fp16);
    CHECK(!big.weight("embed.weight").is_fp16());
    CHECK(big.fp32_weights() == std::vector<std::string>{"embed.weight"});
}

// A safetensors file of `tensors` with `cfg_json` as its config; dtype(name) picks "F32", "F16" or
// "BF16" (rounded to nearest even) for each tensor. aligned: as indranet_hf writes files, the data and
// every matrix start on a 16 KiB boundary, with __padding tensors filling the gaps.
static void write_safetensors(const std::string& path, const std::string& cfg_json,
                              const std::map<std::string, Tensor>& tensors,
                              const std::function<std::string(const std::string&)>& dtype, bool aligned = false) {
    constexpr size_t kAlign = 16384;
    std::string escaped;
    for (char ch : cfg_json) {
        if (ch == '"' || ch == '\\') escaped += '\\';
        escaped += ch;
    }
    std::string header = "{\"__metadata__\": {\"config\": \"" + escaped + "\"}", data;
    int pads = 0;
    for (const auto& [name, t] : tensors) {
        const std::string type = dtype(name);
        if (aligned && t.dim() == 2 && data.size() % kAlign) {
            const size_t at = data.size(), pad = kAlign - at % kAlign;
            data.append(pad, '\0');
            header +=
                std::format(", \"__padding{}\": {{\"dtype\": \"U8\", \"shape\": [{}], \"data_offsets\": [{}, {}]}}",
                            pads++, pad, at, data.size());
        }
        const size_t begin = data.size();
        for (float v : t.data) {
            if (type == "F32") {
                data.append(reinterpret_cast<const char*>(&v), 4);
                continue;
            }
            uint16_t hb;
            if (type == "F16") {
                hb = detail::float_to_half(v);
            } else {
                const auto x = std::bit_cast<uint32_t>(v);
                hb = static_cast<uint16_t>((x + 0x7FFF + ((x >> 16) & 1)) >> 16);
            }
            data.append(reinterpret_cast<const char*>(&hb), 2);
        }
        std::string shape;
        for (int64_t d : t.shape) shape += std::format("{}{}", (shape.empty() ? "" : ", "), d);
        header += std::format(", \"{}\": {{\"dtype\": \"{}\", \"shape\": [{}], \"data_offsets\": [{}, {}]}}", name,
                              type, shape, begin, data.size());
    }
    header += "}";
    if (aligned) header.append((kAlign - (8 + header.size()) % kAlign) % kAlign, ' ');
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = header.size();
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(n >> (8 * i));
    f.write(reinterpret_cast<const char*>(len), 8);
    f << header << data;
}

// What a file stores for v in `dtype`, read back as float32.
static float stored(float v, const std::string& dtype) {
    if (dtype == "F16") return detail::half_to_float(detail::float_to_half(v));
    if (dtype == "BF16") {
        const auto x = std::bit_cast<uint32_t>(v);
        v = std::bit_cast<float>(((x + 0x7FFF + ((x >> 16) & 1)) >> 16) << 16);
    }
    return v;
}

// Model::load reads each 2-D tensor straight into its weight, converting as it reads when the file's
// type differs from the one kept in memory: the same model as building it from the file's values.
static void check_load_layout(const std::string& path, const std::string& cfg_json, const ModelConfig& cfg,
                              std::map<std::string, Tensor> w,
                              const std::function<std::string(const std::string&)>& dtype, bool aligned) {
    write_safetensors(path, cfg_json, w, dtype, aligned);
    for (auto& [name, weight] : w)
        for (float& v : weight.data) v = stored(v, dtype(name));  // what the file holds
    const Model want16(cfg, w, WeightFormat::fp16), want32(cfg, w);
    const Model got16 = Model::load(path, WeightFormat::fp16), got32 = Model::load(path);
    std::remove(path.c_str());
    CHECK(got16.weight_format() == WeightFormat::fp16 && got32.weight_format() == WeightFormat::fp32);
    CHECK(got16.weights().size() == want16.weights().size() && got32.weights().size() == want32.weights().size());
    bool same = true;
    for (const auto& [name, w] : want16.weights()) {
        const Weight* g = got16.weight_or_null(name);
        same &= g && g->is_fp16() == w.is_fp16() && g->to_tensor().data == w.to_tensor().data;
        const Weight* g32 = got32.weight_or_null(name);
        same &= g32 && !g32->is_fp16() && g32->to_tensor().data == want32.weight(name).to_tensor().data;
    }
    CHECK(same);
    CHECK(got16.fp32_weights() == want16.fp32_weights());
    // Aligned files: a matrix stored in the type kept in memory (and, for fp16, all finite) is used in
    // place from the mapping; every other one is a copy. Unaligned files: always copies.
    bool borrowed_right = true;
    for (const auto& [name, w] : got16.weights()) {
        if (w.dim() != 2) continue;
        const std::string type = dtype(name);
        borrowed_right &= w.is_borrowed() == (aligned && type == "F16" && w.is_fp16());
        borrowed_right &= got32.weight(name).is_borrowed() == (aligned && type == "F32");
    }
    CHECK(borrowed_right);
    const Tokens t = seq(10, 2, cfg.vocab_size);  // the file is gone by now: the mapping outlives it
    CHECK(got16.forward(t).data == want16.forward(t).data);
    CHECK(got32.forward(t).data == want32.forward(t).data);
}

static void check_load(const std::string& path, const std::string& cfg_json, const ModelConfig& cfg,
                       const std::map<std::string, Tensor>& w,
                       const std::function<std::string(const std::string&)>& dtype) {
    check_load_layout(path, cfg_json, cfg, w, dtype, false);
    check_load_layout(path, cfg_json, cfg, w, dtype, true);
}

static void test_fp16_load() {
    const std::string path = std::format("/tmp/tfi_unit_fp16_{}.safetensors", ::getpid());
    const ModelConfig cfg = tiny([](ModelConfig& c) {
        c.bias = true;
        c.norm = "layernorm";
    });
    const std::string cfg_json = R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                                 R"("mlp_multiple_of": 8, "bias": true, "norm": "layernorm"})";
    std::map<std::string, Tensor> w;
    const Model m = random_model(cfg, 9);
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    // F16 embedding; a BF16 matrix and bias; the rest F32, one matrix with a value outside fp16's range
    // (with fp16, read again as float32).
    w["blocks.1.attn.q_proj.weight"].data[5] = 1e5f;
    auto mixed = [](const std::string& n) -> std::string {
        if (n == "embed.weight") return "F16";
        if (n == "blocks.0.mlp.up.weight" || n == "blocks.0.mlp.up.bias") return "BF16";
        return "F32";
    };
    check_load(path, cfg_json, cfg, w, mixed);
    // Everything in fp16, and everything in BF16.
    w["blocks.1.attn.q_proj.weight"].data[5] = 0.5f;
    check_load(path, cfg_json, cfg, w, [](const std::string&) { return std::string("F16"); });
    check_load(path, cfg_json, cfg, w, [](const std::string&) { return std::string("BF16"); });
    // An F16 tensor holding an infinity is not taken as fp16: it stays float32, as a float32 weight
    // outside fp16's range does.
    w["embed.weight"].data[7] = INFINITY;
    check_load(path, cfg_json, cfg, w, mixed);
    write_safetensors(path, cfg_json, w, mixed);
    const Model inf16 = Model::load(path, WeightFormat::fp16);
    std::remove(path.c_str());
    CHECK(!inf16.weight("embed.weight").is_fp16() && std::isinf(inf16.weight("embed.weight").to_tensor().data[7]));
    CHECK(inf16.fp32_weights() == std::vector<std::string>{"embed.weight"});
    CHECK(inf16.weight("blocks.0.attn.q_proj.weight").is_fp16());
    // An embedding of more than one conversion block (2^20 values), F32 and F16 in the file.
    const ModelConfig big = tiny([](ModelConfig& c) {
        c.vocab_size = 70000;
        c.n_layers = 1;
    });
    const std::string big_json =
        R"({"vocab_size": 70000, "d_model": 16, "n_layers": 1, "n_heads": 4, "max_seq_len": 32, )"
        R"("mlp_multiple_of": 8})";
    std::map<std::string, Tensor> wb;
    const Model mb = random_model(big, 3);
    for (const auto& [name, w] : mb.weights()) wb[name] = w.to_tensor();
    CHECK(wb.at("embed.weight").numel() > (int64_t(1) << 20));
    wb["embed.weight"].data[(int64_t(1) << 20) + 3] = -7e4f;  // in the second block: fp16 keeps it float32
    check_load(path, big_json, big, wb, [](const std::string&) { return std::string("F32"); });
    wb["embed.weight"].data[(int64_t(1) << 20) + 3] = 0.25f;
    check_load(path, big_json, big, wb, [](const std::string&) { return std::string("F16"); });
}

// Every weight must have the shape the config implies: a model with any one weight a row (or an
// element) short, or missing, is refused when it is built, not read out of bounds later.
// bias as parts: parsed from true / false / a list; a model has exactly the biases its config names
// (a stray one is refused rather than used unannounced); and a qkv-only model (Qwen2's) works.
static void test_bias_parts() {
    Biases b;
    b = true;
    CHECK(b.str() == "all" && b.has("norm"));
    CHECK(Biases::from_json(json::parse("false")).str() == "none");
    CHECK(Biases::from_json(json::parse(R"(["mlp", "qkv"])")).str() == "qkv,mlp");
    CHECK_THROWS(Biases::from_json(json::parse(R"(["qkv", "qkv"])")));
    CHECK_THROWS(Biases::from_json(json::parse(R"(["attn"])")));
    CHECK_THROWS(Biases::from_json(json::parse("1")));

    const ModelConfig qkv = tiny([](ModelConfig& c) { c.bias = Biases::from_json(json::parse(R"(["qkv"])")); });
    const Model m = random_model(qkv, 3);
    CHECK(m.weights().contains("blocks.0.attn.k_proj.bias") && !m.weights().contains("blocks.0.attn.o_proj.bias") &&
          !m.weights().contains("blocks.0.mlp.up.bias"));
    // q, k and v biases change the output (they are used), and match the computation spelled out
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    auto zeroed = w;
    for (auto& [name, w] : zeroed)
        if (name.find("_proj.bias") != std::string::npos) std::ranges::fill(w.data, 0.f);
    const Tokens t = seq(9, 4);
    CHECK(max_abs_diff(Model(qkv, zeroed).forward(t), m.forward(t)) > 1e-3f);
    // a bias the config doesn't name: refused, whichever part it belongs to
    for (const char* extra : {"blocks.0.attn.o_proj.bias", "blocks.1.mlp.down.bias"}) {
        auto more = w;
        more[extra] = Tensor({16}, std::vector<float>(16, 0.f));
        CHECK_THROWS(Model(qkv, more));
    }
    auto fewer = w;
    fewer.erase("blocks.1.attn.v_proj.bias");
    CHECK_THROWS(Model(qkv, fewer));
    // a layer's own setting decides for that layer
    const ModelConfig per_layer =
        tiny([](ModelConfig& c) { c.layer_overrides[1] = json::parse(R"({"bias": ["o"]})"); });
    const Model pl = random_model(per_layer, 5);
    CHECK(pl.weights().contains("blocks.1.attn.o_proj.bias") && !pl.weights().contains("blocks.0.attn.o_proj.bias"));
}

// The tokenizer's parts on their own (the whole tokenizer is checked against Hugging Face's
// tokenizers library by python/tests/test_hf.py): UTF-8, NFC / NFD, the split-pattern regex, and BPE
// rules on small hand-written tokenizer.json files.
static void test_tokenizer_parts() {
    using namespace detail;
    // UTF-8: round trip, and invalid input refused (stray continuation, overlong, surrogate, truncated)
    const std::string mixed = "a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80";  // a é 中 😀
    CHECK(utf8_decode(mixed) == U32String({U'a', 0xE9, 0x4E2D, 0x1F600}) && utf8_encode(utf8_decode(mixed)) == mixed);
    for (const char* bad : {"\x80", "\xC0\xAF", "\xED\xA0\x80", "\xE4\xB8"}) CHECK_THROWS(utf8_decode(bad));
    // categories and White_Space
    CHECK(std::string(unicode_category(U'A')) == "Lu" && unicode_letter(0x4E2D) && unicode_number(0x2163) &&
          !unicode_letter(U'1'));
    CHECK(unicode_white_space(0x3000) && unicode_white_space(0x85) && !unicode_white_space(0x200B));
    // NFC / NFD: composition, Hangul, mark order, an excluded composition, and blocking
    CHECK(unicode_nfc(U"é") == U"é" && unicode_nfd(U"é") == U"é");
    CHECK(unicode_nfc(U"각") == U"각" && unicode_nfd(U"각") == U"각");
    CHECK(unicode_nfd(U"ạ́") == U"ạ́");  // dot below (220) before acute (230)
    CHECK(unicode_nfc(U"ạ́") == U"ạ́");  // ạ then the acute
    CHECK(unicode_nfc(U"क़") == U"क़");  // excluded: stays decomposed
    CHECK(unicode_nfc(U"plain ascii") == U"plain ascii");

    // regex: the subset tokenizer patterns use
    auto spans = [](const char* pattern, const char32_t* text) {
        std::vector<std::u32string> out;
        const U32String s(text);
        for (const auto& [begin, end] : Regex(pattern).find_all(s)) out.push_back(s.substr(begin, end - begin));
        return out;
    };
    using V = std::vector<std::u32string>;
    CHECK(spans(R"(\p{N}{1,3})", U"12345") == (V{U"123", U"45"}));
    CHECK(spans(R"((?i:'s|'t)|\p{L}+)", U"IT'S it's") == (V{U"IT", U"'S", U"it", U"'s"}));
    CHECK(spans(R"(\s+(?!\S)|\s+)", U"a   b") == (V{U"  ", U" "}));  // the lookahead leaves one space for " b"
    CHECK(spans(R"([^\r\n\p{L}\p{N}]?\p{L}+)", U"中文 été") == (V{U"中文", U" été"}));
    CHECK(spans(R"(a+?b|a)", U"aab") == (V{U"aab"}));
    CHECK(spans(R"(x\x{263A}é[^a-c])", U"x☺éd") == (V{U"x☺éd"}));
    CHECK(spans(R"(\s+)", std::u32string(100000, U' ').c_str()).size() == 1);  // long runs: no deep recursion
    for (const char* bad : {"(a", "a{2,1}", "\\q", "(?<name>a)", "a++", "[b-a]", "^a"}) CHECK_THROWS(Regex(bad));

    // BPE: merges by rank (leftmost on ties), byte fallback, unknowns, specials around the text
    auto tokenizer = [](const std::string& extra, const std::string& model_extra = "") {
        return Tokenizer::from_json(
            R"({"model": {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "bc": 5, "abc": 6,
            "<0xC3>": 7, "<0xA9>": 8, "▁": 9, "<s>": 10, "aa": 11}, "merges": ["a b", "ab c", "b c", "a a"])" +
            model_extra + "}" + extra + "}");
    };
    auto ids = [](const Tokenizer& t, const std::string& s, bool special = true) { return t.encode(s, special); };
    const Tokenizer plain = tokenizer("");
    CHECK(ids(plain, "abc") == (Tokens{6}) && ids(plain, "bca") == (Tokens{5, 1}) &&
          ids(plain, "aaa") == (Tokens{11, 1}));
    CHECK(ids(plain, "axz") == (Tokens{1}));  // no unk_token and no fallback: unknown characters dropped
    const Tokenizer unk = tokenizer("", R"(, "unk_token": "<unk>", "fuse_unk": true)");
    CHECK(ids(unk, "axza") == (Tokens{1, 0, 1}));
    const Tokenizer fallback = tokenizer("", R"(, "unk_token": "<unk>", "byte_fallback": true)");
    CHECK(ids(fallback, "a\xC3\xA9z") == (Tokens{1, 7, 8, 0}));  // é as its two bytes; z has no bytes in the vocabulary
    const Tokenizer templ =
        tokenizer(R"(, "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>"}},
        {"Sequence": {"id": "A"}}], "special_tokens": {"<s>": {"id": "<s>", "ids": [10], "tokens": ["<s>"]}}})");
    CHECK(ids(templ, "ab") == (Tokens{10, 4}) && ids(templ, "ab", false) == (Tokens{4}));
    // Metaspace "first": the replacement prepended only at the start of the input, not after an added token
    const Tokenizer meta =
        tokenizer(R"(, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first",
        "split": false}, "added_tokens": [{"id": 10, "content": "<s>", "special": true}])");
    CHECK(ids(meta, "a b") == (Tokens{9, 1, 9, 2}) && ids(meta, "<s>a") == (Tokens{10, 1}));
    // added tokens: leftmost-longest, rstrip eats the space after, single_word needs word boundaries
    const Tokenizer added =
        tokenizer(R"(, "added_tokens": [{"id": 20, "content": "XY"}, {"id": 21, "content": "XYZ", "rstrip": true},
        {"id": 22, "content": "Q", "single_word": true}, {"id": 99, "content": "abc"}])");
    // ids as tokenizers assigns them: past the vocabulary (12 entries) in order, whatever the file says
    CHECK(ids(added, "aXYZ  b") == (Tokens{1, 13, 2}) && ids(added, "XYa") == (Tokens{12, 1}));
    CHECK(ids(added, "aQ Q") == (Tokens{1, 14}));  // inside a word Q is not the token (and is dropped)
    CHECK(ids(added, "cabc") == (Tokens{3, 6}));   // text in the vocabulary: the vocabulary's id
    // ignore_merges: a word in the vocabulary is that token, even when the merges wouldn't reach it
    // (Llama 3); Digits: each digit alone, or runs of them
    auto small = [](const std::string& model_extra, const std::string& pre = "null") {
        return Tokenizer::from_json(R"({"pre_tokenizer": )" + pre +
                                    R"(, "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2,
            "bc": 3, "abc": 4, "1": 5, "2": 6, "12": 7}, "merges": ["b c", "1 2"])" +
                                    model_extra + "}}");
    };
    CHECK(ids(small(""), "abc") == (Tokens{0, 3}) && ids(small(R"(, "ignore_merges": true)"), "abc") == (Tokens{4}));
    CHECK(ids(small("", R"({"type": "Digits", "individual_digits": true})"), "a12") == (Tokens{0, 5, 6}));
    CHECK(ids(small("", R"({"type": "Digits", "individual_digits": false})"), "a12") == (Tokens{0, 7}));
    // what the engine doesn't implement is refused by name
    CHECK_THROWS(Tokenizer::from_json(R"({"model": {"type": "Unigram", "vocab": []}})"));
    CHECK_THROWS(tokenizer(R"(, "normalizer": {"type": "NFKC"})"));
    CHECK_THROWS(tokenizer(R"(, "pre_tokenizer": {"type": "Whitespace"})"));
    CHECK_THROWS(Tokenizer::from_json(R"({"model": {"type": "BPE", "vocab": {"a": 0}, "merges": ["a b"]}})"));
}

static void test_weight_shapes_checked() {
    for (const auto& [name, cfg] : configs()) {
        const Model m = random_model(cfg, 1);
        std::map<std::string, Tensor> w;
        for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
        bool all = true;
        for (const auto& e : w) {
            auto shorter = w;
            Tensor& t = shorter[e.first];
            Shape s = t.shape;
            s[0] -= 1;
            t = Tensor(s, std::vector<float>(t.data.begin(), t.data.begin() + shape_numel(s)));
            bool threw = false;
            try {
                Model bad(cfg, shorter);
            } catch (const std::runtime_error&) { threw = true; }
            if (!threw)
                std::printf("     %s: accepted %s with shape %s\n", name.c_str(), e.first.c_str(),
                            shape_str(s).c_str());
            all &= threw;
            auto missing = w;
            missing.erase(e.first);
            threw = false;
            try {
                Model bad(cfg, missing);
            } catch (const std::runtime_error&) { threw = true; }
            if (!threw) std::printf("     %s: accepted a model without %s\n", name.c_str(), e.first.c_str());
            all &= threw;
        }
        CHECK(all);
    }
}

// The portable half of chunked GPU work: packing groups under a budget, and running chunks through a
// fake backend that logs every call and fails on command. Every chunk is submitted before the first
// wait; only failed chunks run again, each up to twice; a third failure throws GpuFailure.
static void test_gpu_chunks() {
    using detail::Chunk;
    auto same = [](const std::vector<Chunk>& a, const std::vector<std::pair<size_t, size_t>>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].first != b[i].first || a[i].last != b[i].second) return false;
        return true;
    };
    CHECK(same(detail::pack_chunks({1, 1, 1, 1}, 2), {{0, 2}, {2, 4}}));
    CHECK(same(detail::pack_chunks({1, 5, 1, 1}, 2), {{0, 1}, {1, 2}, {2, 4}}));  // too big: alone
    CHECK(same(detail::pack_chunks({1, 1, 1}, 100), {{0, 3}}));
    CHECK(same(detail::pack_chunks({0, 0, 3}, 2), {{0, 2}, {2, 3}}));
    CHECK(detail::pack_chunks({}, 2).empty());

    struct Fake : detail::ChunkBackend {
        std::vector<std::string> log;      // "s<first>" per submit, "w<first>" per wait
        std::vector<size_t> ticket_chunk;  // ticket -> chunk's first group
        std::map<size_t, int> fail;        // chunk's first group -> failures left
        size_t submit(const Chunk& c) override {
            log.push_back(std::format("s{}", c.first));
            ticket_chunk.push_back(c.first);
            return ticket_chunk.size() - 1;
        }
        bool wait(size_t t) override {
            const size_t c = ticket_chunk.at(t);
            log.push_back(std::format("w{}", c));
            if (fail[c] > 0) {
                --fail[c];
                return false;
            }
            return true;
        }
    };
    const std::vector<Chunk> chunks = {{0, 2}, {2, 3}, {3, 5}};
    {
        Fake f;
        detail::run_chunks(f, chunks);
        CHECK((f.log == std::vector<std::string>{"s0", "s2", "s3", "w0", "w2", "w3"}));
    }
    {  // chunk 2 fails once, chunk 3 twice: each runs again after all the first waits, on its own
        Fake f;
        f.fail[2] = 1;
        f.fail[3] = 2;
        detail::run_chunks(f, chunks);
        CHECK((f.log ==
               std::vector<std::string>{"s0", "s2", "s3", "w0", "w2", "w3", "s2", "w2", "s3", "w3", "s3", "w3"}));
    }
    {  // three failures in a row: GpuFailure
        Fake f;
        f.fail[3] = 3;
        bool threw = false;
        try {
            detail::run_chunks(f, chunks);
        } catch (const detail::GpuFailure&) { threw = true; }
        CHECK(threw);
    }
    {  // a test-injected failure counts like a GPU failure, once per request
        Fake f;
        detail::set_gpu_fail_chunks(1);
        detail::run_chunks(f, chunks);
        CHECK((f.log == std::vector<std::string>{"s0", "s2", "s3", "w0", "w2", "w3", "s0", "w0"}));
        detail::set_gpu_fail_chunks(0);
    }
    {  // nothing to run
        Fake f;
        detail::run_chunks(f, {});
        CHECK(f.log.empty());
    }
    CHECK(detail::gpu_chunk_flops() == 1.5e12);
    detail::set_gpu_chunk_flops(7);
    CHECK(detail::gpu_chunk_flops() == 7);
    detail::set_gpu_chunk_flops(0);
}

// d_mlp 0 asks for the default width, as indranet reads it: a config saying 0 (top level or in a layer
// override) takes weights of the default width, plain and gated MLPs alike.
static void test_d_mlp_zero_is_default() {
    for (const char* mlp : {"mlp", "swiglu"}) {
        const std::string base =
            std::string(R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                        R"("mlp_multiple_of": 8, "mlp": ")") +
            mlp + "\"";
        const ModelConfig zero = ModelConfig::from_json(base + R"(, "d_mlp": 0})");
        CHECK(!zero.d_mlp);
        const ModelConfig layer =
            ModelConfig::from_json(base + R"(, "d_mlp": 24, "layer_overrides": {"1": {"d_mlp": 0}}})");
        CHECK(layer.d_mlp && *layer.d_mlp == 24 && !layer.layer_config(1).d_mlp);
        const Model source = random_model(tiny([mlp](ModelConfig& c) { c.mlp = mlp; }), 3);  // default width
        std::map<std::string, Tensor> w;
        for (const auto& [name, weight] : source.weights()) w[name] = weight.to_tensor();
        bool loads = true;
        try {
            const Model m(zero, w);
            CHECK(same_values(m.forward(seq(6, 4)), source.forward(seq(6, 4))));
        } catch (const std::exception& e) {
            loads = false;
            std::printf("     %s, d_mlp 0: %s\n", mlp, e.what());
        }
        CHECK(loads);
    }
}

// Headers whose shapes don't fit the file are refused before anything is allocated or multiplied out:
// a shape whose element count overflows 64 bits (or any product of its leading dimensions does), one
// larger than the file, a negative dimension, a byte count that doesn't match. Zero-size tensors load.
static void test_safetensors_bad_shapes() {
    const std::string path = std::format("/tmp/tfi_unit_shapes_{}.safetensors", ::getpid());
    auto write = [&](const std::string& tensor_json, size_t data_bytes) {
        const std::string header = "{" + tensor_json + "}";
        std::ofstream f(path, std::ios::binary);
        uint64_t n = header.size();
        for (int i = 0; i < 8; ++i) f.put(static_cast<char>((n >> (8 * i)) & 0xff));
        f << header << std::string(data_bytes, '\0');
    };
    auto refused = [&](const std::string& tensor_json, size_t data_bytes, const std::string& why) {
        write(tensor_json, data_bytes);
        std::string error;
        try {
            load_safetensors(path);
        } catch (const std::runtime_error& e) { error = e.what(); }
        if (error.find(why) == std::string::npos)
            std::printf("     %s: got '%s'\n", tensor_json.c_str(), error.c_str());
        CHECK(error.find(why) != std::string::npos);
    };
    refused(R"("x": {"dtype": "F32", "shape": [4294967296, 4294967296], "data_offsets": [0, 0]})", 0,
            "larger than the file");
    refused(R"("x": {"dtype": "F16", "shape": [4611686018427387904, 4], "data_offsets": [0, 16]})", 16,
            "larger than the file");
    refused(R"("x": {"dtype": "F32", "shape": [1099511627776, 1099511627776, 0], "data_offsets": [0, 0]})", 0,
            "larger than the file");
    refused(R"("x": {"dtype": "I64", "shape": [3], "data_offsets": [0, 16]})", 16, "larger than the file");
    refused(R"("x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 20]})", 24, "byte size does not match");
    refused(R"("x": {"dtype": "F32", "shape": [2, -3], "data_offsets": [0, 24]})", 24, "negative dimension");
    write(
        R"("x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}, "e": {"dtype": "F32", "shape": [0, 1099511627776, 1099511627776], "data_offsets": [24, 24]})",
        24);
    const SafeTensors ok = load_safetensors(path);
    CHECK(ok.tensors.at("x").shape == Shape({2, 3}) && ok.tensors.at("x").numel() == 6);
    CHECK(ok.tensors.at("e").numel() == 0);
    std::remove(path.c_str());
    CHECK_THROWS(shape_numel({int64_t{1} << 40, int64_t{1} << 40}));
    CHECK(shape_numel({0, int64_t{1} << 40, int64_t{1} << 40}) == 0);
    CHECK(shape_numel({}) == 1 && shape_numel({3, 4}) == 12);
}

static void test_vector_math() {
    auto ulp_error = [](float got, double want) {
        const float w = static_cast<float>(std::fabs(want));
        const double ulp = w < std::numeric_limits<float>::min() ? std::numeric_limits<float>::denorm_min()
                                                                 : double(std::nextafter(w, INFINITY)) - w;
        return std::fabs(double(got) - want) / ulp;
    };
    // Every 16th float from lo up to hi. Floats in increasing order are consecutive integers: k for
    // the float k steps above zero, -k for k steps below (0 is -0.0, as nextafter from below gives),
    // so jumping 16 at a time visits them without stepping through the 15 in between.
    auto order = [](float x) {
        const auto b = std::bit_cast<uint32_t>(x);
        return (b >> 31) ? -static_cast<int64_t>(b & 0x7FFFFFFFu) : static_cast<int64_t>(b);
    };
    auto from_order = [](int64_t k) {
        const uint32_t b = k <= 0 ? 0x80000000u | static_cast<uint32_t>(-k) : static_cast<uint32_t>(k);
        return std::bit_cast<float>(b);
    };
    auto sweep = [&](float lo, float hi, auto f, auto ref, double max_ulp) {
        double worst = 0;
        for (int64_t k = order(lo), end = order(hi); k <= end; k += 16) {
            const float x = from_order(k);
            worst = std::max(worst, ulp_error(f(x), ref(double(x))));
        }
        CHECK(worst <= max_ulp);
    };
    sweep(-87.3f, 88.7f, [](float x) { return vm::exp(x); }, [](double x) { return std::exp(x); }, 1.5);
    sweep(-6.f, 6.f, [](float x) { return vm::erf(x); }, [](double x) { return std::erf(x); }, 3.0);
    sweep(-12.f, 12.f, [](float x) { return vm::tanh(x); }, [](double x) { return std::tanh(x); }, 2.0);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(vm::exp(-inf) == 0.f && vm::exp(-200.f) == 0.f && vm::exp(inf) == inf && vm::exp(100.f) == inf);
    CHECK(vm::exp(0.f) == 1.f && vm::erf(0.f) == 0.f && vm::tanh(0.f) == 0.f);
    CHECK(vm::erf(inf) == 1.f && vm::erf(-inf) == -1.f && vm::tanh(inf) == 1.f && vm::tanh(-inf) == -1.f);
    CHECK(std::isnan(vm::exp(nan)) && std::isnan(vm::erf(nan)) && std::isnan(vm::tanh(nan)));
    // Softmax: masked (-inf) entries are exactly 0, the rest sum to 1.
    float row[11] = {0.5f, -inf, 3.f, -1.f, -inf, 2.f, 0.f, 1.f, -2.f, 4.f, -inf};
    softmax_(row);
    CHECK(row[1] == 0.f && row[4] == 0.f && row[10] == 0.f);
    double total = 0;
    for (float v : row) total += v;
    CHECK_NEAR(total, 1.0, 1e-6);
    CHECK_NEAR(row[9] / row[2], std::exp(1.0), 1e-5);
}

#ifdef TFI_USE_ACCELERATE
// The GPU linear layer (macOS) against the CPU one, on awkward sizes, with and without a bias.
static void test_gpu_linear() {
    std::unique_ptr<detail::Gpu> gpu = detail::Gpu::create();
    if (!gpu) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    for (const auto& dims :
         std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {7, 13, 5}, {300, 67, 129}, {64, 512, 2048}}) {
        const int64_t m = dims[0], k = dims[1], n = dims[2];
        Tensor x({m, k}), W({n, k}), b({n});
        for (float& v : x.data) v = nd(rng);
        for (float& v : W.data) v = nd(rng);
        for (float& v : b.data) v = nd(rng);
        // A Gpu caches weights by address, so each must outlive the Gpu that saw it: a fresh one per W.
        const Weight Ww(W), bw(b);
        const std::unique_ptr<detail::Gpu> g = detail::Gpu::create();
        for (const Weight* bias : {static_cast<const Weight*>(nullptr), &bw}) {
            const Tensor want = linear(x, W, bias ? &b : nullptr), got = g->linear(x, Ww, bias);
            CHECK(got.shape == want.shape);
            CHECK(max_abs_diff(got, want) <= 1e-4f * std::max(1.f, max_abs(want)));
        }
    }
    CHECK_THROWS(gpu->linear(Tensor({2, 3}), Weight(Tensor({4, 5})), nullptr));

    // fp16 inputs: close to fp32. A value outside fp16's range (in x or in W) falls back to fp32.
    Tensor x({300, 512}), W({257, 512}), b({257});
    for (float& v : x.data) v = nd(rng);
    for (float& v : W.data) v = nd(rng);
    for (float& v : b.data) v = nd(rng);
    const Weight Ww(W), bw(b);
    const Tensor want = linear(x, W, &b);
    CHECK(max_abs_diff(gpu->linear(x, Ww, &bw, /*fp16=*/true), want) <= 1e-2f * std::max(1.f, max_abs(want)));
    CHECK(max_abs_diff(gpu->linear(x, Ww, &bw, true), want) > 0.f);  // really fp16
    Tensor xbig = x, Wbig = W;
    xbig.data[17] = 1e5f;
    Wbig.data[33] = -7e4f;
    const Weight Wbigw(Wbig);
    for (const auto& [xi, Wi] : std::vector<std::pair<const Tensor*, const Weight*>>{{&xbig, &Ww}, {&x, &Wbigw}}) {
        const Tensor exact = linear(*xi, *Wi, &bw), got = gpu->linear(*xi, *Wi, &bw, true);
        CHECK(max_abs(got) < INFINITY);
        CHECK(max_abs_diff(got, exact) <= 1e-4f * std::max(1.f, max_abs(exact)));
    }

    // linear_many (one submission) = separate linear calls, bit for bit: weights of different widths
    // (as Q/K/V with grouped-query attention), with and without bias, an empty weight, and in fp16 a
    // weight outside fp16's range that alone falls back to fp32.
    Tensor Wk({64, 512}), Wv({64, 512}), Wempty({0, 512});
    for (float& v : Wk.data) v = nd(rng);
    for (float& v : Wv.data) v = nd(rng);
    const Weight Wkw(Wk), Wvw(Wv), Wemptyw(Wempty);
    const std::vector<const Weight*> Ws = {&Ww, &Wkw, &Wbigw, &Wemptyw, &Wvw};
    const std::vector<const Weight*> bs = {&bw, nullptr, &bw, nullptr, nullptr};
    for (const bool fp16 : {false, true})
        for (const Tensor* xi : {&x, &xbig}) {
            const std::vector<Tensor> got = gpu->linear_many(*xi, Ws, bs, fp16);
            CHECK(got.size() == Ws.size());
            for (size_t i = 0; i < Ws.size(); ++i) {
                const Tensor want_i = gpu->linear(*xi, *Ws[i], bs[i], fp16);
                CHECK(got[i].shape == want_i.shape);
                CHECK(got[i].data == want_i.data);
            }
        }
    const std::vector<Tensor> nobias = gpu->linear_many(x, {&Ww, &Wkw}, {});
    CHECK(nobias[0].data == gpu->linear(x, Ww, nullptr).data);
    CHECK(nobias[1].data == gpu->linear(x, Wkw, nullptr).data);
    CHECK(gpu->linear_many(Tensor({0, 512}), {&Ww, &Wkw}, {})[1].shape == Shape({0, 64}));
    CHECK_THROWS(gpu->linear_many(x, {&Ww, &Wkw}, {&bw}));  // one bias per weight
    const Weight narrow(Tensor({4, 5}));
    CHECK_THROWS(gpu->linear_many(x, {&Ww, &narrow}, {}));  // wrong input width
}

// Gpu::Seq's operations against the CPU's. Each block creates its Gpu after its tensors, so the Gpu
// (which caches weights by address) is gone before they are.
static void test_gpu_seq() {
    if (!detail::Gpu::create()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    auto near = [](float got, float want, float tol) {
        if (std::isnan(want)) CHECK(std::isnan(got));
        else CHECK_NEAR(got, want, tol * std::max(1.f, std::fabs(want)));
    };
    {  // Activations: the same polynomials as the CPU (vmath.h), over a wide range and NaN.
        Tensor xs({4002});
        for (int64_t i = 0; i < 4001; ++i) xs.data[i] = -20.f + 0.01f * static_cast<float>(i);
        xs.data[4001] = NAN;
        const auto g = detail::Gpu::create();
        detail::Gpu::Seq s(*g);
        const char* names[] = {"relu", "gelu", "gelu_tanh", "silu", "tanh", "relu2"};
        for (int a = 0; a < 6; ++a) {
            Tensor want = xs;
            detail::activate_(want, names[a]);
            std::ranges::copy(xs.data, s.reserve(0, xs.numel()));
            s.activate(0, xs.numel(), static_cast<detail::GpuAct>(a));
            s.run(false);
            for (int64_t i = 0; i < xs.numel(); ++i) near(s.data(0)[i], want.data[i], 1e-6f);
        }
        // glu, add, copy.
        const int64_t n = 1000;
        Tensor gate({n}), up({n});
        for (float& v : gate.data) v = 3.f * nd(rng);
        for (float& v : up.data) v = nd(rng);
        std::ranges::copy(gate.data, s.reserve(1, n));
        std::ranges::copy(up.data, s.reserve(2, n));
        s.reserve(3, n);
        s.reserve(4, n);
        s.glu(1, 2, n, detail::GpuAct::Silu);
        s.add(1, 2, 3, n);
        s.copy(3, 4, n);
        s.run(false);
        Tensor want = gate;
        detail::activate_(want, "silu");
        for (int64_t i = 0; i < n; ++i) {
            near(s.data(1)[i], want.data[i] * up.data[i], 1e-6f);
            CHECK(s.data(3)[i] == s.data(1)[i] + up.data[i]);
            CHECK(s.data(4)[i] == s.data(3)[i]);
        }
    }
    {  // layernorm (with bias) and rmsnorm; rows wider than a threadgroup.
        const int64_t rows = 37, d = 300;
        Tensor x({rows, d}), w({d}), b({d});
        for (float& v : x.data) v = 3.f + 2.f * nd(rng);
        for (float& v : w.data) v = 1.f + 0.2f * nd(rng);
        for (float& v : b.data) v = 0.1f * nd(rng);
        const auto g = detail::Gpu::create();
        for (const bool layernorm : {true, false}) {
            detail::Gpu::Seq s(*g);
            std::ranges::copy(x.data, s.reserve(0, rows * d));
            s.reserve(1, rows * d);
            s.norm(0, 1, rows, w, layernorm ? &b : nullptr, layernorm, 1e-5f);
            s.run(false);
            for (int64_t r = 0; r < rows; ++r) {
                double mean = 0, var = 0;
                for (int64_t j = 0; j < d; ++j) mean += layernorm ? x.data[r * d + j] / double(d) : 0.0;
                for (int64_t j = 0; j < d; ++j)
                    var += (x.data[r * d + j] - mean) * (x.data[r * d + j] - mean) / double(d);
                for (int64_t j = 0; j < d; ++j) {
                    const double y =
                        (x.data[r * d + j] - mean) / std::sqrt(var + 1e-5) * w.data[j] + (layernorm ? b.data[j] : 0.f);
                    near(s.data(1)[r * d + j], static_cast<float>(y), 1e-5f);
                }
            }
            CHECK_THROWS(s.norm(0, 1, rows, w, &b, false, 1e-5f));  // rmsnorm has no bias
        }
    }
    {  // linear (+ bias) -> activation -> linear -> add, in fp32 and fp16. An input outside fp16's
        // range, first or in the middle (after an in-place activation), reruns the whole thing in fp32.
        const int64_t R = 300, k = 64, h = 96;
        Tensor x({R, k}), W1({h, k}), b1({h}), W2({k, h});
        for (float& v : x.data) v = nd(rng);
        for (float& v : W1.data) v = 0.3f * nd(rng);
        for (float& v : b1.data) v = 0.1f * nd(rng);
        for (float& v : W2.data) v = 0.3f * nd(rng);
        Tensor b1big = b1;
        b1big.data[3] = 1e5f;  // hidden column 3 overflows fp16, only as the second multiply's input
        Tensor xbig = x;
        xbig.data[5] = 1e5f;
        const Weight W1w(W1), W2w(W2), b1w(b1), b1bigw(b1big);
        auto bias_of = [&](const Tensor& t) { return &t == &b1big ? &b1bigw : &b1w; };
        const auto g = detail::Gpu::create();
        auto run_seq = [&](const Tensor& xin, const Tensor& bias, bool fp16) {
            detail::Gpu::Seq s(*g);
            std::ranges::copy(xin.data, s.reserve(0, R * k));
            s.reserve(1, R * h);
            s.reserve(2, R * k);
            s.reserve(3, R * k);
            s.linear(0, 1, R, W1w, bias_of(bias));
            s.activate(1, R * h, detail::GpuAct::Gelu);
            s.linear(1, 2, R, W2w, nullptr);
            s.add(2, 0, 3, R * k);
            s.run(fp16);
            return Tensor({R, k}, std::vector<float>(s.data(3), s.data(3) + R * k));
        };
        auto cpu = [&](const Tensor& xin, const Tensor& bias) {
            Tensor hid = linear(xin, W1, &bias);
            detail::activate_(hid, "gelu");
            Tensor y = linear(hid, W2, nullptr);
            for (int64_t i = 0; i < y.numel(); ++i) y.data[i] += xin.data[i];
            return y;
        };
        const Tensor want = cpu(x, b1), f32 = run_seq(x, b1, false), f16 = run_seq(x, b1, true);
        CHECK(max_abs_diff(f32, want) <= 1e-4f * std::max(1.f, max_abs(want)));
        CHECK(max_abs_diff(f16, want) <= 2e-2f * std::max(1.f, max_abs(want)));
        CHECK(max_abs_diff(f16, f32) > 0.f);  // really fp16
        for (const auto& [xi, bi] : std::vector<std::pair<const Tensor*, const Tensor*>>{{&xbig, &b1}, {&x, &b1big}}) {
            const Tensor over = run_seq(*xi, *bi, true);
            CHECK(over.data == run_seq(*xi, *bi, false).data);
            const Tensor exact = cpu(*xi, *bi);
            CHECK(max_abs_diff(over, exact) <= 1e-4f * std::max(1.f, max_abs(exact)));
        }
    }
    {  // Slots must be reserved and large enough; a failed run records nothing further.
        const Weight W(Tensor({4, 5}));
        const auto g = detail::Gpu::create();
        detail::Gpu::Seq s(*g);
        s.reserve(0, 10);
        s.linear(0, 1, 2, W, nullptr);
        CHECK_THROWS(s.run(false));  // slot 1 not reserved
        s.reserve(1, 7);
        s.linear(0, 1, 2, W, nullptr);
        CHECK_THROWS(s.run(false));  // slot 1 holds 7 < 2 x 4
        s.reserve(1, 8);
        s.run(false);  // nothing left to run
        s.linear(0, 1, 2, W, nullptr);
        s.run(false);
        for (int64_t i = 0; i < 8; ++i) CHECK(s.data(1)[i] == 0.f);
        CHECK_THROWS(s.data(9));
    }
}

// Model types the GPU blocks must handle beyond configs(): post-norm (with and without the
// parallel residual), every activation, GLU variants, qk_norm, sandwich without a norm, and layers
// of different sizes.
static std::vector<std::pair<std::string, ModelConfig>> gpu_block_configs() {
    std::vector<std::pair<std::string, ModelConfig>> v = configs();
    v.push_back({"post-norm layernorm, gelu_tanh mlp", tiny([](ModelConfig& c) {
                     c.norm_position = "post";
                     c.norm = "layernorm";
                     c.bias = true;
                     c.mlp = "mlp";
                     c.activation = "gelu_tanh";
                 })});
    v.push_back({"post-norm parallel, relu", tiny([](ModelConfig& c) {
                     c.norm_position = "post";
                     c.parallel_residual = true;
                     c.mlp = "mlp";
                     c.activation = "relu";
                 })});
    v.push_back({"post-norm, no norm, attention-only", tiny([](ModelConfig& c) {
                     c.norm_position = "post";
                     c.norm = "none";
                     c.mlp = "none";
                 })});
    v.push_back({"geglu tanh + qk_norm", tiny([](ModelConfig& c) {
                     c.mlp = "geglu";
                     c.activation = "gelu_tanh";
                     c.qk_norm = true;
                 })});
    v.push_back({"glu relu2", tiny([](ModelConfig& c) {
                     c.mlp = "glu";
                     c.activation = "relu2";
                 })});
    v.push_back({"sandwich layernorm, tanh mlp", tiny([](ModelConfig& c) {
                     c.norm_position = "sandwich";
                     c.norm = "layernorm";
                     c.bias = true;
                     c.mlp = "mlp";
                     c.activation = "tanh";
                 })});
    v.push_back({"sandwich, no norm", tiny([](ModelConfig& c) {
                     c.norm_position = "sandwich";
                     c.norm = "none";
                 })});
    v.push_back({"per-layer sizes", tiny([](ModelConfig& c) {
                     c.n_layers = 3;
                     c.layer_overrides[0] = json::parse(R"({"n_heads": 2, "d_mlp": 24})");
                     c.layer_overrides[2] = json::parse(R"({"n_heads": 8, "d_head": 2, "d_mlp": 72})");
                 })});
    return v;
}

// GPU attention on sequences longer than its blocks (32 rows) and tiles (16 keys): several per
// segment, segments starting mid-block, prefixes of earlier keys longer than a tile, windows shorter
// and longer than a tile, and head sizes the tiled kernel takes (multiples of 8 up to 128) and does
// not. Both GPU kernels = the CPU's attention, in batched passes.
static std::vector<std::pair<std::string, ModelConfig>> long_attention_configs() {
    auto cfg = [](int64_t d_model, int64_t heads, std::function<void(ModelConfig&)> edit) {
        return tiny([&](ModelConfig& c) {
            c.d_model = d_model;
            c.n_heads = heads;
            c.max_seq_len = 64;
            if (edit) edit(c);
        });
    };
    return {
        {"D=4 (per-row kernel), rope, window 5", cfg(16, 4, [](ModelConfig& c) { c.sliding_window = 5; })},
        {"D=24, rope, window 40, softcap", cfg(48, 2,
                                               [](ModelConfig& c) {
                                                   c.sliding_window = 40;
                                                   c.attn_logit_softcap = 3.0;
                                               })},
        {"D=64, alibi, GQA, window 20, softcap", cfg(128, 2,
                                                     [](ModelConfig& c) {
                                                         c.positional = "alibi";
                                                         c.n_kv_heads = 1;
                                                         c.sliding_window = 20;
                                                         c.attn_logit_softcap = 5.0;
                                                     })},
        {"D=128, partial rope, qk_norm, GQA", cfg(512, 4,
                                                  [](ModelConfig& c) {
                                                      c.rope_fraction = 0.5;
                                                      c.qk_norm = true;
                                                      c.n_kv_heads = 2;
                                                  })},
        {"D=160 (per-row kernel), learned", cfg(320, 2,
                                                [](ModelConfig& c) {
                                                    c.positional = "learned";
                                                    c.tie_embeddings = false;
                                                })},
    };
}

static void test_gpu_attention_long() {
    for (const auto& [name, c] : long_attention_configs()) {
        const Model m = random_model(c, 51);
        if (!m.gpu()) {
            std::printf("     (no Metal device: GPU path not tested)\n");
            return;
        }
        detail::set_gpu_min_rows(1);
        detail::set_gpu_min_work(1);
        const int64_t L = c.n_layers, d = c.d_model, T = 48;
        ActivationCache cache;
        m.run_with_cache(seq(T, 9), cache);
        auto lp = [](int64_t i) { return std::format("blocks.{}.", i); };
        detail::RowBatch rb;
        for (int64_t i = 0; i < L; ++i) {
            rb.prefix_k.push_back(&cache.at(lp(i) + "attn.hook_k"));
            rb.prefix_v.push_back(&cache.at(lp(i) + "attn.hook_v"));
        }
        rb.gpu = m.gpu();
        for (const auto& [s0, n] :
             std::vector<std::pair<int64_t, int64_t>>{{0, T}, {17, T - 17}, {40, 8}, {3, 21}, {33, 1}})
            rb.add(s0, n);
        Tensor x({rb.rows(), d}), want({rb.rows(), d});
        const Tensor fin = m.final_norm(cache.at(lp(L - 1) + "hook_resid_post"));
        const Tensor& pre = cache.at(lp(0) + "hook_resid_pre");
        for (size_t s = 0; s < rb.start.size(); ++s) {
            std::copy(pre.ptr() + rb.start[s] * d, pre.ptr() + (rb.start[s] + rb.len[s]) * d,
                      x.ptr() + rb.offset[s] * d);
            std::copy(fin.ptr() + rb.start[s] * d, fin.ptr() + (rb.start[s] + rb.len[s]) * d,
                      want.ptr() + rb.offset[s] * d);
        }
        const Tensor cpu = m.forward_rows_final(0, x, rb, nullptr);
        check_close(cpu, want);
        detail::set_gpu_attention(true);
        for (bool tiled : {true, false}) {
            detail::set_gpu_attention_tiled(tiled);
            check_close(m.forward_rows_final(0, x, rb, nullptr), cpu);
        }
        detail::set_gpu_attention_tiled(true);
        detail::set_gpu_attention(false);
        detail::set_gpu_min_rows(0);
        detail::set_gpu_min_work(0);
    }
}

// Single passes in GPU-only mode run on the GPU (Model::forward_from through run_rows), hooks
// included: every activation run_with_cache keeps (hook_pattern too) and the logits = the CPU's,
// with and without the resident path; hooks that edit every activation (the pattern too, so z is
// recomputed from it) have the same effect as on the CPU and are called as often. In both mode,
// short sequences stay on the CPU, bit for bit. Activations are compared relative to each one's
// largest value: nearly one-hot patterns (the large random model) magnify rounding differences.
static void test_gpu_single_pass(const Model& m, int64_t T) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens t = seq(T, 31);
    std::map<std::string, int> calls;
    Hooks edit;  // nudges every activation, differently at each element
    edit.add_filter([](const std::string& n) { return n != "hook_embed" && n != "hook_pos_embed"; },
                    [&](const std::string& n, Tensor& a) {
                        ++calls[n];
                        for (int64_t i = 0; i < a.numel(); ++i) a.data[i] *= 1.f + 0.01f * static_cast<float>(i % 5);
                    });
    const double tol = tol_for(m, 1e-4);
    auto near = [tol](const Tensor& got, const Tensor& want) {
        CHECK(got.shape == want.shape);
        if (got.shape == want.shape) CHECK(max_abs_diff(got, want) <= tol * std::max(1.f, max_abs(want)));
    };
    auto pass = [&](Device dev, bool resident, ActivationCache& cache, Tensor& edited,
                    std::map<std::string, int>& n_calls) {
        set_device(dev);
        detail::set_gpu_resident(resident);
        const Tensor logits = m.run_with_cache(t, cache);
        calls.clear();
        edited = m.forward(t, &edit);
        n_calls = calls;
        detail::set_gpu_resident(true);
        set_device(Device::Both);
        return logits;
    };
    ActivationCache cpu;
    Tensor cpu_edited;
    std::map<std::string, int> cpu_calls;
    const Tensor cpu_logits = pass(Device::Cpu, true, cpu, cpu_edited, cpu_calls);
    CHECK(cpu.contains("blocks.0.attn.hook_pattern"));
    for (bool resident : {true, false}) {
        ActivationCache gpu;
        Tensor gpu_edited;
        std::map<std::string, int> gpu_calls;
        near(pass(Device::Gpu, resident, gpu, gpu_edited, gpu_calls), cpu_logits);
        CHECK(gpu.size() == cpu.size());
        for (const auto& [name, want] : cpu) {
            CHECK(gpu.contains(name));
            if (gpu.contains(name)) {
                CHECK(gpu.at(name).shape == want.shape);
                if (gpu.at(name).shape == want.shape)
                    CHECK(max_abs_diff(gpu.at(name), want) <= tol_for(m, 1e-3) * std::max(1.f, max_abs(want)));
            }
        }
        near(gpu_edited, cpu_edited);
        CHECK(gpu_calls == cpu_calls);
        if (resident) CHECK(max_abs_diff(gpu_edited, cpu_edited) > 0.f);  // the GPU really ran
    }
    // Both mode: from gpu_min_rows rows with enough work, the GPU (its attention too).
    {
        detail::set_gpu_min_rows(1);
        detail::set_gpu_min_work(1);
        ActivationCache both;
        Tensor both_edited;
        std::map<std::string, int> both_calls;
        near(pass(Device::Both, true, both, both_edited, both_calls), cpu_logits);
        near(both_edited, cpu_edited);
        CHECK(both_calls == cpu_calls);
        CHECK(max_abs_diff(both_edited, cpu_edited) > 0.f);
        detail::set_gpu_min_rows(0);
        detail::set_gpu_min_work(0);
    }
    // Both mode, default thresholds: where the rule says the CPU, the CPU, bit for bit.
    double weights = 0;
    for (const auto& [name, w] : m.weights())
        if (w.dim() == 2 && name.starts_with("blocks.")) weights += static_cast<double>(w.numel());
    const int64_t L = m.cfg().n_layers;
    if (!detail::single_pass_on_gpu(T, L, weights / static_cast<double>(L), true, detail::gpu_min_rows(),
                                    detail::gpu_min_work())) {
        ActivationCache both;
        Tensor both_edited;
        std::map<std::string, int> both_calls;
        CHECK(same_values(pass(Device::Both, true, both, both_edited, both_calls), cpu_logits));
        CHECK(same_values(both_edited, cpu_edited));
    }
}

// Capture hooks (Hooks::add_capture, which the GPU hands over after its runs instead of stopping)
// next to hooks that edit every activation, registered before and after them: in GPU-only mode the
// captured activations and the logits = the CPU's, where a capture registered after an edit sees
// the edited activation and one registered before sees it unedited.
static void test_gpu_captures(const Model& m, int64_t T) {
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens t = seq(T, 32);
    auto all = [](const std::string& n) { return n != "hook_embed" && n != "hook_pos_embed"; };
    HookFn nudge = [](const std::string&, Tensor& a) {
        for (int64_t i = 0; i < a.numel(); ++i) a.data[i] *= 1.f + 0.01f * static_cast<float>(i % 3);
    };
    auto pass = [&](Device dev, bool capture_first, ActivationCache& got) {
        set_device(dev);
        Hooks h;
        auto store = [&got](const std::string& n, Tensor& a) { got[n] = a; };
        if (capture_first) h.add_capture(all, store);
        h.add_filter([](const std::string& n) { return n.starts_with("blocks.0."); }, nudge);  // edits in block 0 only
        if (!capture_first) h.add_capture(all, store);
        const Tensor logits = m.forward(t, &h);
        set_device(Device::Both);
        return logits;
    };
    for (bool capture_first : {true, false}) {
        ActivationCache cpu, gpu;
        const Tensor want = pass(Device::Cpu, capture_first, cpu);
        const Tensor got = pass(Device::Gpu, capture_first, gpu);
        CHECK(max_abs_diff(got, want) <= 1e-4f * std::max(1.f, max_abs(want)));
        CHECK(gpu.size() == cpu.size());
        for (const auto& [name, w] : cpu) {
            CHECK(gpu.contains(name));
            if (gpu.contains(name) && gpu.at(name).shape == w.shape)
                CHECK(max_abs_diff(gpu.at(name), w) <= 1e-3f * std::max(1.f, max_abs(w)));
            else CHECK(false);
        }
        CHECK(max_abs_diff(got, want) > 0.f);  // the GPU really ran
    }
}

// Both mode with the default thresholds, once the GPU has started: a model with 3M weights per
// layer sends a single pass of 2 rows to the GPU, and keeps one of 1 row on the CPU, bit for bit.
static void test_gpu_single_pass_defaults() {
    const Model m = random_model(tiny([](ModelConfig& c) {
                                     c.d_model = 512;
                                     c.n_heads = 8;
                                 }),
                                 55);
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    double weights = 0;
    for (const auto& [name, w] : m.weights())
        if (w.dim() == 2 && name.starts_with("blocks.0.")) weights += static_cast<double>(w.numel());
    CHECK(weights >= detail::kSinglePassLayerWeights);
    for (int64_t T : {1, 2}) {
        const Tokens t = seq(T, 33);
        set_device(Device::Cpu);
        const Tensor cpu = m.forward(t);
        set_device(Device::Both);
        const Tensor both = m.forward(t);
        if (T == 1) CHECK(same_values(both, cpu));
        else CHECK(max_abs_diff(both, cpu) > 0.f && max_abs_diff(both, cpu) <= 1e-4f * std::max(1.f, max_abs(cpu)));
    }
}

// Batched passes with the residual stream kept on the GPU (Model::run_rows): unpatched rows equal
// the plain forward pass, and every pass equals the one-linear-layer-at-a-time GPU path, with hooks
// on CPU-side activations (q, z) and on GPU-side ones (which bring that block back to the CPU).
static void test_gpu_resident(const Model& m) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    detail::set_gpu_min_rows(1);
    detail::set_gpu_min_work(1);
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model, T = 9;
    const Tokens t = seq(T, 7);
    ActivationCache cache;
    m.run_with_cache(t, cache);
    auto lp = [](int64_t i) { return std::format("blocks.{}.", i); };
    const Tensor fin = m.final_norm(cache.at(lp(L - 1) + "hook_resid_post"));
    // A whole sequence and two that start later, reading earlier keys and values from the cache.
    detail::RowBatch rb;
    for (int64_t i = 0; i < L; ++i) {
        rb.prefix_k.push_back(&cache.at(lp(i) + "attn.hook_k"));
        rb.prefix_v.push_back(&cache.at(lp(i) + "attn.hook_v"));
    }
    rb.gpu = m.gpu();
    for (const auto& [s0, n] : std::vector<std::pair<int64_t, int64_t>>{{0, T}, {4, T - 4}, {2, 5}}) rb.add(s0, n);
    auto stack = [&](const Tensor& full) {
        Tensor x({rb.rows(), d});
        for (size_t s = 0; s < rb.start.size(); ++s)
            std::copy(full.ptr() + rb.start[s] * d, full.ptr() + (rb.start[s] + rb.len[s]) * d,
                      x.ptr() + rb.offset[s] * d);
        return x;
    };
    std::map<std::string, int> calls;
    Hooks hooks;
    const std::string edited = m.cfg().has_mlp() ? "hook_mlp_out" : "hook_attn_out";
    hooks.add_filter(
        [](const std::string& n) {
            return n.find("attn.hook_q") != std::string::npos || n.find("attn.hook_z") != std::string::npos;
        },
        [&](const std::string& n, Tensor& a) {
            ++calls[n];
            a.data[0] += 1.f;
        });
    for (int64_t start = 0; start < L; ++start) {
        const Tensor x = stack(cache.at(lp(start) + "hook_resid_pre"));
        auto pass = [&](bool resident, bool fp16, const Hooks* h) {
            detail::set_gpu_resident(resident);
            rb.fp16 = fp16;
            const Tensor r = m.forward_rows_final(start, x, rb, h);
            detail::set_gpu_resident(true);
            rb.fp16 = false;
            return r;
        };
        const Tensor got = pass(true, false, nullptr);
        check_close(got, stack(fin));
        check_close(got, pass(false, false, nullptr));
        Hooks h = hooks;
        const int64_t mid = std::min(start + 1, L - 1);
        h.add(lp(mid) + edited, [&](const std::string& n, Tensor& a) {
            ++calls[n];
            for (float& v : a.data) v += 0.25f;
        });
        h.add(lp(L - 1) + "hook_resid_pre", [&](const std::string& n, Tensor& a) {
            ++calls[n];
            for (float& v : a.data) v *= 1.1f;
        });
        calls.clear();
        const Tensor a = pass(true, false, &h);
        const std::map<std::string, int> seen = calls;
        calls.clear();
        check_close(a, pass(false, false, &h));
        CHECK(seen == calls);
        CHECK(seen.contains(lp(mid) + edited) && seen.contains(lp(L - 1) + "attn.hook_z"));
        CHECK(max_abs_diff(a, got) > 0.f);
        // fp16: close, and really used.
        const Tensor h16 = pass(true, true, nullptr);
        CHECK(max_abs_diff(h16, got) <= 3e-2f * std::max(1.f, max_abs(got)));
        CHECK(max_abs_diff(h16, got) > 0.f);
    }
    detail::set_gpu_min_rows(0);
    detail::set_gpu_min_work(0);
}

// Batched gradient passes on the GPU (Model::gradients_rows) = one run_with_gradients per sequence,
// at every hook point they support, for log p(target) and a logit difference at each sequence's end.
static void test_gpu_gradients(const Model& m, const std::vector<int64_t>& lens = {9, 5, 12}) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const int64_t L = m.cfg().n_layers, V = m.cfg().vocab_size, d = m.cfg().d_model;
    std::vector<Tokens> seqs;
    for (size_t i = 0; i < lens.size(); ++i) seqs.push_back(seq(lens[i], 21 + i));
    detail::RowBatch rb;
    rb.gpu = m.gpu();
    for (const Tokens& t : seqs) rb.add(0, std::ssize(t));
    Tensor x0({rb.rows(), d});
    for (size_t g = 0; g < seqs.size(); ++g) {
        const Tensor e = m.embed(seqs[g]);
        std::copy_n(e.ptr(), e.numel(), x0.ptr() + rb.offset[g] * d);
    }
    std::vector<std::string> names = {"hook_embed"};
    for (int64_t i = 0; i < L; ++i)
        for (const char* n :
             {"hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_attn_out", "hook_mlp_out", "attn.hook_z"})
            names.push_back(std::format("blocks.{}.{}", i, n));
    for (const int64_t other : {int64_t(-1), int64_t(7)}) {
        // dF/dlogits of one row: log softmax at `target`, or logit[target] - logit[other].
        auto row_grad = [&](const float* logits, float* out) {
            if (other >= 0) {
                out[3] += 1.f;
                out[other] -= 1.f;
                return;
            }
            float mx = -INFINITY, sum = 0.f;
            for (int64_t j = 0; j < V; ++j) mx = std::max(mx, logits[j]);
            for (int64_t j = 0; j < V; ++j) sum += std::exp(logits[j] - mx);
            for (int64_t j = 0; j < V; ++j) out[j] = -std::exp(logits[j] - mx) / sum;
            out[3] += 1.f;
        };
        const std::vector<ActivationCache> got = m.gradients_rows(
            x0, rb,
            [&](int64_t, const Tensor& logits) {
                Tensor dl(logits.shape, 0.f);
                row_grad(logits.ptr(), dl.ptr());
                return dl;
            },
            nullptr);
        CHECK(got.size() == seqs.size());
        for (size_t g = 0; g < seqs.size() && g < got.size(); ++g) {
            ActivationCache want;
            const Device dev = device();
            set_device(Device::Cpu);  // the reference
            m.run_with_gradients(
                seqs[g],
                [&](const Tensor& logits) {
                    Tensor dl(logits.shape, 0.f);
                    const int64_t last = logits.shape[0] - 1;
                    row_grad(logits.ptr() + last * V, dl.ptr() + last * V);
                    return dl;
                },
                want);
            set_device(dev);
            for (const std::string& n : names) {
                CHECK(got[g].contains(n) == want.contains(n));
                if (!got[g].contains(n) || !want.contains(n)) continue;
                const Tensor &a = got[g].at(n), &b = want.at(n);
                CHECK(a.shape == b.shape);
                if (a.shape == b.shape) CHECK(max_abs_diff(a, b) <= 1e-4f * std::max(1.f, max_abs(b)));
            }
        }
    }
    // Only the filter's hook points, and every segment must start at 0.
    const auto only_embed = m.gradients_rows(
        x0, rb, [&](int64_t, const Tensor& l) { return Tensor(l.shape, 0.f); },
        [](const std::string& n) { return n == "hook_embed"; });
    CHECK(only_embed[0].size() == 1 && only_embed[0].contains("hook_embed"));
    detail::RowBatch later = rb;
    later.start[1] = 2;
    CHECK_THROWS(m.gradients_rows(x0, later, [&](int64_t, const Tensor& l) { return Tensor(l.shape, 0.f); }, nullptr));
}
#endif

static void test_nan_never_matches() {
    Tensor a({3}, {1.f, 2.f, 3.f}), b = a;
    b.data[1] = NAN;
    CHECK(std::isnan(max_abs_diff(a, b)));
    CHECK(std::isnan(max_abs_diff(b, a)));
    CHECK(std::isnan(max_abs_diff(b, b)));
    CHECK(!(max_abs_diff(a, b) <= 1e-4f));  // the golden test's comparison must fail
    Tensor inf = a;
    inf.data[0] = INFINITY;
    CHECK(max_abs_diff(inf, inf) == 0.f);
    CHECK(std::isinf(max_abs_diff(inf, a)));
}

// Layers with different head counts and head sizes: every per-head tool must use each layer's own count.
static void test_heterogeneous_heads() {
    const ModelConfig c = tiny([](ModelConfig& c) {
        c.n_layers = 3;
        c.layer_overrides[0] = json::parse(R"({"n_heads": 2})");               // 2 heads of size 8
        c.layer_overrides[2] = json::parse(R"({"n_heads": 8, "d_head": 2})");  // 8 heads of size 2
    });
    const Model m = random_model(c, 3);
    CHECK(m.max_heads() == 8);
    const Tokens t = seq(10, 2);
    Tokens corrupt = t;
    corrupt[4] = (corrupt[4] + 1) % 50;
    auto check_layout = [](const Tensor& x) {
        CHECK(x.shape == (Shape{3, 8}));
        CHECK(std::isfinite(x.at({0, 1})));
        CHECK(std::isnan(x.at({0, 2})));  // layer 0 has only 2 heads
        CHECK(std::isnan(x.at({1, 4})));  // layer 1 has 4
        CHECK(std::isfinite(x.at({2, 7})));
    };
    const HeadScores s = head_scores(m, 5, 2, 0);
    check_layout(s.induction);
    check_layout(s.previous_token);
    CHECK(s.top_two_hop.size() == 5);
    check_layout(head_ablation(m, t, Ablation::Mean, {seq(10, 5), seq(10, 6)}));
    check_layout(head_ablation(m, t, Ablation::Zero));
    check_layout(patch_heads(m, t, corrupt, 9, 1, 2).values);
    check_layout(attribution_patch_heads(m, t, corrupt, 9, 1, 2).values);
    const PatchResult mixed = mixed_patch_heads(m, t, corrupt, 9, 1, 2, 1000);
    check_layout(mixed.values);
    for (int64_t i = 0; i < mixed.effect.numel(); ++i)  // every existing head measured, no missing one
        CHECK(mixed.exact[i] == (std::isnan(mixed.estimate.data[i]) ? 0 : 1));
    check_layout(ov_copying_scores(m, 32));

    const Composition comp = composition_scores(m);
    CHECK(comp.labels.size() == 14);
    CHECK(comp.labels[13] == "L2H7");
    CHECK(std::isnan(comp.k.at({0, 1})));  // same layer
    // L0H1 -> L2H3 (indices 1 and 9) matches the direct d x d computation despite different head sizes.
    const HeadWeights w = head_weights(m, 0, 1), r = head_weights(m, 2, 3);
    const Tensor ov = ov_matrix(w), qk = qk_matrix(r);
    CHECK_NEAR(comp.k.at({1, 9}), frobenius(matmul(qk, ov)) / (frobenius(qk) * frobenius(ov)), 1e-4);
    // Two-head OV paths use the same head order and handle different head sizes.
    const Tensor pair = ov_pair_copying_scores(m, 32);
    CHECK(pair.shape == (Shape{14, 14}));
    CHECK(std::isnan(pair.at({0, 1})) && std::isnan(pair.at({9, 2})));
    CHECK(pair.at({1, 9}) == ov_path_copying(m, {{0, 1}, {2, 3}}, 32));
    CHECK(pair.at({3, 13}) == ov_path_copying(m, {{1, 1}, {2, 7}}, 32));

    const DLAResult dla = direct_logit_attribution(m, t, 9, 1, 2);
    CHECK_NEAR(dla.sum, dla.actual, 1e-3 * std::max(1.f, std::fabs(dla.actual)));
    CHECK(std::ranges::find(dla.labels, "L2H7") != dla.labels.end());
    CHECK(std::ranges::find(dla.labels, "L0H2") == dla.labels.end());
}

// The backward pass against central finite differences of the metric, at every hook point. The
// direction mixes the computed gradient with noise, so a wrong gradient can't agree by accident.
static void test_gradients(const Model& m) {
    const Tokens t = seq(7, 3);
    const int64_t pos = 5;
    for (const int64_t other : {int64_t(3), int64_t(-1)}) {  // logit difference, then log-probability
        ActivationCache grads, acts;
        const float metric = metric_gradients(m, t, pos, 7, other, grads, nullptr, &acts);
        CHECK(metric == logit_metric(m.forward(t), pos, 7, other));  // the same forward pass, bit for bit
        std::mt19937_64 rng(11);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (const std::string& n : m.hook_names()) {
            CHECK(grads.contains(n));
            if (!grads.contains(n)) continue;
            const Tensor& g = grads.at(n);
            CHECK(g.shape == acts.at(n).shape);
            Tensor r(g.shape);
            for (float& v : r.data) v = normal(rng);
            const float gn = frobenius(g), rn = frobenius(r);
            Tensor v(g.shape);
            for (int64_t i = 0; i < v.numel(); ++i) v.data[i] = (gn > 0 ? g.data[i] / gn : 0.f) + 0.5f * r.data[i] / rn;
            auto f = [&](double e) {
                Hooks h;
                h.add(n, [&](const std::string&, Tensor& a) {
                    for (int64_t i = 0; i < a.numel(); ++i) a.data[i] += static_cast<float>(e) * v.data[i];
                });
                return double(logit_metric(m.forward(t, &h), pos, 7, other));
            };
            // Central differences at h and h/2, Richardson-extrapolated: error O(h^4) instead of O(h^2).
            const double h = 1e-2, fd1 = (f(h) - f(-h)) / (2 * h), fd2 = (f(h / 2) - f(-h / 2)) / h;
            const double fd = (4 * fd2 - fd1) / 3;
            const double an = dot(g.data, v.data);
            const bool ok = std::fabs(fd - an) <= 2e-3 * std::max({1.0, std::fabs(fd), std::fabs(an)});
            CHECK(ok);
            if (!ok)
                std::printf("    %s (other %lld): backward %.6g, finite differences %.6g\n", n.c_str(),
                            static_cast<long long>(other), an, fd);
        }
    }
}

static void test_attribution_patching(const Model& m) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t L = m.cfg().n_layers;
    for (const std::string site : {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"}) {
        const PatchResult exact = patch_residual(m, clean, corrupt, site, 6, 1, 2);
        const PatchResult atp = attribution_patch_residual(m, clean, corrupt, site, 6, 1, 2);
        CHECK(atp.effect.shape == exact.effect.shape);
        CHECK(atp.clean_metric == exact.clean_metric);
        CHECK(atp.corrupt_metric == exact.corrupt_metric);
        for (int64_t i = 0; i < L; ++i)
            for (int64_t p = 0; p < 8; ++p) {
                const float e = atp.effect.at({i, p});
                CHECK(std::isnan(e) == std::isnan(exact.effect.at({i, p})));  // same missing sites
                if ((p < 3 || p == 7) && !std::isnan(e)) CHECK(e == 0.f);  // unchanged positions, and one after `pos`
            }
    }
    const PatchResult heads = attribution_patch_heads(m, clean, corrupt, 6, 1, 2);
    const PatchResult exact_heads = patch_heads(m, clean, corrupt, 6, 1, 2);
    CHECK(heads.effect.shape == exact_heads.effect.shape);
    for (int64_t i = 0; i < heads.effect.numel(); ++i)
        CHECK(std::isnan(heads.effect.data[i]) == std::isnan(exact_heads.effect.data[i]));
    const PatchResult same = attribution_patch_residual(m, clean, clean, "resid_pre", 6, 1, 2);
    CHECK(!same.normalized());
    for (float e : same.effect.data) CHECK(e == 0.f);
    CHECK_THROWS(attribution_patch_residual(m, clean, corrupt, "nope", 6, 1, 2));
    CHECK_THROWS(attribution_patch_heads(m, clean, Tokens(7, 0), 6, 1, 2));
    CHECK_THROWS(attribution_patch_residual(m, clean, corrupt, "resid_pre", 6, 1, 2, 0));

    // Integrated-gradient steps run in parallel; the result must not depend on the thread count.
    set_num_threads(1);
    const PatchResult one = attribution_patch_heads(m, clean, corrupt, 6, 1, -1, 5);
    set_num_threads(4);
    const PatchResult four = attribution_patch_heads(m, clean, corrupt, 6, 1, -1, 5);
    set_num_threads(0);
    for (int64_t i = 0; i < one.effect.numel(); ++i)
        CHECK(std::isnan(one.effect.data[i]) ? std::isnan(four.effect.data[i])
                                             : one.effect.data[i] == four.effect.data[i]);
}

// With no final norm and no MLP, the logits are linear in the last layer's outputs, so attribution
// patching there must equal exact patching.
static void test_attribution_exact_when_linear() {
    const Model m = random_model(tiny([](ModelConfig& c) {
                                     c.mlp = "none";
                                     c.norm = "none";
                                     c.n_layers = 3;
                                 }),
                                 5);
    const Tokens clean = seq(9, 2);
    Tokens corrupt = clean;
    corrupt[2] = (corrupt[2] + 11) % 50;
    corrupt[6] = (corrupt[6] + 3) % 50;
    const PatchResult exact = patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9);
    const PatchResult atp = attribution_patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9);
    for (int64_t p = 0; p < 9; ++p)
        CHECK_NEAR(atp.effect.at({2, p}), exact.effect.at({2, p}),
                   1e-4 * std::max(1.f, std::fabs(exact.effect.at({2, p}))));
    const PatchResult eh = patch_heads(m, clean, corrupt, 8, 4, 9),
                      ah = attribution_patch_heads(m, clean, corrupt, 8, 4, 9);
    for (int64_t h = 0; h < 4; ++h)
        CHECK_NEAR(ah.effect.at({2, h}), eh.effect.at({2, h}), 1e-4 * std::max(1.f, std::fabs(eh.effect.at({2, h}))));
    // There the gradient is constant, so averaging it over interpolated inputs changes nothing.
    const PatchResult ig = attribution_patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9, 4);
    for (int64_t p = 0; p < 9; ++p)
        CHECK_NEAR(ig.effect.at({2, p}), exact.effect.at({2, p}),
                   1e-4 * std::max(1.f, std::fabs(exact.effect.at({2, p}))));
    // Earlier layers feed attention patterns, which are nonlinear: there the estimate is only approximate.
    bool differs = false;
    for (int64_t p = 0; p < 9; ++p) differs |= std::fabs(atp.effect.at({0, p}) - exact.effect.at({0, p})) > 1e-3f;
    CHECK(differs);
}

// Mixed patching: exact values at the top-K estimated sites, attribution estimates elsewhere.
static void check_mixed(const PatchResult& mixed, const PatchResult& atp, const PatchResult& exact, int64_t verify,
                        int64_t usable_after_pos_cols /* residual: first column past pos, else -1 */) {
    const int64_t cols = mixed.effect.shape[1];
    CHECK(mixed.estimate.shape == atp.effect.shape);
    CHECK(mixed.exact.size() == static_cast<size_t>(mixed.effect.numel()));
    int64_t flagged = 0, usable = 0;
    float min_flagged = INFINITY, max_unflagged = 0;
    for (int64_t i = 0; i < mixed.effect.numel(); ++i) {
        const float est = atp.effect.data[i];
        CHECK(std::isnan(est) ? std::isnan(mixed.estimate.data[i]) : mixed.estimate.data[i] == est);
        const bool can = !std::isnan(est) && (usable_after_pos_cols < 0 || i % cols < usable_after_pos_cols);
        usable += can;
        if (mixed.exact[i]) {
            ++flagged;
            CHECK(can);
            // Measured: the same up to float rounding (batched passes group different sites).
            CHECK_NEAR(mixed.effect.data[i], exact.effect.data[i],
                       std::max(1e-5, g_half_tol) * std::max(1.f, std::fabs(exact.effect.data[i])));
            min_flagged = std::min(min_flagged, std::fabs(est));
        } else {
            CHECK(std::isnan(est) ? std::isnan(mixed.effect.data[i]) : mixed.effect.data[i] == est);
            if (can) max_unflagged = std::max(max_unflagged, std::fabs(est));
        }
    }
    CHECK(flagged == std::min(verify, usable));
    if (flagged > 0 && flagged < usable) CHECK(min_flagged >= max_unflagged);  // the largest estimates were measured
    if (!std::isnan(mixed.gap()) && mixed.normalized())
        for (int64_t i = 0; i < mixed.effect.numel(); ++i)
            if (!std::isnan(mixed.effect.data[i]))
                CHECK_NEAR(mixed.values.data[i], mixed.effect.data[i] / mixed.gap(), 1e-5);
}

static void test_mixed_patching(const Model& m) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t pos = 6;
    for (const std::string site : {"resid_pre", "attn_out", "mlp_out"}) {
        const PatchResult atp = attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, 2);
        const PatchResult exact = patch_residual(m, clean, corrupt, site, pos, 1, 2);
        for (const int64_t verify : {int64_t(0), int64_t(3), int64_t(1000)})
            check_mixed(mixed_patch_residual(m, clean, corrupt, site, pos, 1, 2, verify, 2), atp, exact, verify,
                        pos + 1);
        // Verifying every site reproduces exact patching.
        const PatchResult all = mixed_patch_residual(m, clean, corrupt, site, pos, 1, 2, 1000, 2);
        for (int64_t i = 0; i < all.effect.numel(); ++i)
            CHECK(std::isnan(exact.effect.data[i]) ? std::isnan(all.effect.data[i])
                                                   : all.effect.data[i] == exact.effect.data[i]);
    }
    const PatchResult atp = attribution_patch_heads(m, clean, corrupt, pos, 1, -1);
    const PatchResult exact = patch_heads(m, clean, corrupt, pos, 1, -1);
    for (const int64_t verify : {int64_t(0), int64_t(2), int64_t(1000)})
        check_mixed(mixed_patch_heads(m, clean, corrupt, pos, 1, -1, verify), atp, exact, verify, -1);
    CHECK_THROWS(mixed_patch_heads(m, clean, corrupt, pos, 1, -1, -1));
}

#ifdef TFI_USE_ACCELERATE
// Attribution and mixed patching with their gradient runs batched on the GPU: close to the CPU's
// results for any batch size, the same for any thread count, and the CPU tests' invariants hold.
static void test_gpu_attribution(const Model& m) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t pos = 6;
    const double tol = tol_for(m, 1e-4);
    auto close = [tol](const Tensor& got, const Tensor& want) {
        CHECK(got.shape == want.shape);
        float scale = 1.f;
        for (float v : want.data)
            if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
        for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
            if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
            else CHECK_NEAR(got.data[i], want.data[i], tol * scale);
    };
    bool differs = false;  // the GPU really ran
    for (const int64_t steps : {1, 3}) {
        set_use_gpu(false);
        std::vector<Tensor> cpu;
        for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"})
            cpu.push_back(attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, steps).effect);
        cpu.push_back(attribution_patch_heads(m, clean, corrupt, pos, 1, -1, steps).effect);
        set_use_gpu(true);
        detail::set_gpu_min_rows(1);
        detail::set_gpu_min_work(1);
        for (const int64_t rows : {0, 7, 1000}) {  // default, one sequence per batch, all in one
            detail::set_patch_batch_rows(0, rows);
            size_t i = 0;
            for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"}) {
                const Tensor g = attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, steps).effect;
                close(g, cpu[i]);
                differs |= max_abs_diff(g, cpu[i++]) > 0.f;
            }
            close(attribution_patch_heads(m, clean, corrupt, pos, 1, -1, steps).effect, cpu[i]);
        }
        detail::set_patch_batch_rows(0, 0);
        detail::set_gpu_min_rows(0);
        detail::set_gpu_min_work(0);
    }
    CHECK(differs);
    detail::set_gpu_min_rows(1);
    detail::set_gpu_min_work(1);
    set_num_threads(1);
    const Tensor a1 = attribution_patch_residual(m, clean, corrupt, "attn_out", pos, 1, 2, 5).effect;
    const Tensor h1 = mixed_patch_heads(m, clean, corrupt, pos, 1, -1, 3, 5).effect;
    set_num_threads(4);
    auto same = [](const Tensor& a, const Tensor& b) {  // bit for bit, NaN matching NaN (missing heads)
        CHECK(a.shape == b.shape);
        for (int64_t i = 0; i < a.numel() && a.shape == b.shape; ++i)
            CHECK(std::isnan(b.data[i]) ? std::isnan(a.data[i]) : a.data[i] == b.data[i]);
    };
    same(attribution_patch_residual(m, clean, corrupt, "attn_out", pos, 1, 2, 5).effect, a1);
    same(mixed_patch_heads(m, clean, corrupt, pos, 1, -1, 3, 5).effect, h1);
    set_num_threads(0);
    test_attribution_patching(m);
    test_mixed_patching(m);
    detail::set_gpu_min_rows(0);
    detail::set_gpu_min_work(0);
}
#endif

// fp16 weights on the GPU: batched passes (resident and one linear layer at a time, fp32 and fp16
// inputs) and batched gradients are bit for bit those of the float32 model with the rounded weights.
#ifdef TFI_USE_ACCELERATE
static void test_gpu_fp16_matches_rounded(const Model& m16) {
    const Model r = rounded_copy(m16);
    if (!m16.gpu() || !r.gpu()) return;
    detail::set_gpu_min_rows(1);
    detail::set_gpu_min_work(1);
    const int64_t L = m16.cfg().n_layers, T = 9;
    const Tokens t = seq(T, 7);
    ActivationCache cache;
    m16.run_with_cache(t, cache);
    auto lp = [](int64_t i) { return std::format("blocks.{}.", i); };
    detail::RowBatch b16, br;
    for (int64_t i = 0; i < L; ++i) {
        b16.prefix_k.push_back(&cache.at(lp(i) + "attn.hook_k"));
        b16.prefix_v.push_back(&cache.at(lp(i) + "attn.hook_v"));
    }
    for (const auto& [s0, n] : std::vector<std::pair<int64_t, int64_t>>{{0, T}, {4, T - 4}}) b16.add(s0, n);
    br = b16;
    b16.gpu = m16.gpu();
    br.gpu = r.gpu();
    const int64_t d = m16.cfg().d_model;
    for (int64_t start = 0; start < L; ++start) {
        const Tensor& full = cache.at(lp(start) + "hook_resid_pre");
        Tensor x({b16.rows(), d});
        for (size_t s = 0; s < b16.start.size(); ++s)
            std::copy(full.ptr() + b16.start[s] * d, full.ptr() + (b16.start[s] + b16.len[s]) * d,
                      x.ptr() + b16.offset[s] * d);
        for (const bool resident : {true, false})
            for (const bool fp16 : {false, true}) {
                detail::set_gpu_resident(resident);
                b16.fp16 = br.fp16 = fp16;
                CHECK(m16.forward_rows(start, x, b16).data == r.forward_rows(start, x, br).data);
            }
    }
    detail::set_gpu_resident(true);
    b16.fp16 = br.fp16 = false;
    detail::RowBatch g16, gr;  // whole sequences, as attribution patching stacks them
    g16.add(0, T);
    g16.add(0, T);
    gr = g16;
    g16.gpu = m16.gpu();
    gr.gpu = r.gpu();
    Tensor x0({2 * T, d});
    const Tensor e = m16.embed(t);
    std::copy_n(e.ptr(), T * d, x0.ptr());
    for (int64_t i = 0; i < T * d; ++i) x0.data[T * d + i] = 0.5f * e.data[i];
    auto dmetric = [](int64_t s, const Tensor& logits) {
        Tensor g(logits.shape);
        g.data[1] = 1.f;
        g.data[2] = s ? -2.f : -1.f;
        return g;
    };
    const std::vector<ActivationCache> a = m16.gradients_rows(x0, g16, dmetric, nullptr);
    const std::vector<ActivationCache> b = r.gradients_rows(x0, gr, dmetric, nullptr);
    CHECK(a.size() == b.size());
    for (size_t s = 0; s < a.size() && s < b.size(); ++s) {
        CHECK(a[s].size() == b[s].size() && !a[s].empty());
        for (const auto& [name, t] : b[s]) CHECK(a[s].contains(name) && a[s].at(name).data == t.data);
    }
    detail::set_gpu_min_rows(0);
    detail::set_gpu_min_work(0);
}
#endif

#ifdef TFI_USE_ACCELERATE
// The GPU is used only when the batches big enough for it add up to enough work to repay starting it:
// a tiny model never starts it, whatever the batch sizes, unless the threshold is lowered.
static void test_gpu_work_threshold() {
    const ModelConfig cfg = tiny();
    const Tokens clean = seq(12, 1), corrupt = [&] {
        Tokens t = clean;
        t[5] = (t[5] + 1) % 50;
        return t;
    }();
    detail::set_gpu_min_rows(1);  // every batch is GPU-sized
    CHECK(detail::gpu_min_work() == 6e10);
    {
        const Model m = random_model(cfg, 2);
        const Tensor r = patch_residual(m, clean, corrupt, "resid_pre", 11, 1, 2).effect;
        const Tensor h = attribution_patch_residual(m, clean, corrupt, "resid_pre", 11, 1, 2, 8).effect;
        const Tensor a = head_ablation(m, clean, Ablation::Zero, {});
        CHECK(!m.has_gpu());  // far below the threshold: everything ran on the CPU
        detail::set_gpu_min_work(1);
        const Model g = random_model(cfg, 2);
        check_close(patch_residual(g, clean, corrupt, "resid_pre", 11, 1, 2).effect, r);
        CHECK(g.has_gpu() || !detail::Gpu::create());  // now it starts (when there is a Metal device)
        check_close(attribution_patch_residual(g, clean, corrupt, "resid_pre", 11, 1, 2, 8).effect, h);
        check_close(head_ablation(g, clean, Ablation::Zero, {}), a);
    }
    detail::set_gpu_min_work(0);
    CHECK(detail::gpu_min_work() == 6e10);
    detail::set_gpu_min_rows(0);
}
#endif

#ifdef TFI_USE_ACCELERATE
// Gpu::Batch's operations against the CPU: multiplies on sub-matrices (offsets, strides, transposed
// or not, fp16 weights in place), block sums of squares, row argmaxes (the first on ties), and
// layouts that don't fit their buffers refused.
static void test_gpu_batch() {
    const auto g = detail::Gpu::create();
    if (!g) return;
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd;
    using Mat = detail::Gpu::Batch::Mat;
    Tensor A({40, 70}), B({90, 70}), Bk({70, 33});
    for (Tensor* t : {&A, &B, &Bk})
        for (float& v : t->data) v = nd(rng);
    const Weight B16 = Weight::fp16(B), B32(B);
    detail::Gpu::Batch batch(*g);
    const int a = batch.buffer(A.numel()), bk = batch.buffer(Bk.numel());
    std::ranges::copy(A.data, batch.floats(a));
    std::ranges::copy(Bk.data, batch.floats(bk));
    const int w16 = batch.weight(B16), w32 = batch.weight(B32);
    const int c1 = batch.buffer(30 * 60), c2 = batch.buffer(30 * 60), c3 = batch.buffer(40 * 33);
    // C (30, 50) at column 5 of a 60-wide buffer = A[5:35, 10:70] B[20:70, 10:70]^T.
    batch.matmul(Mat{a, 5 * 70 + 10, 70}, Mat{w16, 20 * 70 + 10, 70}, Mat{c1, 5, 60}, 30, 50, 60, true);
    batch.matmul(Mat{a, 5 * 70 + 10, 70}, Mat{w32, 20 * 70 + 10, 70}, Mat{c2, 5, 60}, 30, 50, 60, true);
    batch.end_group();
    batch.matmul(Mat{a, 0, 70}, Mat{bk, 0, 33}, Mat{c3, 0, 33}, 40, 33, 70, false);
    const int sums = batch.buffer(6 * 5), top = batch.buffer(40);
    batch.block_sumsq(Mat{c3, 0, 33}, 36, 30, 6, 6, sums, 0);
    batch.row_argmax(Mat{c3, 0, 33}, 40, 33, top, 0);
    batch.run();
    const Tensor Bh = B16.to_tensor();
    double e16 = 0, e32 = 0;
    for (int64_t i = 0; i < 30; ++i)
        for (int64_t j = 0; j < 50; ++j) {
            double w16v = 0, w32v = 0;
            for (int64_t k = 0; k < 60; ++k) {
                w16v += double(A.data[(5 + i) * 70 + 10 + k]) * Bh.data[(20 + j) * 70 + 10 + k];
                w32v += double(A.data[(5 + i) * 70 + 10 + k]) * B.data[(20 + j) * 70 + 10 + k];
            }
            e16 = std::max(e16, std::fabs(batch.floats(c1)[i * 60 + 5 + j] - w16v));
            e32 = std::max(e32, std::fabs(batch.floats(c2)[i * 60 + 5 + j] - w32v));
        }
    CHECK(e16 < 1e-4 && e32 < 1e-4);  // the fp16 weight's own values, multiplied in fp32
    const Tensor C = matmul(A, Bk);
    CHECK(max_abs_diff(Tensor({40, 33}, std::vector<float>(batch.floats(c3), batch.floats(c3) + 40 * 33)), C) < 1e-4f);
    bool sums_ok = true, top_ok = true;
    for (int64_t bi = 0; bi < 6; ++bi)
        for (int64_t bj = 0; bj < 5; ++bj) {
            double want = 0;
            for (int64_t r = 0; r < 6; ++r)
                for (int64_t q = 0; q < 6; ++q)
                    want += double(batch.floats(c3)[(bi * 6 + r) * 33 + bj * 6 + q]) *
                            batch.floats(c3)[(bi * 6 + r) * 33 + bj * 6 + q];
            sums_ok &= std::fabs(batch.floats(sums)[bi * 5 + bj] - want) <= 1e-4 * std::max(1.0, want);
        }
    for (int64_t r = 0; r < 40; ++r)
        top_ok &= int64_t(batch.uints(top)[r]) == topk(std::span(batch.floats(c3) + r * 33, 33), 1)[0];
    CHECK(sums_ok && top_ok);
    // Ties: the first index wins.
    const int tie = batch.buffer(2 * 600), tie_top = batch.buffer(2);
    float* tv = batch.floats(tie);
    for (int64_t i = 0; i < 1200; ++i) tv[i] = -1.f;
    tv[417] = tv[523] = tv[599] = 2.f;
    tv[600 + 3] = tv[600 + 300] = 0.f;
    batch.row_argmax(Mat{tie, 0, 600}, 2, 600, tie_top, 0);
    batch.run();
    CHECK(batch.uints(tie_top)[0] == 417 && batch.uints(tie_top)[1] == 3);
    CHECK_THROWS(batch.matmul(Mat{a, 0, 70}, Mat{bk, 0, 33}, Mat{c3, 0, 33}, 41, 33, 70, false));  // past A's end
    CHECK_THROWS(batch.block_sumsq(Mat{c3, 0, 33}, 36, 30, 7, 6, sums, 0));                        // blocks don't tile
    CHECK_THROWS(
        batch.matmul(Mat{a, 0, 70}, Mat{bk, 0, 33}, Mat{w32, 0, 33}, 40, 33, 70, false));  // a weight as output
    batch.release(c1);
    CHECK_THROWS(batch.floats(c1));
}

// circuits on the GPU = on the CPU: composition scores to float rounding (a different but equal
// formula), OV copying and two-head paths exactly but for near-ties. Splitting the work into
// smaller command buffers or runs changes nothing; a command buffer that fails is run again, with
// the same results; one that keeps failing leaves the work to the CPU.
static void test_gpu_circuits(const Model& m) {
    if (!m.gpu()) return;
    const int64_t n = 64;
    set_use_gpu(false);
    const Composition cc = composition_scores(m);
    const Tensor oc = ov_copying_scores(m, n), pc = ov_pair_copying_scores(m, n);
    set_use_gpu(true);
    detail::set_gpu_min_work(1);
    auto close = [](const Tensor& a, const Tensor& b, double tol) {
        if (a.shape != b.shape) return false;
        for (int64_t i = 0; i < a.numel(); ++i)
            if (std::isnan(a.data[i]) ? !std::isnan(b.data[i]) : !(std::fabs(a.data[i] - b.data[i]) <= tol))
                return false;
        return true;
    };
    auto same = [&](const Tensor& a, const Tensor& b) { return close(a, b, 0.0); };
    const Composition cg = composition_scores(m);
    const Tensor og = ov_copying_scores(m, n), pg = ov_pair_copying_scores(m, n);
    CHECK(m.has_gpu());
    CHECK(close(cg.q, cc.q, 1e-5) && close(cg.k, cc.k, 1e-5) && close(cg.v, cc.v, 1e-5));
    CHECK(close(og, oc, 1.5 / n) && close(pg, pc, 1.5 / n));
    // One group per command buffer, one layer per run: the same.
    detail::set_gpu_chunk_flops(1);
    detail::set_circuits_run_bytes(1);
    const Composition cs = composition_scores(m);
    CHECK(same(cs.q, cg.q) && same(cs.k, cg.k) && same(cs.v, cg.v));
    CHECK(same(ov_copying_scores(m, n), og) && same(ov_pair_copying_scores(m, n), pg));
    // A failed command buffer runs again: the same results.
    detail::set_gpu_fail_chunks(2);
    const Composition cr = composition_scores(m);
    CHECK(same(cr.q, cg.q) && same(cr.k, cg.k) && same(cr.v, cg.v));
    detail::set_gpu_fail_chunks(2);
    CHECK(same(ov_copying_scores(m, n), og));
    detail::set_gpu_fail_chunks(2);
    CHECK(same(ov_pair_copying_scores(m, n), pg));
    // Failing every time: the CPU's results.
    detail::set_gpu_fail_chunks(1 << 30);
    const Composition cf = composition_scores(m);
    CHECK(same(cf.q, cc.q) && same(cf.k, cc.k) && same(cf.v, cc.v));
    CHECK(same(ov_copying_scores(m, n), oc) && same(ov_pair_copying_scores(m, n), pc));
    detail::set_gpu_fail_chunks(0);
    detail::set_gpu_chunk_flops(0);
    detail::set_circuits_run_bytes(0);

    // A model with fp16 weights: its circuits on the GPU take fp16 inputs, close to the CPU's (which computes them in
    // float32 from the same fp16 weights), and really fp16 (composition differs from float32 in its last bits).
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    const Model m16(m.cfg(), w, WeightFormat::fp16);
    set_use_gpu(false);
    const Composition c16 = composition_scores(m16);
    const Tensor o16 = ov_copying_scores(m16, n), p16 = ov_pair_copying_scores(m16, n);
    set_use_gpu(true);
    const Composition ch = composition_scores(m16);
    double comp_scale = 0, diff32 = 0;
    for (const float v : c16.q.data)
        if (!std::isnan(v)) comp_scale = std::max(comp_scale, double(std::fabs(v)));
    for (int64_t i = 0; i < ch.q.numel(); ++i)
        if (!std::isnan(c16.q.data[i])) diff32 = std::max(diff32, double(std::fabs(ch.q.data[i] - c16.q.data[i])));
    const double tol = 5e-3 * std::max(1.0, comp_scale);
    CHECK(close(ch.q, c16.q, tol) && close(ch.k, c16.k, tol) && close(ch.v, c16.v, tol));
    CHECK(diff32 > 0);
    CHECK(close(ov_copying_scores(m16, n), o16, 4.0 / n) && close(ov_pair_copying_scores(m16, n), p16, 4.0 / n));
    // Values too large for fp16: the work runs again in float32, as close to the CPU as float32 runs are. First an
    // input that doesn't fit (W_O x 1e6, which the model keeps float32), then products that don't while their inputs
    // do (embeddings and W_O x 1000).
    for (const float scale : {1e6f, 1e3f}) {
        std::map<std::string, Tensor> wb = w;
        for (auto& [name, w] : wb)
            if (name.find("o_proj.weight") != std::string::npos || (scale < 1e6f && name == "embed.weight"))
                for (float& v : w.data) v *= scale;
        const Model big(m.cfg(), wb, WeightFormat::fp16);
        set_use_gpu(false);
        const Composition bc = composition_scores(big);
        const Tensor oc_big = ov_copying_scores(big, n), pc_big = ov_pair_copying_scores(big, n);
        set_use_gpu(true);
        const Composition bg = composition_scores(big);
        if (scale == 1e6f) CHECK(close(bg.q, bc.q, 1e-5) && close(bg.k, bc.k, 1e-5) && close(bg.v, bc.v, 1e-5));
        CHECK(close(ov_copying_scores(big, n), oc_big, 1.5 / n) &&
              close(ov_pair_copying_scores(big, n), pc_big, 1.5 / n));
    }
    detail::set_gpu_min_work(0);
}
#endif

// Which single passes go to the GPU (detail::single_pass_on_gpu), with the default thresholds and
// GPU-only mode's.
static void test_single_pass_rule() {
    using detail::single_pass_on_gpu;
    const int64_t rows = 256;
    const double work = 6e10, l = 12.6e6, xxs = 0.8e6;         // weights per layer of 24x1024 and 4x256 models
    CHECK(!single_pass_on_gpu(1, 24, l, true, rows, work));    // a matrix-vector pass: the CPU
    CHECK(single_pass_on_gpu(2, 24, l, true, rows, work));     // a big model, the GPU running
    CHECK(!single_pass_on_gpu(2, 24, l, false, rows, work));   // ... but not to start it
    CHECK(!single_pass_on_gpu(16, 4, xxs, true, rows, work));  // a small model, few rows: the CPU
    CHECK(single_pass_on_gpu(64, 4, xxs, true, rows, work));   // ... enough rows x weights
    CHECK(single_pass_on_gpu(300, 2, 1e3, true, rows, work));  // the batched passes' rule
    CHECK(!single_pass_on_gpu(300, 2, 1e3, false, rows, work));
    CHECK(single_pass_on_gpu(5000, 24, l, false, rows, work));  // enough work to start the GPU
    CHECK(!single_pass_on_gpu(4, 0, l, true, rows, work));      // no blocks to run
    CHECK(single_pass_on_gpu(1, 2, 1e3, false, 1, 0));          // GPU-only: everything
}

#ifdef TFI_USE_ACCELERATE
// One control for where computations run: Cpu, Both (sizes decide), Gpu (no thresholds, no CPU
// fallback). set_use_gpu is the earlier interface to the same setting.
static void test_device_modes() {
    set_device(Device::Cpu);
    CHECK(device() == Device::Cpu && !use_gpu() && !detail::gpu_only());
    set_use_gpu(true);
    CHECK(device() == Device::Both && use_gpu());
    set_use_gpu(false);
    CHECK(device() == Device::Cpu);
    set_device(Device::Both);
    if (!detail::Gpu::create()) {
        CHECK_THROWS(set_device(Device::Gpu));
        return;
    }
    // A tiny model: Both keeps it on the CPU (too little work), Gpu runs it on the GPU anyway.
    const ModelConfig cfg = tiny();
    const Tokens clean = seq(12, 1), corrupt = [&] {
        Tokens t = clean;
        t[5] = (t[5] + 1) % 50;
        return t;
    }();
    const Model both = random_model(cfg, 2);
    const Tensor rb = patch_residual(both, clean, corrupt, "resid_pre", 11, 1, 2).effect;
    const Tensor ab = attribution_patch_residual(both, clean, corrupt, "resid_pre", 11, 1, 2, 8).effect;
    const Tensor zb = head_ablation(both, clean, Ablation::Zero, {});
    const Composition cb = composition_scores(both);
    CHECK(!both.has_gpu());
    set_device(Device::Gpu);
    CHECK(device() == Device::Gpu && use_gpu() && detail::gpu_only());
    CHECK(detail::gpu_min_rows() == 1 && detail::gpu_min_work() == 0);
    const Model g = random_model(cfg, 2);
    check_close(patch_residual(g, clean, corrupt, "resid_pre", 11, 1, 2).effect, rb);
    CHECK(g.has_gpu());
    check_close(attribution_patch_residual(g, clean, corrupt, "resid_pre", 11, 1, 2, 8).effect, ab);
    check_close(head_ablation(g, clean, Ablation::Zero, {}), zb);
    const Composition cg = composition_scores(g);
    for (int64_t i = 0; i < cb.q.numel(); ++i)
        CHECK(std::isnan(cb.q.data[i]) ? std::isnan(cg.q.data[i]) : std::fabs(cg.q.data[i] - cb.q.data[i]) <= 1e-5f);
    // GPU work that keeps failing: Gpu reports it, Both quietly uses the CPU.
    detail::set_gpu_fail_chunks(1 << 30);
    bool threw = false;
    try {
        composition_scores(g);
    } catch (const detail::GpuFailure&) { threw = true; }
    CHECK(threw);
    set_device(Device::Both);
    detail::set_gpu_min_work(1);
    const Composition fb = composition_scores(g);  // falls back to the CPU: its exact results
    bool same = true;
    for (int64_t i = 0; i < cb.q.numel(); ++i)
        same &= std::isnan(cb.q.data[i]) ? std::isnan(fb.q.data[i]) : fb.q.data[i] == cb.q.data[i];
    CHECK(same);
    detail::set_gpu_fail_chunks(0);
    detail::set_gpu_min_work(0);
    CHECK(detail::gpu_min_rows() == 256 && detail::gpu_min_work() == 6e10);
}
#endif

#ifdef TFI_USE_ACCELERATE
// Memory-mapped weights on the GPU: a model loaded in place from an aligned file (float32 and fp16)
// gives what the CPU gives on every GPU path, the GPU using the mapping as it uses its own memory.
static void test_gpu_mapped_weights() {
    const std::string path = std::format("/tmp/tfi_unit_mapped_{}.safetensors", ::getpid());
    const ModelConfig cfg = tiny([](ModelConfig& c) {
        c.bias = true;
        c.norm = "layernorm";
    });
    const std::string cfg_json = R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                                 R"("mlp_multiple_of": 8, "bias": true, "norm": "layernorm"})";
    std::map<std::string, Tensor> w;
    const Model source = random_model(cfg, 11);
    for (const auto& [name, weight] : source.weights()) w[name] = weight.to_tensor();
    for (const char* type : {"F32", "F16"}) {
        write_safetensors(path, cfg_json, w, [&](const std::string&) { return std::string(type); }, true);
        const Model m = Model::load(path, std::string(type) == "F16" ? WeightFormat::fp16 : WeightFormat::fp32);
        std::remove(path.c_str());
        CHECK(m.weight("blocks.0.mlp.up.weight").is_borrowed() && m.weight("embed.weight").is_borrowed());
        test_gpu_resident(m);
        test_gpu_gradients(m);
        test_gpu_single_pass(m, 9);
    }
}

// Heads wider than the GPU's attention kernels take (Gpu::kMaxHeadDim): in both mode a single pass the
// scheduler sends to the GPU keeps attention on the CPU and gives the CPU's result, hook_pattern
// included; GPU-only mode refuses with the reason. Heads of exactly kMaxHeadDim run on the GPU.
static void test_gpu_wide_heads() {
    const Model probe = random_model(tiny(), 1);
    if (!probe.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens t = seq(9, 61);
    for (int64_t D : {detail::Gpu::kMaxHeadDim + 8, detail::Gpu::kMaxHeadDim}) {
        const Model m = random_model(tiny([D](ModelConfig& c) {
                                         c.n_heads = 2;
                                         c.d_head = D;
                                     }),
                                     62);
        set_device(Device::Cpu);
        ActivationCache cpu;
        const Tensor want = m.run_with_cache(t, cpu);
        set_device(Device::Both);
        detail::set_gpu_min_rows(1);  // the scheduler sends this single pass to the GPU
        detail::set_gpu_min_work(1);
        ActivationCache both;
        Tensor got;
        bool threw = false;
        try {
            got = m.run_with_cache(t, both);
        } catch (const std::exception& e) {
            threw = true;
            std::printf("     both mode, head size %lld: %s\n", static_cast<long long>(D), e.what());
        }
        detail::set_gpu_min_rows(0);
        detail::set_gpu_min_work(0);
        CHECK(!threw);
        if (!threw) {
            CHECK(max_abs_diff(got, want) <= 1e-4f * std::max(1.f, max_abs(want)));
            CHECK(both.contains("blocks.0.attn.hook_pattern"));
            if (both.contains("blocks.0.attn.hook_pattern"))
                CHECK(max_abs_diff(both.at("blocks.0.attn.hook_pattern"), cpu.at("blocks.0.attn.hook_pattern")) <=
                      1e-4f);
        }
        set_device(Device::Gpu);
        std::string error;
        try {
            got = m.forward(t);
        } catch (const std::exception& e) { error = e.what(); }
        set_device(Device::Both);
        if (D > detail::Gpu::kMaxHeadDim) {
            CHECK(error.find("head size") != std::string::npos);
        } else {
            CHECK(error.empty());
            if (error.empty()) CHECK(max_abs_diff(got, want) <= 1e-4f * std::max(1.f, max_abs(want)));
        }
    }
}
#endif

// Test groups, for running part of the suite:
//   core      json, tensors, matmul, vector math, weight shapes
//   analysis  per model type: cache and hooks, lens, DLA, patching, ablation, circuits
//   grad      gradients against finite differences, attribution and mixed patching
//   fp16      fp16 conversion and weights, and the analysis invariants with fp16 weights
//   gpu       GPU work: chunking and thresholds everywhere; kernels, resident passes, attention on macOS
// Linear layers on 1-4 rows multiply straight from the weights (src/model/weight.cpp): = a double-precision
// reference, for float32 and fp16 weights and sizes with a tail past the 8-wide steps; fp16 weights =
// float32 weights holding the same rounded values, bit for bit (on 1-4 rows); the same with one thread
// or many.
static void test_linear_few_rows() {
    std::mt19937_64 rng(5);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (const auto& [in, out] : std::vector<std::pair<int64_t, int64_t>>{{16, 50}, {13, 7}, {2048, 300}, {5, 70000}}) {
        Tensor W({out, in});
        for (float& v : W.data) v = normal(rng);
        Tensor bias({out});
        for (float& v : bias.data) v = normal(rng);
        const Weight w32(W), w16 = Weight::fp16(W);
        const Weight rounded(w16.to_tensor());  // float32 weights holding fp16's values
        const Weight b(bias);
        for (int64_t rows : {1, 2, 3, 4, 5, 8, 9, 16, 17}) {
            Tensor x({rows, in});
            for (float& v : x.data) v = normal(rng);
            for (const Weight* wp : {&w32, &rounded}) {
                const Tensor y = linear(x, *wp, &b);
                const Tensor dense = wp->to_tensor();
                float worst = 0;
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t j = 0; j < out; ++j) {
                        double want = bias.data[j];
                        for (int64_t k = 0; k < in; ++k) want += double(x.data[r * in + k]) * dense.data[j * in + k];
                        worst = std::max(worst, float(std::fabs(y.data[r * out + j] - want) / (1.0 + std::fabs(want))));
                    }
                CHECK(worst < 1e-5f * std::sqrt(float(in)));
            }
            if (rows <= 16) {
                CHECK(linear(x, w16, &b).data == linear(x, rounded, &b).data);  // 17: the library's blocks
                // A row's output doesn't depend on the other rows (groups of 4 each run as 1-4 rows would).
                for (const Weight* wp : {&w32, &w16}) {
                    const Tensor all = linear(x, *wp, &b);
                    for (int64_t r = 0; r < rows; ++r)
                        CHECK(linear(tfi::rows(x, r, r + 1), *wp, &b).data ==
                              std::vector<float>(all.ptr() + r * out, all.ptr() + (r + 1) * out));
                }
            }
            set_num_threads(1);
            const Tensor one = linear(x, w16, &b);
            set_num_threads(0);
            CHECK(one.data == linear(x, w16, &b).data);
        }
    }
}

// Rows of logits from several calls, stacked.
static Tensor stack_rows(const std::vector<Tensor>& parts) {
    std::vector<float> data;
    int64_t n = 0, V = 0;
    for (const Tensor& p : parts) {
        data.insert(data.end(), p.data.begin(), p.data.end());
        n += p.shape[0];
        V = p.shape[1];
    }
    return Tensor({n, V}, std::move(data));
}

// Positions [first, first + n) of a whole pass's hook_pattern (H, T, T), over keys 0..first+n-1.
static Tensor pattern_rows(const Tensor& P, int64_t first, int64_t n) {
    const int64_t H = P.shape[0], T = P.shape[1], nk = first + n;
    Tensor out({H, n, nk});
    for (int64_t h = 0; h < H; ++h)
        for (int64_t r = 0; r < n; ++r)
            for (int64_t j = 0; j < nk; ++j) out.data[(h * n + r) * nk + j] = P.data[(h * T + first + r) * T + j];
    return out;
}

// forward_cached, in chunks of every size, gives forward()'s rows; the cache holds hook_k / hook_v; a
// copy or a truncated cache continues on its own; hooks see the new rows and their edits carry over.
static void test_forward_cached(const Model& m) {
    const int64_t T = 12, L = m.cfg().n_layers;
    const Tokens t = seq(T, 61);
    ActivationCache acts;
    const Tensor full = m.run_with_cache(t, acts);
    for (const std::vector<int64_t>& chunks : std::vector<std::vector<int64_t>>{
             {12}, {5, 1, 1, 1, 1, 1, 1, 1}, {3, 4, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}) {
        KVCache cache;
        std::vector<Tensor> parts;
        int64_t at = 0;
        for (int64_t n : chunks) {
            const Tokens part(t.begin() + at, t.begin() + at + n);
            parts.push_back(m.forward_cached(part, cache));
            at += n;
            CHECK(cache.size() == at);
        }
        check_close(stack_rows(parts), full);
        for (int64_t l = 0; l < L; ++l) {
            check_close(cache.keys(l), acts.at(std::format("blocks.{}.attn.hook_k", l)));
            check_close(cache.values(l), acts.at(std::format("blocks.{}.attn.hook_v", l)));
        }
    }
    KVCache cache;
    check_close(m.forward_cached(Tokens(t.begin(), t.begin() + 7), cache, nullptr, true), rows(full, 6, 7));
    // A copy continues on its own, with other tokens; the original is untouched.
    KVCache branch = cache;
    Tokens other(t.begin(), t.begin() + 7);
    other.push_back((t[7] + 3) % m.cfg().vocab_size);
    check_close(m.forward_cached({other[7]}, branch), rows(m.forward(other), 7, 8));
    CHECK(branch.size() == 8 && cache.size() == 7);
    check_close(m.forward_cached(Tokens(t.begin() + 7, t.end()), cache), rows(full, 7, T));
    // Truncated back to 4 positions, then continued.
    cache.truncate(4);
    CHECK(cache.size() == 4);
    check_close(m.forward_cached(Tokens(t.begin() + 4, t.end()), cache), rows(full, 4, T));
    CHECK_THROWS(cache.truncate(13));
    CHECK_THROWS(cache.truncate(-1));
    CHECK_THROWS(m.forward_cached({}, cache));
    cache.truncate(0);
    CHECK(cache.empty());
    CHECK_THROWS(cache.keys(0));

    // Hooks on a cached step see its rows: the pattern of the new queries over every key.
    {
        KVCache c;
        m.forward_cached(Tokens(t.begin(), t.begin() + 9), c);
        ActivationCache seen;
        Hooks capture;
        capture.add_filter([](const std::string&) { return true; },
                           [&](const std::string& n, Tensor& a) { seen[n] = a; });
        m.forward_cached(Tokens(t.begin() + 9, t.end()), c, &capture);
        for (int64_t l = 0; l < L; ++l) {
            const std::string p = std::format("blocks.{}.", l);
            check_close(seen.at(p + "attn.hook_pattern"), pattern_rows(acts.at(p + "attn.hook_pattern"), 9, 3));
            check_close(seen.at(p + "hook_resid_post"), rows(acts.at(p + "hook_resid_post"), 9, T));
        }
        check_close(seen.at("hook_final"), rows(acts.at("hook_final"), 9, T));
        if (m.cfg().embedding_positional())
            check_close(seen.at("hook_pos_embed"), rows(acts.at("hook_pos_embed"), 9, T));
    }
    // rerun_cached: a cached position computed again from a block, from its input there, gives the same row,
    // keys and values, and the positions after it keep theirs; with hooks from that block on, the row, keys
    // and values are those of a step of that token with the hooks.
    {
        auto first_positions = [](const Tensor& k, int64_t n) {
            const int64_t H = k.shape[0], P = k.shape[1], D = k.shape[2];
            Tensor out({H, n, D});
            for (int64_t h = 0; h < H; ++h)
                std::copy(k.ptr() + h * P * D, k.ptr() + (h * P + n) * D, out.ptr() + h * n * D);
            return out;
        };
        KVCache c;
        m.forward_cached(t, c);
        const std::string top = std::format("blocks.{}.", L - 1);
        Hooks edit;
        edit.add(top + "attn.hook_k", [](const std::string&, Tensor& k) {
            for (float& x : k.data) x *= 1.5f;
        });
        edit.add(top + "hook_resid_post", [](const std::string&, Tensor& a) {
            for (float& x : a.data) x *= 0.5f;
        });
        for (int64_t b : {int64_t(0), L - 1})
            for (int64_t p : {int64_t(0), int64_t(4), T - 1}) {
                const Tensor in = rows(acts.at(std::format("blocks.{}.hook_resid_pre", b)), p, p + 1);
                KVCache r = c;
                check_close(m.rerun_cached(b, p, in, r), rows(full, p, p + 1));
                CHECK(r.size() == T);
                for (int64_t l = 0; l < L; ++l) {
                    check_close(r.keys(l), acts.at(std::format("blocks.{}.attn.hook_k", l)));
                    check_close(r.values(l), acts.at(std::format("blocks.{}.attn.hook_v", l)));
                }
                KVCache step = c;
                step.truncate(p);
                const Tensor want = m.forward_cached({t[static_cast<size_t>(p)]}, step, &edit);
                KVCache got = c;
                check_close(m.rerun_cached(b, p, in, got, &edit), want);
                for (int64_t l = 0; l < L; ++l) {
                    check_close(first_positions(got.keys(l), p + 1), step.keys(l));
                    check_close(first_positions(got.values(l), p + 1), step.values(l));
                }
                if (p + 3 > T)
                    continue;  // continued by a step of several tokens (the GPU, in that mode): it reads the new values
                got.truncate(p + 1);
                const Tokens rest(t.begin() + p + 1, t.end());
                check_close(m.forward_cached(rest, got), m.forward_cached(rest, step));
            }
        const Tensor in0 = rows(acts.at("blocks.0.hook_resid_pre"), 0, 1);
        KVCache r = c;
        CHECK_THROWS(m.rerun_cached(L, 0, in0, r));
        CHECK_THROWS(m.rerun_cached(-1, 0, in0, r));
        CHECK_THROWS(m.rerun_cached(0, T, in0, r));
        CHECK_THROWS(m.rerun_cached(0, 0, rows(acts.at("blocks.0.hook_resid_pre"), 0, 2), r));
        KVCache none;
        CHECK_THROWS(m.rerun_cached(0, 0, in0, none));
        // rerun_final: a position's logits from hook_final's output there, the cache left as it was.
        for (int64_t p : {int64_t(0), int64_t(4), T - 1}) {
            const KVCache before = c;
            check_close(m.rerun_final(p, rows(acts.at("hook_final"), p, p + 1), c), rows(full, p, p + 1));
            CHECK(c.size() == before.size());
            for (int64_t l = 0; l < L; ++l) check_close(c.keys(l), before.keys(l));
        }
        const Tensor f0 = rows(acts.at("hook_final"), 0, 1);
        CHECK_THROWS(m.rerun_final(T, f0, c));
        CHECK_THROWS(m.rerun_final(-1, f0, c));
        CHECK_THROWS(m.rerun_final(0, rows(acts.at("hook_final"), 0, 2), c));
        CHECK_THROWS(m.rerun_final(0, f0, none));
    }
    // Edits at every step = the same edits on the whole pass: zeroing a head's z, scaling keys (what
    // later tokens read from the cache) and editing the pattern.
    {
        Hooks edit;
        edit.add("blocks.0.attn.hook_z", [](const std::string&, Tensor& z) {
            const int64_t D = z.shape[2];
            for (int64_t r = 0; r < z.shape[0]; ++r) std::fill_n(z.ptr() + r * z.shape[1] * D, D, 0.f);
        });
        edit.add(std::format("blocks.{}.attn.hook_k", L - 1), [](const std::string&, Tensor& k) {
            for (float& x : k.data) x *= 1.5f;
        });
        edit.add("blocks.0.attn.hook_pattern", [](const std::string&, Tensor& P) {  // the first key's weight, doubled
            const int64_t nk = P.shape[2];
            for (int64_t r = 0; r < P.shape[0] * P.shape[1]; ++r) P.data[r * nk] *= 2.f;
        });
        const Tensor want = m.forward(t, &edit);
        KVCache c;
        std::vector<Tensor> parts{m.forward_cached(Tokens(t.begin(), t.begin() + 5), c, &edit)};
        for (int64_t i = 5; i < T; ++i) parts.push_back(m.forward_cached({t[i]}, c, &edit));
        check_close(stack_rows(parts), want);
    }
    // Another model's cache is refused; a failed call leaves the cache as it was.
    {
        KVCache c;
        m.forward_cached(Tokens(t.begin(), t.begin() + 3), c);
        const Model wider = random_model(tiny([](ModelConfig& cf) { cf.n_layers = 3; }), 1);
        CHECK_THROWS(wider.forward_cached({1}, c));
        Hooks bad;
        bad.add("blocks.0.attn.hook_k", [](const std::string&, Tensor& k) { k = Tensor({1, 1, 1}); });
        CHECK_THROWS(m.forward_cached({1, 2}, c, &bad));
        CHECK(c.size() == 3);
        check_close(m.forward_cached(Tokens(t.begin() + 3, t.end()), c), rows(full, 3, T));
    }
    if (m.cfg().positional == "learned") {  // positions past max_seq_len are refused, the cache unchanged
        KVCache c;
        m.forward_cached(seq(m.cfg().max_seq_len, 2), c);
        CHECK_THROWS(m.forward_cached({1}, c));
        CHECK(c.size() == m.cfg().max_seq_len);
    }
}

// forward_cached on a KVBatch: three sequences of different lengths, continued in uneven steps (one
// sitting a step out), each = its own whole pass; hooks see the stacked rows; hook_pattern is refused.
static void test_forward_cached_batch(const Model& m) {
    const std::vector<Tokens> text = {seq(14, 81), seq(11, 82), seq(16, 83)};
    std::vector<Tensor> full;
    for (const Tokens& t : text) full.push_back(m.forward(t));
    KVBatch batch(3);
    std::vector<int64_t> at = {0, 0, 0};
    std::vector<std::vector<Tensor>> parts(3);
    const std::vector<std::vector<int64_t>> steps = {{5, 9, 1}, {1, 0, 1}, {1, 1, 4}, {3, 1, 1}, {4, 0, 9}};
    for (const auto& step : steps) {
        std::vector<Tokens> toks(3);
        int64_t rows = 0;
        for (int i = 0; i < 3; ++i) {
            toks[i] = Tokens(text[i].begin() + at[i], text[i].begin() + at[i] + step[i]);
            rows += step[i];
        }
        Hooks seen;
        int64_t got_rows = -1;
        seen.add_capture([](const std::string& n) { return n == "blocks.0.hook_resid_post"; },
                         [&](const std::string&, Tensor& a) { got_rows = a.shape[0]; });
        const std::vector<Tensor> lg = m.forward_cached(toks, batch, &seen);
        CHECK(got_rows == rows);
        for (int i = 0; i < 3; ++i) {
            CHECK(lg[i].empty() == (step[i] == 0));
            if (step[i]) parts[i].push_back(lg[i]);
            at[i] += step[i];
            CHECK(batch[i].size() == at[i]);
        }
    }
    for (int i = 0; i < 3; ++i) check_close(stack_rows(parts[i]), rows(full[i], 0, at[i]));
    // last_only: each sequence's last row
    KVBatch b2(2);
    const std::vector<Tensor> last =
        m.forward_cached({Tokens(text[0].begin(), text[0].begin() + 6), Tokens(text[1].begin(), text[1].begin() + 3)},
                         b2, nullptr, true);
    check_close(last[0], rows(full[0], 5, 6));
    check_close(last[1], rows(full[1], 2, 3));
    Hooks pattern;
    pattern.add_capture([](const std::string& n) { return n == "blocks.0.attn.hook_pattern"; },
                        [](const std::string&, Tensor&) {});
    CHECK_THROWS(m.forward_cached({Tokens{1}, Tokens{2}}, b2, &pattern));
    CHECK_THROWS(m.forward_cached({Tokens{1}}, b2));  // one token list per sequence
}

// generate_batch = generate() on each prompt (seed + i), hooks reaching only their sequence, early
// stoppers sitting out, traces as generate()'s; bad arguments refused.
static void test_generate_batch(const Model& m) {
    const std::vector<Tokens> prompts = {seq(5, 91), seq(8, 92), seq(3, 93)};
    GenerateOptions o;
    o.max_new_tokens = 6;
    // greedy: each token is the whole pass's argmax (up to a near tie)
    const std::vector<Tokens> greedy = generate_batch(m, prompts, o);
    for (size_t i = 0; i < prompts.size(); ++i) {
        CHECK(greedy[i].size() == 6);
        Tokens text = prompts[i];
        for (int64_t tkn : greedy[i]) {
            const Tensor lg = m.forward(text);
            const float* last = lg.ptr() + (lg.shape[0] - 1) * lg.shape[1];
            const int64_t want = std::max_element(last, last + lg.shape[1]) - last;
            CHECK(tkn == want || std::fabs(last[tkn] - last[want]) < 1e-4f);
            text.push_back(tkn);
        }
    }
    // sampling: sequence i = generate() with seed + i; a hook for sequence 1 only
    o.temperature = 1.0;
    o.seed = 3;
    Hooks halve;
    halve.add("blocks.0.hook_attn_out", [](const std::string&, Tensor& a) {
        for (float& v : a.data) v *= 0.5f;
    });
    const std::vector<Tokens> sampled = generate_batch(m, prompts, o, {nullptr, &halve, nullptr});
    for (size_t i = 0; i < prompts.size(); ++i) {
        GenerateOptions oi = o;
        oi.seed = o.seed + i;
        CHECK(sampled[i] == generate(m, prompts[i], oi, i == 1 ? &halve : nullptr));
    }
    // a sequence that stops early sits the rest out; the others go on
    o.temperature = 0;
    o.stop_tokens = {greedy[0][0]};
    const std::vector<Tokens> stopped = generate_batch(m, prompts, o);
    CHECK(stopped[0].size() == 1 && stopped[0][0] == greedy[0][0]);
    for (size_t i = 1; i < prompts.size(); ++i) {
        GenerateOptions oi = o;
        CHECK(stopped[i] == generate(m, prompts[i], oi));
    }
    o.stop_tokens.clear();
    // traces: as generate()'s
    if (m.cfg().norm_position != "post") {
        std::vector<std::vector<TraceStep>> traces;
        const std::vector<Tokens> traced = generate_batch(m, prompts, o, {}, &traces);
        CHECK(traced == greedy && traces.size() == prompts.size());
        for (size_t i = 0; i < prompts.size() && i < traces.size(); ++i) {
            std::vector<TraceStep> one;
            generate(m, prompts[i], o, nullptr, &one);
            CHECK(traces[i].size() == one.size());
            for (size_t k = 0; k < one.size() && k < traces[i].size(); ++k) {
                CHECK(traces[i][k].token == one[k].token && traces[i][k].position == one[k].position);
                CHECK_NEAR(traces[i][k].prob, one[k].prob, 1e-4);
                for (size_t s = 0; s < one[k].lens.size() && s < traces[i][k].lens.size(); ++s)
                    CHECK_NEAR(traces[i][k].lens[s].target_prob, one[k].lens[s].target_prob, 1e-4);
                for (size_t c = 0; c < one[k].dla.values.size() && c < traces[i][k].dla.values.size(); ++c)
                    CHECK_NEAR(traces[i][k].dla.values[c], one[k].dla.values[c],
                               1e-3 * std::max(1.f, std::fabs(one[k].dla.values[c])));
            }
        }
    }
    CHECK_THROWS(generate_batch(m, prompts, o, {nullptr}));  // hooks: none or one per prompt
    GenerateOptions cb = o;
    cb.on_token = [](int64_t) { return true; };
    CHECK_THROWS(generate_batch(m, prompts, cb));
    GenerateOptions att = o;
    att.trace.attention_top = 2;
    std::vector<std::vector<TraceStep>> t;
    CHECK_THROWS(generate_batch(m, prompts, att, {}, &t));
    CHECK(generate_batch(m, {}, o).empty());
}

// Hooks rounding every layer's keys and values to fp16 values (clamped to fp16's range): what a model with
// fp16 weights stores in its KV cache, as a whole pass's reference.
static Hooks round_kv_to_half(const Model& m) {
    Hooks h;
    for (int64_t l = 0; l < m.cfg().n_layers; ++l)
        for (const char* n : {".attn.hook_k", ".attn.hook_v"})
            h.add(std::format("blocks.{}{}", l, n), [](const std::string&, Tensor& a) {
                for (float& v : a.data)
                    v = detail::half_to_float(detail::float_to_half(std::clamp(v, -65504.f, 65504.f)));
            });
    return h;
}

// With fp16 weights the KV cache holds fp16 values: cached decoding (in chunks, and a batch) = the whole pass
// with keys and values rounded to fp16, up to an fp16 step where the two round differently; keys() are
// fp16 values; overflow is clamped, not infinite; a float32 model can't continue the cache, nor the reverse.
static void test_fp16_kv(const Model& m16, const ModelConfig& cfg) {
    const Hooks round = round_kv_to_half(m16);
    const Tokens t = seq(12, 74);
    ActivationCache acts;
    const Tensor want = m16.run_with_cache(t, acts, nullptr, &round);
    // An fp16 step of difference in a key moves logits by ~1e-3; on the GPU the multiplies' fp16 inputs add ~1e-3 each.
    const bool gpu = device() != Device::Cpu && m16.gpu();
    const double rel = gpu ? 2e-2 : 2e-3, of_scale = gpu ? 2e-3 : 1e-4;
    auto close_to = [](const Tensor& got, const Tensor& w, double rel, double of_scale) {
        CHECK(got.shape == w.shape);
        const float sc = max_abs(w);
        for (int64_t i = 0; i < got.numel() && got.shape == w.shape; ++i)
            CHECK_NEAR(got.data[i], w.data[i], rel * std::max(1.f, std::fabs(w.data[i])) + of_scale * sc);
    };
    auto close = [&](const Tensor& got, const Tensor& w) { close_to(got, w, rel, of_scale); };
    KVCache cache;
    std::vector<Tensor> parts{m16.forward_cached(Tokens(t.begin(), t.begin() + 6), cache)};
    for (int64_t i : {6, 7}) parts.push_back(m16.forward_cached({t[i]}, cache));
    parts.push_back(m16.forward_cached(Tokens(t.begin() + 8, t.end()), cache));
    close(stack_rows(parts), want);
    for (int64_t l = 0; l < m16.cfg().n_layers; ++l) {
        const Tensor k = cache.keys(l);
        close(k, acts.at(std::format("blocks.{}.attn.hook_k", l)));
        bool exact = true;  // every stored value an fp16 one
        for (float v : k.data) exact = exact && detail::half_to_float(detail::float_to_half(v)) == v;
        CHECK(exact);
    }
    // A chunk of 20 tokens continuing a cache: more query rows than the fp16 prefix is read a row at a time for (it
    // is widened as a block and multiplied in the library).
    {
        const Tokens t2 = seq(24, 75);
        const Tensor want2 = m16.forward(t2, &round);
        KVCache c2;
        std::vector<Tensor> p2{m16.forward_cached(Tokens(t2.begin(), t2.begin() + 4), c2)};
        p2.push_back(m16.forward_cached(Tokens(t2.begin() + 4, t2.end()), c2));
        // 24 positions through up to 12 layers: more keys near an fp16 rounding boundary, each a step (~3e-4 of the
        // scale) where the two round differently.
        close_to(stack_rows(p2), want2, rel, std::max(of_scale, 5e-4));
    }
    // Over a long prefix the key / value heads go side by side on the CPU: the same bits as one after another, for
    // any thread count (forced here from one cached key).
    if (device() == Device::Cpu) {
        auto steps = [&](int64_t threshold, int64_t threads) {
            detail::set_kv_side_by_side(threshold);
            set_num_threads(threads);
            KVCache c3;
            std::vector<Tensor> p3{m16.forward_cached(Tokens(t.begin(), t.begin() + 6), c3)};
            for (int64_t i = 6; i < 12; ++i) p3.push_back(m16.forward_cached({t[static_cast<size_t>(i)]}, c3));
            detail::set_kv_side_by_side(0);
            set_num_threads(0);
            return stack_rows(p3);
        };
        const Tensor in_turn = steps(int64_t(1) << 40, 0);
        CHECK(steps(1, 0).data == in_turn.data && steps(1, 1).data == in_turn.data && steps(1, 3).data == in_turn.data);
    }
    KVBatch batch(2);
    const std::vector<Tensor> two =
        m16.forward_cached({Tokens(t.begin(), t.begin() + 5), Tokens(t.begin(), t.begin() + 9)}, batch);
    close(two[0], rows(want, 0, 5));
    close(two[1], rows(want, 0, 9));
    // overflow: keys pushed past fp16's range are clamped to it, not infinite, and each step's own rows are
    // rounded (so clamped) before attention reads them, as the earlier positions are: = the whole pass that
    // pushes them, then rounds. (Not rounding a step's rows first would make it attend to ~1e6, not 65504.)
    Hooks huge;
    huge.add("blocks.0.attn.hook_k", [](const std::string&, Tensor& a) {
        for (float& v : a.data) v *= 1e6f;
    });
    Hooks huge_then_round;
    huge_then_round.extend(huge).extend(round);
    const Tensor want_big = m16.forward(t, &huge_then_round);
    KVCache big;
    std::vector<Tensor> big_parts{m16.forward_cached(Tokens(t.begin(), t.begin() + 4), big, &huge)};
    for (int64_t i : {4, 5, 6}) big_parts.push_back(m16.forward_cached({t[i]}, big, &huge));
    // Keys at 65504 give scores around 1e6, where adding a dot product in another order (an fp16 cache's keys are
    // read a row at a time; the whole pass multiplies in the library) moves a score by ~0.1 and a peaked softmax
    // with it: a looser tolerance for this ill-conditioned case (measured: 2.8e-3).
    close_to(stack_rows(big_parts), rows(want_big, 0, 7), std::max(rel, 1e-2), std::max(of_scale, 1e-2));
    bool finite = true;
    for (float v : big.keys(0).data) finite = finite && std::isfinite(v) && std::fabs(v) <= 65504.f;
    CHECK(finite);
    // precisions don't mix
    const Model m32 = random_model(cfg, 42);
    CHECK_THROWS(m32.forward_cached({1}, cache));
    KVCache c32;
    m32.forward_cached({1, 2}, c32);
    CHECK_THROWS(m16.forward_cached({3}, c32));
    CHECK_THROWS(m32.forward_cached({Tokens{1}, Tokens{}}, batch));
}

// KVCache::clamped(): an fp16 cache counts the keys and values it cut off at fp16's limit as it stored them, from CPU
// and GPU steps alike (on the current device); truncating drops the dropped positions' counts. A float32 cache, or
// one whose values all fit, counts none. generate() and generate_batch() pass the count on (kv_clamped).
static void test_kv_clamped(const Model& m16, const Model& m32) {
    const Tokens t = seq(7, 75);
    Hooks huge;
    huge.add("blocks.0.attn.hook_k", [](const std::string&, Tensor& a) {
        for (float& v : a.data) v *= 1e6f;
    });
    // Before rounding: values past 65520 (the first that rounds to infinity) must be cut off, and from 65488 up
    // they may round to 65504.
    int64_t must = 0, may = 0;
    Hooks seen;
    seen.extend(huge);
    seen.add_capture([](const std::string& n) { return n.ends_with(".hook_k") || n.ends_with(".hook_v"); },
                     [&](const std::string&, Tensor& a) {
                         for (float v : a.data) {
                             must += std::fabs(v) > 65520.f;
                             may += std::fabs(v) >= 65488.f;
                         }
                     });
    {
        const Device d = device();
        set_device(Device::Cpu);
        m16.forward(t, &seen);
        set_device(d);
    }
    KVCache c;
    m16.forward_cached(Tokens(t.begin(), t.begin() + 4), c, &huge);
    for (int64_t i : {4, 5, 6}) m16.forward_cached({t[i]}, c, &huge);
    const int64_t n = c.clamped();
    CHECK(must > 0 && must <= n && n <= may);
    int64_t at_limit = 0;
    for (int64_t l = 0; l < m16.cfg().n_layers; ++l)
        for (const Tensor& kv : {c.keys(l), c.values(l)})
            for (float v : kv.data) at_limit += std::fabs(v) == 65504.f;
    CHECK(at_limit == n);
    CHECK(KVCache(c).clamped() == n);
    c.truncate(4);
    CHECK(c.clamped() > 0 && c.clamped() < n);
    c.clear();
    CHECK(c.clamped() == 0);
    KVCache plain, c32;
    m16.forward_cached(t, plain);
    m32.forward_cached(t, c32, &huge);
    CHECK(plain.clamped() == 0 && c32.clamped() == 0);
    GenerateOptions o;
    o.max_new_tokens = 2;
    int64_t got = -1;
    o.kv_clamped = &got;
    generate(m16, t, o, &huge);
    CHECK(got > 0);
    generate(m16, t, o);
    CHECK(got == 0);
    got = -1;
    generate_batch(m16, {t, Tokens(t.begin(), t.begin() + 3)}, o, {nullptr, &huge});
    CHECK(got > 0);
    generate_batch(m16, {t, t}, o);
    CHECK(got == 0);
}

// generate(): greedy = argmax of forward() step by step; stop tokens, max_new_tokens, on_token; hooks
// apply at every step; sampling is reproducible from its seed and follows top_k / top_p.
static void test_generate(const Model& m) {
    const Tokens prompt = seq(5, 71);
    GenerateOptions o;
    o.max_new_tokens = 8;
    const Tokens got = generate(m, prompt, o);
    CHECK(got.size() == 8);
    Tokens seqn = prompt;
    for (int64_t step = 0; step < 8; ++step) {
        const Tensor lg = m.forward(seqn);
        const int64_t V = lg.shape[1];
        const float* last = lg.ptr() + (lg.shape[0] - 1) * V;
        // The cached pass's argmax may differ from the full pass's only where two logits nearly tie.
        const int64_t want = std::max_element(last, last + V) - last;
        CHECK(got[step] == want || std::fabs(last[got[step]] - last[want]) < 1e-4f);
        seqn.push_back(got[step]);
    }
    // Stops at a stop token (returned) and when on_token says so.
    o.stop_tokens = {got[2]};
    const Tokens stopped = generate(m, prompt, o);
    CHECK(!stopped.empty() && stopped.back() == got[2] && stopped.size() <= 3);
    o.stop_tokens.clear();
    int calls = 0;
    o.on_token = [&](int64_t) { return ++calls < 4; };
    CHECK(generate(m, prompt, o).size() == 4 && calls == 4);
    o.on_token = nullptr;
    o.max_new_tokens = 0;
    CHECK(generate(m, prompt, o).empty());
    o.max_new_tokens = 8;
    // Hooks at every step: the same tokens as forward() with the hook, step by step.
    Hooks zero;
    zero.add("blocks.0.hook_attn_out", [](const std::string&, Tensor& a) { std::ranges::fill(a.data, 0.f); });
    const Tokens hooked = generate(m, prompt, o, &zero);
    seqn = prompt;
    for (int64_t step = 0; step < 8; ++step) {
        const Tensor lg = m.forward(seqn, &zero);
        const float* last = lg.ptr() + (lg.shape[0] - 1) * lg.shape[1];
        const int64_t want = std::max_element(last, last + lg.shape[1]) - last;
        CHECK(hooked[step] == want || std::fabs(last[hooked[step]] - last[want]) < 1e-4f);
        seqn.push_back(hooked[step]);
    }
    // Sampling: the same seed gives the same tokens; others differ somewhere.
    o.temperature = 1.0;
    o.seed = 5;
    const Tokens s1 = generate(m, prompt, o), s2 = generate(m, prompt, o);
    CHECK(s1 == s2);
    bool differs = false;
    o.temperature = 1e3;  // nearly uniform, whatever the logits' scale (models without norms reach ~100)
    const Tokens s3 = generate(m, prompt, o);
    for (uint64_t seed = 6; seed < 12 && !differs; ++seed) {
        o.seed = seed;
        differs = generate(m, prompt, o) != s3;
    }
    CHECK(differs);
    o.temperature = 1.0;
    o.top_k = 1;  // top_k 1 is greedy
    CHECK(generate(m, prompt, o) == got);
    o.top_k = 0;
    o.temperature = -1;
    CHECK_THROWS(generate(m, prompt, o));
    o.temperature = 1;
    o.top_p = 0;
    CHECK_THROWS(generate(m, prompt, o));
    o.top_p = 1;
    CHECK_THROWS(generate(m, {}, o));
}

// Two-digit layers: the cached pass recognizes each layer's keys and values by their hook names
// ("blocks.11.attn.hook_k"), in float32 and with the fp16 cache's rounding.
static void test_forward_cached_many_layers() {
    const ModelConfig c = tiny([](ModelConfig& x) { x.n_layers = 12; });
    const Model m = random_model(c, 61);
    const Tokens t = seq(9, 62);
    const Tensor full = m.forward(t);
    KVCache cache;
    std::vector<Tensor> parts{m.forward_cached(Tokens(t.begin(), t.begin() + 5), cache)};
    for (size_t i = 5; i < t.size(); ++i) parts.push_back(m.forward_cached({t[i]}, cache));
    const Tensor got = stack_rows(parts);
    float worst = 0, scale = 1;
    for (size_t i = 0; i < full.data.size(); ++i) {
        worst = std::max(worst, std::fabs(got.data[i] - full.data[i]));
        scale = std::max(scale, std::fabs(full.data[i]));
    }
    CHECK(got.shape == full.shape && worst < 1e-4f * scale);
    CHECK(cache.keys(11).shape[1] == 9 && cache.values(10).shape[1] == 9);
    const Model m16 = random_model(c, 61, WeightFormat::fp16);
    const Device before = device();
    set_device(Device::Cpu);
    test_fp16_kv(m16, c);
    test_kv_clamped(m16, m);
#ifdef TFI_USE_ACCELERATE
    if (detail::Gpu::create()) {  // GPU steps round and clamp on the GPU
        set_device(Device::Gpu);
        test_kv_clamped(m16, m);
        CHECK(m16.has_gpu());
    }
#endif
    set_device(before);
}

// generate()'s trace: each step = the full-pass analyses at that position on the text so far (the
// chosen token's probability and top tokens, logit lens, DLA, where each head attended), hooks included.
static void test_generate_trace(const Model& m) {
    const Tokens prompt = seq(6, 73);
    GenerateOptions o;
    o.max_new_tokens = 5;
    o.trace.top_k = 3;
    o.trace.attention_top = 2;
    const bool post = m.cfg().norm_position == "post";
    o.trace.dla = !post;
    Hooks zero;  // an intervention: the trace must show the model as edited
    zero.add("blocks.0.hook_attn_out", [](const std::string&, Tensor& a) {
        for (float& v : a.data) v *= 0.5f;
    });
    for (const Hooks* hooks : {static_cast<const Hooks*>(nullptr), static_cast<const Hooks*>(&zero)}) {
        std::vector<TraceStep> trace;
        const Tokens got = generate(m, prompt, o, hooks, &trace);
        CHECK(trace.size() == got.size());
        Tokens text = prompt;
        for (size_t i = 0; i < trace.size() && i < got.size(); ++i) {
            const TraceStep& st = trace[i];
            const int64_t pos = std::ssize(text) - 1;
            CHECK(st.token == got[i] && st.position == pos);
            ActivationCache acts;
            const Tensor lg = m.run_with_cache(text, acts, nullptr, hooks);
            Tensor p = rows(lg, pos, pos + 1);
            softmax_(p.data);
            CHECK_NEAR(st.prob, p.data[st.token], 1e-4);
            CHECK_NEAR(st.logprob, std::log(p.data[st.token]), 1e-3);
            CHECK(st.top_tokens.size() == 3 && st.top_probs[0] >= st.top_probs[2]);
            const std::vector<LensEntry> lens = logit_lens_from(m, acts, pos, st.token, 3);
            CHECK(st.lens.size() == lens.size());
            for (size_t s = 0; s < lens.size() && s < st.lens.size(); ++s) {
                CHECK(st.lens[s].stage == lens[s].stage);
                CHECK_NEAR(st.lens[s].target_prob, lens[s].target_prob, 1e-4);
                CHECK(std::abs(st.lens[s].target_rank - lens[s].target_rank) <= 1);  // near ties may swap
            }
            if (!post) {
                const DLAResult dla = direct_logit_attribution_from(m, acts, pos, st.token);
                CHECK(st.dla.labels == dla.labels);
                for (size_t k = 0; k < dla.values.size() && k < st.dla.values.size(); ++k)
                    CHECK_NEAR(st.dla.values[k], dla.values[k], 1e-3 * std::max(1.f, std::fabs(dla.values[k])));
                // Parts add up to the logit, unless the hook edited attention's output after the heads' z,
                // which DLA attributes from (the full pass then misses it the same way).
                if (!hooks) CHECK_NEAR(st.dla.sum, st.dla.actual, 1e-3 * std::max(1.f, std::fabs(st.dla.actual)));
            }
            size_t a = 0;  // the heads' most attended positions = the whole pass's pattern row
            for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
                const Tensor& P = acts.at(std::format("blocks.{}.attn.hook_pattern", l));
                const int64_t H = P.shape[0], T = P.shape[1];
                for (int64_t h = 0; h < H && a < st.attention.size(); ++h, ++a) {
                    const TraceStep::Attention& at = st.attention[a];
                    CHECK(at.head == std::format("L{}H{}", l, h) && at.positions.size() == 2);
                    for (size_t j = 0; j < at.positions.size(); ++j)
                        CHECK_NEAR(at.weights[j], P.data[(h * T + pos) * T + at.positions[j]], 1e-4);
                }
            }
            CHECK(a == st.attention.size());
            text.push_back(got[i]);
        }
    }
    if (post) {  // DLA is refused up front on post-norm models, not mid-generation
        GenerateOptions bad = o;
        bad.trace.dla = true;
        std::vector<TraceStep> t;
        CHECK_THROWS(generate(m, prompt, bad, nullptr, &t));
    }
}

// Speculative decoding: greedy output = plain greedy's whatever proposes (prompt lookup, another model,
// the model itself), with hooks too; the trace = plain generation's; the counts; the position limit.
static void test_generate_speculative(const Model& m, const Model& other) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    Tokens prompt = seq(6, 81);
    prompt.insert(prompt.end(), prompt.begin(), prompt.end());  // repeated: prompt lookup finds matches
    GenerateOptions o;
    o.max_new_tokens = 14;
    o.stop_at_eos = false;
    const Tokens plain = generate(m, prompt, o);
    SpeculativeStats st;
    o.speculative.stats = &st;
    for (int64_t k : {1, 3, 8}) {
        o.speculative.draft_tokens = k;
        for (const Model* d : {static_cast<const Model*>(nullptr), &other, &m}) {
            o.speculative.draft = d;
            CHECK(generate(m, prompt, o) == plain);
            // Each pass gives the proposals it keeps and one token of the model's.
            CHECK(st.steps >= 1 && st.accepted <= st.drafted && st.steps + st.accepted == std::ssize(plain));
            if (d == &m)  // its own draft: the prompt's pass gives one token, then every proposal kept, k + 1 a step
                CHECK(st.accepted == st.drafted && st.steps == 1 + (13 + k) / (k + 1));
        }
    }
    o.speculative.draft = nullptr;
    o.speculative.ngram = 1;
    CHECK(generate(m, prompt, o) == plain);
    o.speculative.ngram = 3;
    // Hooks: the tokens of plain generation with the same hooks (the draft runs without them).
    Hooks half, head;
    half.add("blocks.0.hook_attn_out", [](const std::string&, Tensor& a) {
        for (float& v : a.data) v *= 0.5f;
    });
    head.add("blocks.0.attn.hook_z", [](const std::string&, Tensor& z) {  // one head, in every row
        const int64_t H = z.shape[1], D = z.shape[2];
        for (int64_t r = 0; r < z.shape[0]; ++r) std::fill(z.ptr() + r * H * D, z.ptr() + (r * H + 1) * D, 0.f);
    });
    o.speculative.draft_tokens = 0;
    const Tokens plain_hooked = generate(m, prompt, o, &half), plain_head = generate(m, prompt, o, &head);
    o.speculative.draft_tokens = 3;
    for (const Model* d : {static_cast<const Model*>(nullptr), &m}) {
        o.speculative.draft = d;
        CHECK(generate(m, prompt, o, &half) == plain_hooked);
        CHECK(generate(m, prompt, o, &head) == plain_head);
    }
    // hook_final zeroing its row now and then (all logits equal: greedy takes id 0): a guess it was predicted to
    // leave alone gets its logits computed again from the real output.
    {
        int calls = 0;
        Hooks zero;
        zero.add("hook_final", [&](const std::string&, Tensor& a) {
            const int k = calls++;
            if (k == 4 || k == 9 || k == 11) std::ranges::fill(a.data, 0.f);
        });
        o.speculative.draft_tokens = 0;
        const Tokens want = generate(m, prompt, o, &zero);
        CHECK(want != plain);
        o.speculative.draft_tokens = 4;
        for (const Model* d : {static_cast<const Model*>(nullptr), &m}) {
            o.speculative.draft = d;
            calls = 0;
            CHECK(generate(m, prompt, o, &zero) == want);
        }
    }
    // Greedy picks from top_k 1 at a temperature: proposals checked by id, as at temperature 0.
    o.speculative.draft = &other;
    o.temperature = 0.8;
    o.top_k = 1;
    CHECK(generate(m, prompt, o) == plain);
    o.temperature = 0;
    o.top_k = 0;
    // The trace: plain generation's steps, rejected proposals left out.
    const bool post = m.cfg().norm_position == "post";
    o.trace.top_k = 3;
    o.trace.attention_top = 2;
    o.trace.dla = !post;
    std::vector<TraceStep> t0, t1;
    o.speculative.draft_tokens = 0;
    generate(m, prompt, o, &half, &t0);
    o.speculative.draft_tokens = 3;
    o.speculative.draft = &other;
    generate(m, prompt, o, &half, &t1);
    // Tolerances as test_generate_trace's: on the GPU a step of several rows multiplies differently from
    // one of a single row, and models without norms have logits near 100.
    CHECK(t0.size() == t1.size());
    for (size_t i = 0; i < t0.size() && i < t1.size(); ++i) {
        const TraceStep &a = t0[i], &b = t1[i];
        CHECK(a.token == b.token && a.position == b.position && a.top_tokens == b.top_tokens);
        CHECK_NEAR(a.prob, b.prob, 1e-4);
        CHECK(a.lens.size() == b.lens.size());
        for (size_t j = 0; j < a.lens.size() && j < b.lens.size(); ++j)
            CHECK_NEAR(a.lens[j].target_prob, b.lens[j].target_prob, 1e-4);
        CHECK(a.dla.labels == b.dla.labels);
        for (size_t j = 0; j < a.dla.values.size() && j < b.dla.values.size(); ++j)
            CHECK_NEAR(a.dla.values[j], b.dla.values[j], 1e-3 * std::max(1.f, std::fabs(a.dla.values[j])));
        CHECK(a.attention.size() == b.attention.size());
        for (size_t j = 0; j < a.attention.size() && j < b.attention.size(); ++j) {
            CHECK(a.attention[j].head == b.attention[j].head && a.attention[j].positions == b.attention[j].positions);
            for (size_t q = 0; q < a.attention[j].weights.size() && q < b.attention[j].weights.size(); ++q)
                CHECK_NEAR(a.attention[j].weights[q], b.attention[j].weights[q], 1e-4);
        }
    }
    o.trace = {};
    o.speculative.draft = nullptr;
    // Learned positions: as far as plain generation goes (proposals stop at the last position), no further.
    if (m.cfg().positional == "learned") {
        const int64_t most = m.cfg().max_seq_len - std::ssize(prompt) + 1;  // the last from position max - 1
        int64_t got = 0;
        o.max_new_tokens = 100;  // stopped by on_token instead, so proposals aren't already limited by max_new_tokens
        o.on_token = [&](int64_t) { return ++got < most; };
        o.speculative.draft_tokens = 0;
        const Tokens longest = generate(m, prompt, o);
        CHECK(std::ssize(longest) == most);
        o.speculative.draft_tokens = 8;
        for (const Model* d : {static_cast<const Model*>(nullptr), &other}) {
            o.speculative.draft = d;
            got = 0;
            CHECK(generate(m, prompt, o) == longest);
        }
        o.on_token = nullptr;
        o.max_new_tokens = most + 1;
        CHECK_THROWS(generate(m, prompt, o));
        o.speculative.draft_tokens = 0;
        CHECK_THROWS(generate(m, prompt, o));
        o.max_new_tokens = 14;
    }
    // Stops at a stop token and when on_token says so, mid-step.
    o.speculative.draft = &m;
    o.speculative.draft_tokens = 8;
    o.stop_tokens = {plain[4]};
    const Tokens stopped = generate(m, prompt, o);
    CHECK(!stopped.empty() && stopped.back() == plain[4] && stopped.size() <= 5);
    o.stop_tokens.clear();
    int calls = 0;
    o.on_token = [&](int64_t) { return ++calls < 3; };
    CHECK(generate(m, prompt, o).size() == 3 && calls == 3);
    o.on_token = nullptr;
    o.max_new_tokens = 0;
    CHECK(generate(m, prompt, o).empty() && st.steps == 0);
    o.max_new_tokens = 1;
    CHECK(generate(m, prompt, o) == Tokens{plain[0]});
    CHECK_THROWS(generate_batch(m, {prompt}, o));
    o.speculative.draft_tokens = -1;
    CHECK_THROWS(generate(m, prompt, o));
    o.speculative.draft_tokens = 2;
    o.speculative.ngram = 0;
    CHECK_THROWS(generate(m, prompt, o));
}

// Speculative sampling draws from the model's distribution: the joint frequencies of three new tokens, over
// many seeds, against the exact probabilities (forward passes, Sampler::distribution), for prompt lookup and
// a draft model of another distribution (and another vocabulary size), with temperature, top_k and top_p.
static void test_speculative_sampling() {
    const ModelConfig c = tiny([](ModelConfig& x) { x.vocab_size = 5; });
    const ModelConfig c6 = tiny([](ModelConfig& x) { x.vocab_size = 6; });
    const Model m = random_model(c, 3), draft = random_model(c6, 4);
    const Tokens prompt = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};  // every token occurs earlier: prompt lookup always proposes
    const int n = 4000;
    int64_t rejected = 0;  // over all settings: the test weights differ between standard libraries
    for (int setting = 0; setting < 2; ++setting) {
        GenerateOptions o;
        o.max_new_tokens = 3;  // the first from the prompt's pass, the next two from a speculative step
        o.stop_at_eos = false;
        o.temperature = setting ? 5.0 : 3.0;
        o.top_k = setting ? 4 : 0;
        o.top_p = setting ? 0.9 : 1.0;
        std::vector<double> want(125, 0), p1, p2, p3;
        Sampler exact(o);
        const Tensor l1 = m.forward(prompt);
        exact.distribution(l1.row_view(l1.shape[0] - 1), p1);
        for (int64_t a = 0; a < 5; ++a) {
            Tokens t = prompt;
            t.push_back(a);
            const Tensor l2 = m.forward(t);
            exact.distribution(l2.row_view(l2.shape[0] - 1), p2);
            for (int64_t b = 0; b < 5; ++b) {
                Tokens u = t;
                u.push_back(b);
                const Tensor l3 = m.forward(u);
                exact.distribution(l3.row_view(l3.shape[0] - 1), p3);
                for (int64_t e = 0; e < 5; ++e) want[static_cast<size_t>((a * 5 + b) * 5 + e)] = p1[a] * p2[b] * p3[e];
            }
        }
        for (const Model* d : {static_cast<const Model*>(nullptr), &draft}) {
            o.speculative.draft = d;
            o.speculative.draft_tokens = 2;
            SpeculativeStats st;
            o.speculative.stats = &st;
            std::vector<double> got(125, 0);
            for (int i = 0; i < n; ++i) {
                o.seed = static_cast<uint64_t>(i);
                const Tokens t = generate(m, prompt, o);
                got[static_cast<size_t>((t[0] * 5 + t[1]) * 5 + t[2])] += 1.0 / n;
                rejected += st.drafted - st.accepted;
            }
            for (size_t j = 0; j < 125; ++j)
                CHECK_NEAR(got[j], want[j], 5 * std::sqrt(want[j] * (1 - want[j]) / n) + 1e-3);
        }
    }
    CHECK(rejected > n / 10);  // the rejection branch ran
}

// A draft with a smaller vocabulary: once the text holds a token it has no row for, the model continues
// alone, from the prompt on if the prompt holds one.
static void test_speculative_small_draft() {
    const Model m = random_model(tiny(), 42), draft = random_model(tiny([](ModelConfig& c) { c.vocab_size = 20; }), 43);
    GenerateOptions o;
    o.max_new_tokens = 8;
    o.stop_at_eos = false;
    SpeculativeStats st;
    auto speculate = [&](const Tokens& prompt) {
        GenerateOptions s = o;
        s.speculative.draft_tokens = 3;
        s.speculative.draft = &draft;
        s.speculative.stats = &st;
        return generate(m, prompt, s);
    };
    int left = 0;
    for (uint64_t seed = 0; seed < 20; ++seed) {
        const Tokens prompt = seq(6, seed, 20), plain = generate(m, prompt, o);
        left += std::ranges::any_of(plain.begin(), plain.end() - 1, [](int64_t t) { return t >= 20; });
        CHECK(speculate(prompt) == plain);
        CHECK(st.steps + st.accepted == std::ssize(plain));
    }
    CHECK(left > 0);                                // the case in question ran
    const Tokens outside = {1, 2, 3, 30, 1, 2, 3};  // repeated: falling back to prompt lookup would propose
    CHECK(speculate(outside) == generate(m, outside, o));
    CHECK(st.drafted == 0 && st.steps == o.max_new_tokens);
    o.temperature = 1.0;
    for (uint64_t seed = 0; seed < 20; ++seed) {
        o.seed = seed;
        CHECK(speculate(seq(6, seed, 20)).size() == 8);
        CHECK(speculate(outside).size() == 8);
    }
}

// One hook setup for the runtime tests: `calls` receives each call's hook point and shape, `state` whatever the
// hooks remember, so a run's whole visible behavior can be compared with decoding without speculation.
struct RuntimeHookRun {
    std::vector<std::pair<std::string, Shape>> calls;
    std::vector<float> state;
    Hooks hooks;
};
using RuntimeHookSetup = std::function<void(RuntimeHookRun&)>;

static std::vector<std::pair<std::string, RuntimeHookSetup>> runtime_hook_setups(const Model& m) {
    const std::string first = "blocks.0.", last = std::format("blocks.{}.", m.cfg().n_layers - 1);
    const std::string A = first + "hook_resid_post", B = last + "hook_resid_post", Z = first + "attn.hook_z",
                      K = first + "attn.hook_k", P = first + "attn.hook_pattern";
    using Edit = std::function<void(Tensor&, std::vector<float>&)>;
    auto on = [](const std::string& point, Edit edit) -> RuntimeHookSetup {
        return [point, edit](RuntimeHookRun& r) {
            r.hooks.add(point, [&r, edit](const std::string& n, Tensor& a) {
                r.calls.push_back({n, a.shape});
                edit(a, r.state);
            });
        };
    };
    auto row = [](const Tensor& a) { return a.numel() / a.shape[0]; };
    auto negate = [](Tensor& a, int64_t from, int64_t to) {
        for (int64_t i = from; i < to; ++i) a.ptr()[i] = -a.ptr()[i];
    };
    auto count = [](std::vector<float>& s) {
        if (s.empty()) s.push_back(0);
        return ++s[0];
    };
    std::vector<std::pair<std::string, RuntimeHookSetup>> v;
    v.push_back({"a[-1]", on(A, [=](Tensor& a, std::vector<float>&) { negate(a, a.numel() - row(a), a.numel()); })});
    v.push_back({"a[0]", on(A, [=](Tensor& a, std::vector<float>&) { negate(a, 0, row(a)); })});
    v.push_back({"a - a.mean()", on(A, [=](Tensor& a, std::vector<float>&) {
                     const int64_t w = row(a), n = a.shape[0];
                     std::vector<float> mean(static_cast<size_t>(w), 0.f);
                     for (int64_t r = 0; r < n; ++r)
                         for (int64_t j = 0; j < w; ++j)
                             mean[static_cast<size_t>(j)] += a.ptr()[r * w + j] / static_cast<float>(n);
                     for (int64_t r = 0; r < n; ++r)
                         for (int64_t j = 0; j < w; ++j)
                             a.ptr()[r * w + j] = 0.5f * a.ptr()[r * w + j] - 0.5f * mean[static_cast<size_t>(j)];
                 })});
    v.push_back({"len(a) == 1", on(A, [=](Tensor& a, std::vector<float>&) {
                     if (a.shape[0] == 1) negate(a, 0, a.numel());
                 })});
    v.push_back({"counter", on(A, [=](Tensor& a, std::vector<float>& s) {
                     if (static_cast<int64_t>(count(s)) % 3 == 0) negate(a, 0, a.numel());
                 })});
    v.push_back(
        {"log", on(A, [](Tensor& a, std::vector<float>& s) { s.insert(s.end(), a.ptr(), a.ptr() + a.numel()); })});
    v.push_back({"random noise", [A](RuntimeHookRun& r) {
                     auto rng = std::make_shared<std::mt19937>(5);
                     r.hooks.add(A, [&r, rng](const std::string& n, Tensor& a) {
                         r.calls.push_back({n, a.shape});
                         for (float& x : a.data) x += 0.2f * (static_cast<float>((*rng)() >> 8) * 0x1p-24f - 0.5f);
                         r.state.assign(1, static_cast<float>((*rng)() >> 8));
                     });
                 }});
    v.push_back({"value from the last call", on(A, [=](Tensor& a, std::vector<float>& s) {
                     const int64_t w = row(a);
                     const std::vector<float> keep(a.ptr() + a.numel() - w, a.ptr() + a.numel());
                     if (!s.empty())
                         for (int64_t i = 0; i < a.numel(); ++i)
                             a.ptr()[i] += 0.7f * std::tanh(s[static_cast<size_t>(i % w)]);
                     s = keep;
                 })});
    v.push_back({"earlier token's later layer", [A](RuntimeHookRun& r) {
                     r.hooks.add(A, [&r](const std::string& n, Tensor& a) {
                         r.calls.push_back({n, a.shape});
                         const int64_t w = a.numel() / a.shape[0];
                         if (!r.state.empty())
                             for (int64_t i = 0; i < a.numel(); ++i)
                                 a.ptr()[i] += 0.7f * std::tanh(r.state[static_cast<size_t>(i % w)]);
                     });
                     r.hooks.add("hook_final", [&r](const std::string& n, Tensor& a) {
                         r.calls.push_back({n, a.shape});
                         const int64_t w = a.numel() / a.shape[0];
                         r.state.assign(a.ptr() + a.numel() - w, a.ptr() + a.numel());
                     });
                 }});
    v.push_back({"first call only", on(A, [=](Tensor& a, std::vector<float>& s) {
                     if (count(s) == 1) negate(a, a.numel() - row(a), a.numel());
                 })});
    v.push_back({"alternating edits", on(A, [=](Tensor& a, std::vector<float>& s) {
                     if (static_cast<int64_t>(count(s)) % 2 == 0) negate(a, 0, a.numel());
                 })});
    v.push_back({"same edit on hook_z", on(Z, [](Tensor& z, std::vector<float>&) {
                     const int64_t H = z.shape[1], D = z.shape[2];
                     for (int64_t r = 0; r < z.shape[0]; ++r)
                         std::fill(z.ptr() + r * H * D, z.ptr() + (r * H + 1) * D, 0.f);
                 })});
    v.push_back({"hook_k, what later tokens read", on(K, [](Tensor& k, std::vector<float>&) {
                     for (float& x : k.data) x *= 0.5f;
                 })});
    v.push_back({"hook_pattern", on(P, [](Tensor& p, std::vector<float>& s) {
                     s.assign(1, static_cast<float>(p.shape[2]));
                     const int64_t H = p.shape[0], T = p.shape[1], K2 = p.shape[2];
                     for (int64_t r = 0; r < H * T; ++r) {
                         float* row2 = p.ptr() + r * K2;
                         std::reverse(row2, row2 + K2);
                     }
                 })});
    v.push_back({"capture", [B](RuntimeHookRun& r) {
                     r.hooks.add_capture([B](const std::string& n) { return n == B; },
                                         [&r](const std::string& n, Tensor& a) {
                                             r.calls.push_back({n, a.shape});
                                             r.state.push_back(a.ptr()[0]);
                                         });
                 }});
    v.push_back(
        {"edits, and a capture of the last block", [=](RuntimeHookRun& r) {  // what it sees depends on the edits
             on(A, [=](Tensor& a, std::vector<float>& s) {
                 if (static_cast<int64_t>(count(s)) % 3 == 0) negate(a, 0, a.numel());
             })(r);
             r.hooks.add_capture([B](const std::string& n) { return n == B; },
                                 [&r](const std::string& n, Tensor& a) {
                                     r.calls.push_back({n, a.shape});
                                     r.state.insert(r.state.end(), a.ptr() + a.numel() - a.shape.back(),
                                                    a.ptr() + a.numel());
                                 });
         }});
    v.push_back({"several hook points", [=](RuntimeHookRun& r) {
                     on(A, [=](Tensor& a, std::vector<float>& s) {
                         if (static_cast<int64_t>(count(s)) % 4 == 0) negate(a, 0, a.numel());
                     })(r);
                     r.hooks.add(B, [&r](const std::string& n, Tensor& a) {
                         r.calls.push_back({n, a.shape});
                         for (float& x : a.data) x *= 0.75f;
                     });
                 }});
    return v;
}

// A hook that changes what it does once predictions trust it: the guesses in that pass hold the wrong edit, and
// the real hook's output must replace it. The model is its own draft, so proposals survive until the edit starts.
static void test_runtime_mispredictions(const Model& m) {
    Tokens prompt = seq(6, 81);
    prompt.insert(prompt.end(), prompt.begin(), prompt.end());
    GenerateOptions o;
    o.max_new_tokens = 40;
    if (m.cfg().positional == "learned")
        o.max_new_tokens = std::min<int64_t>(40, m.cfg().max_seq_len - std::ssize(prompt));
    o.stop_at_eos = false;
    const Tokens unhooked = generate(m, prompt, o);
    const std::string point = "blocks.0.hook_resid_post";
    int64_t reruns = 0;
    bool changed = false;
    for (int start : {9, 12, 15, 18, 23, 29})
        for (int stop_at : {0, 1, 5}) {  // the edit lasts, or stops again after 1 or 5 calls
            if (start >= o.max_new_tokens) continue;
            int calls = 0;
            std::vector<Shape> seen;
            Hooks h;
            h.add(point, [&](const std::string&, Tensor& a) {
                seen.push_back(a.shape);
                const int c = calls++;
                if (c >= start && (!stop_at || c < start + stop_at))
                    for (float& x : a.data) x = -x;
            });
            const Tokens want = generate(m, prompt, o, &h);
            const std::vector<Shape> want_calls = seen;
            changed = changed || want != unhooked;
            for (int64_t k : {2, 4, 8}) {
                GenerateOptions s = o;
                s.speculative.draft_tokens = k;
                s.speculative.draft = &m;
                SpeculativeStats st;
                s.speculative.stats = &st;
                calls = 0;
                seen.clear();
                CHECK(generate(m, prompt, s, &h) == want);
                CHECK(seen == want_calls);
                reruns += st.reruns;
            }
        }
    CHECK(changed);     // the edits matter
    CHECK(reruns > 0);  // and were mispredicted
}

// DecodeRuntime: whatever the proposals, hooks get exactly the calls of decoding without speculation, keep the
// same state, and the text is the same.
static void test_runtime_hooks(const Model& m, const Model& other) {
    Tokens prompt = seq(6, 81);
    prompt.insert(prompt.end(), prompt.begin(), prompt.end());
    GenerateOptions o;
    o.max_new_tokens = 14;
    o.stop_at_eos = false;
    for (const auto& [name, setup] : runtime_hook_setups(m)) {
        RuntimeHookRun base;
        setup(base);
        const Tokens want = generate(m, prompt, o, &base.hooks);
        CHECK(base.calls.size() >= 14);
        for (int64_t k : {1, 3, 8})
            for (const Model* d : {static_cast<const Model*>(nullptr), &other, &m}) {
                GenerateOptions s = o;
                s.speculative.draft_tokens = k;
                s.speculative.draft = d;
                RuntimeHookRun run;
                setup(run);
                CHECK(generate(m, prompt, s, &run.hooks) == want);
                CHECK(run.calls == base.calls);
                CHECK(run.state.size() == base.state.size());
                for (size_t i = 0; i < run.state.size() && i < base.state.size(); ++i)
                    CHECK_NEAR(run.state[i], base.state[i], 2e-2 * std::max(1.f, std::fabs(base.state[i])));
            }
    }
}

static void test_runtime_interface() {
    const Model m = random_model(tiny(), 42), other = random_model(tiny(), 43);
    const Tokens prompt = {5, 9, 13, 2, 7, 5, 9, 13};
    GenerateOptions o;
    o.max_new_tokens = 10;
    o.stop_at_eos = false;
    // Lazy: the prompt's call, then each token's call only when the token after it is asked for.
    for (int64_t k : {0, 4}) {
        GenerateOptions s = o;
        s.speculative.draft_tokens = k;
        s.speculative.draft = k ? &m : nullptr;
        int calls = 0;
        Hooks h;
        h.add("blocks.0.hook_resid_post", [&](const std::string&, Tensor&) { ++calls; });
        const Tokens want = generate(m, prompt, s, &h);
        CHECK(calls == 10);  // the last token ends the text and is never run
        calls = 0;
        DecodeRuntime rt(m, prompt, s, &h);
        CHECK(calls == 0);
        Tokens got;
        while (const auto t = rt.next()) {
            got.push_back(*t);
            CHECK(calls == static_cast<int>(got.size()));
        }
        CHECK(got == want && calls == 10);
        CHECK(!rt.next().has_value());
    }
    // A stop token is delivered and never reaches the hooks; proposals don't go past it.
    const Tokens plain = generate(m, prompt, o);
    size_t at = 1;
    while (at < plain.size() && std::find(plain.begin(), plain.begin() + static_cast<std::ptrdiff_t>(at), plain[at]) !=
                                    plain.begin() + static_cast<std::ptrdiff_t>(at))
        ++at;
    if (at < plain.size()) {
        GenerateOptions s = o;
        s.stop_tokens = {plain[at]};
        s.speculative.draft_tokens = 8;
        s.speculative.draft = &m;
        SpeculativeStats st;
        s.speculative.stats = &st;
        CHECK(generate(m, prompt, s) == Tokens(plain.begin(), plain.begin() + static_cast<std::ptrdiff_t>(at) + 1));
        CHECK(st.drafted <= static_cast<int64_t>(at));
    }
    // A failing hook ends the decoding.
    int n = 0;
    Hooks bad;
    bad.add("hook_final", [&](const std::string&, Tensor&) {
        if (++n == 3) throw std::runtime_error("hook failed");
    });
    GenerateOptions s = o;
    s.speculative.draft_tokens = 4;
    s.speculative.draft = &m;
    DecodeRuntime rt(m, prompt, s, &bad);
    bool threw = false;
    try {
        while (rt.next()) {}
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw && !rt.next().has_value());
    CHECK_THROWS(DecodeRuntime(m, Tokens{}, o));
    // Prompt lookup continues a repeating text past its end: after x y x y ... x it proposes y x y x ..., as many
    // as asked for, not only the token or two after the last occurrence.
    bool repeats = false;
    for (int64_t x = 0; x < 50 && !repeats; ++x)
        for (int64_t y = 0; y < 50 && !repeats; ++y) {
            Tokens xy;
            for (int i = 0; i < 4; ++i) xy.insert(xy.end(), {x, y});
            GenerateOptions g = o;
            g.max_new_tokens = 1;
            if (x == y || generate(m, xy, g) != Tokens{x}) continue;
            repeats = true;
            GenerateOptions s = o;
            s.max_new_tokens = 12;
            s.speculative.draft_tokens = 8;
            SpeculativeStats st;
            s.speculative.stats = &st;
            DecodeRuntime rt(m, xy, s);
            rt.next();
            rt.next();
            CHECK(st.drafted == 8);
        }
    CHECK(repeats);
    // Four blocks: an edit at the end of block 0 that is mispredicted means computing the token again from block
    // 1 (most of the model), so it is only speculated on once predictions have earned trust; at the end of block 2,
    // from block 3, so it always is.
    const Model m4 = random_model(tiny([](ModelConfig& c) { c.n_layers = 4; }), 42);
    const std::string low = "blocks.0.hook_resid_post", high = "blocks.2.hook_resid_post";
    // After a token is computed again from block 3, the guesses after it that its old values reached there are
    // computed again from there too, and their hooks from there on run only then: a capture of block 3 sees
    // what it sees without speculation (to rounding: float32 on the CPU), though the edit (now and then, small
    // enough for the draft's guesses to stay right) is predicted right on those tokens.
    {
        GenerateOptions p = o;
        p.max_new_tokens = 60;
        int c = 0;
        std::vector<float> seen;
        Hooks h;
        h.add(high, [&](const std::string&, Tensor& a) {
            const int k = c++;
            if (k == 9 || k == 20 || k == 26 || k == 37 || k == 45)
                for (int64_t r = 0; r < a.shape[0]; ++r) a.ptr()[r * a.shape[1]] += 0.05f;
        });
        h.add_capture([](const std::string& n) { return n == "blocks.3.hook_resid_post"; },
                      [&](const std::string&, Tensor& a) {
                          seen.insert(seen.end(), a.ptr() + a.numel() - a.shape[1], a.ptr() + a.numel());
                      });
        const Tokens text = generate(m4, prompt, p, &h);
        const std::vector<float> want = seen;
        c = 0;
        seen.clear();
        p.speculative.draft_tokens = 6;
        p.speculative.draft = &m4;
        SpeculativeStats st;
        p.speculative.stats = &st;
        CHECK(generate(m4, prompt, p, &h) == text);
        CHECK(seen.size() == want.size());
        float worst = 0;
        for (size_t i = 0; i < seen.size() && i < want.size(); ++i)
            worst = std::max(worst, std::fabs(seen[i] - want[i]));
        CHECK(worst < 1e-4f);
        CHECK(st.reruns > 0 && st.reruns < st.accepted && st.steps < 30);
    }
    // Hooks whose edits repeat keep speculating, including when they alternate: nearly nothing is recomputed.
    for (int period : {1, 2, 3, 12}) {
        GenerateOptions p = o;
        p.max_new_tokens = 96;
        int c = 0;
        Hooks small;  // edits small enough that the draft's proposals nearly always stay right
        small.add(low, [&](const std::string&, Tensor& a) {
            if (c++ % period == 0)
                for (float& x : a.data) x *= 1.0001f;
        });
        const Tokens text = generate(m4, prompt, p, &small);
        c = 0;
        p.speculative.draft_tokens = 4;
        p.speculative.draft = &m4;
        SpeculativeStats st;
        p.speculative.stats = &st;
        CHECK(generate(m4, prompt, p, &small) == text);
        CHECK(st.steps < 50 && st.reruns <= (period == 12 ? 4 : 2));
    }
    // Proposals that keep failing stop being made, and the draft stops running with them. The draft is the model
    // with its unembedding negated: it always proposes the token the model likes least.
    const ModelConfig untied = tiny([](ModelConfig& c) { c.tie_embeddings = false; });
    const Model base = random_model(untied, 42);
    std::map<std::string, Tensor> flipped;
    for (const auto& [name, w] : base.weights()) flipped[name] = w.to_tensor();
    for (float& x : flipped.at("unembed.weight").data) x = -x;
    const Model contrary(untied, flipped);
    GenerateOptions u = o;
    u.max_new_tokens = 60;
    const Tokens alone = generate(base, prompt, u);
    u.speculative.draft_tokens = 8;
    u.speculative.draft = &contrary;
    SpeculativeStats st;
    u.speculative.stats = &st;
    CHECK(generate(base, prompt, u) == alone);
    CHECK(st.accepted == 0 && st.drafted > 0 && st.drafted <= 30);
    // A hook whose edits can't be predicted: low in the model it isn't speculated on, so nothing is computed
    // twice; high, every confirmed guess is computed again from block 3, and passes still choose several tokens.
    std::mt19937 noise(1);
    for (const std::string& point : {low, high}) {
        Hooks random_edit;
        random_edit.add(point, [&](const std::string&, Tensor& a) {
            for (float& x : a.data) x += 0.05f * (static_cast<float>(noise() >> 8) * 0x1p-24f - 0.5f);
        });
        GenerateOptions r = o;
        r.max_new_tokens = 40;
        noise.seed(1);
        const Tokens noisy = generate(m4, prompt, r, &random_edit);
        noise.seed(1);
        r.speculative.draft_tokens = 8;
        r.speculative.draft = &m4;
        r.speculative.stats = &st;
        CHECK(generate(m4, prompt, r, &random_edit) == noisy);
        if (point == low) CHECK(st.drafted == 0 && st.reruns == 0);
        else CHECK(st.accepted > 20 && st.reruns == st.accepted && st.steps < 20);
    }
    // Edits from a small set in random order: each id is known but the next can't be predicted, so the predictions
    // lose trust instead of being recomputed over and over.
    std::mt19937 coin(2);
    Hooks random_choice;
    random_choice.add(low, [&](const std::string&, Tensor& a) {
        if (coin() & 256)
            for (float& x : a.data) x *= 1.0001f;
    });
    GenerateOptions c = o;
    c.max_new_tokens = 80;
    const Tokens chosen = generate(m4, prompt, c, &random_choice);
    coin.seed(2);
    c.speculative.draft_tokens = 8;
    c.speculative.draft = &m4;
    c.speculative.stats = &st;
    CHECK(generate(m4, prompt, c, &random_choice) == chosen);
    CHECK(st.reruns <= 8);

    // A hook's real edit is never taken for its prediction. hook_final zeroes every row (all logits equal, so greedy
    // takes id 0, and prompt lookup keeps proposing it), then writes 5e-10 instead, which breaks the tie: values
    // left alone or overwritten must match exactly.
    {
        GenerateOptions s = o;
        s.max_new_tokens = 16;
        const Tokens zeros(6, 0);
        int calls = 0;
        Hooks h;
        h.add("hook_final", [&](const std::string&, Tensor& a) {
            const bool nudge = ++calls > 8;
            for (float& v : a.data) v = nudge ? 5e-10f : 0.f;
        });
        const Tokens want = generate(m, zeros, s, &h);
        CHECK(want.front() == 0 && want.back() != 0);
        calls = 0;
        s.speculative.draft_tokens = 4;
        s.speculative.stats = &st;
        CHECK(generate(m, zeros, s, &h) == want);
        CHECK(st.accepted > 0);
    }
    // ... nor is an overwritten value that moves by one ulp: from the 9th call, every other call writes the float
    // after 1 instead of 1, and those tokens are computed again with it.
    {
        auto reruns = [&](bool nudge) {
            int calls = 0;
            Hooks h;
            h.add("blocks.1.hook_resid_post", [&](const std::string&, Tensor& a) {
                ++calls;
                a.data[0] = nudge && calls > 8 && calls % 2 ? std::nextafter(1.f, 2.f) : 1.f;
            });
            GenerateOptions g = o;
            g.max_new_tokens = 24;
            g.speculative.draft_tokens = 4;
            g.speculative.draft = &m;
            SpeculativeStats s2;
            g.speculative.stats = &s2;
            generate(m, prompt, g, &h);
            return s2.reruns;
        };
        CHECK(reruns(true) > reruns(false));
    }
    // Each token's hooks run on a copy of the caller's hooks, as each pass of ordinary decoding does: a mutable
    // lambda's count starts again at every token, and the caller's Hooks is left as it was.
    {
        GenerateOptions s = o;
        s.max_new_tokens = 12;
        Hooks h;
        h.add("blocks.1.hook_resid_post", [count = 0](const std::string&, Tensor& a) mutable {
            if (++count > 1)
                for (float& v : a.data) v = -v;
        });
        const Tokens want = generate(m, prompt, s, &h);
        s.speculative.draft_tokens = 4;
        s.speculative.draft = &m;
        CHECK(generate(m, prompt, s, &h) == want);
        CHECK(generate(m, prompt, s, &h) == want);  // the same Hooks again
        s.speculative = {};
        CHECK(generate(m, prompt, s, &h) == want);
    }
    // Filters are asked at every token, as each pass of ordinary decoding asks them: hook_final is negated only while
    // `on` is set, which the caller changes between tokens, next to a capture that is always on.
    {
        auto run = [&](int64_t k) {
            bool on = false;
            int finals = 0, embeds = 0;
            Hooks h;
            h.add_capture([](const std::string& n) { return n == "hook_embed"; },
                          [&](const std::string&, Tensor&) { ++embeds; });
            h.add_filter([&on](const std::string& n) { return on && n == "hook_final"; },
                         [&](const std::string&, Tensor& a) {
                             ++finals;
                             for (float& v : a.data) v = -v;
                         });
            GenerateOptions g = o;
            g.max_new_tokens = 12;
            g.speculative.draft_tokens = k;
            g.speculative.draft = k ? &m : nullptr;
            DecodeRuntime rt(m, prompt, g, &h);
            Tokens got;
            while (const auto t = rt.next()) {
                got.push_back(*t);
                const size_t i = got.size();
                on = i == 1 || i == 2 || i == 3 || i >= 7;
            }
            return std::make_tuple(got, finals, embeds);
        };
        const auto want = run(0);
        CHECK(std::get<1>(want) > 0);
        CHECK(run(4) == want);
    }
    // A mispredicted hook_final changes its row's logits only: the guesses after it in the pass still stand, and
    // only the tokens it got wrong are counted as computed again.
    {
        auto run = [&](int64_t k, SpeculativeStats* stats) {
            int c = 0;
            Hooks h;
            h.add("hook_final", [&](const std::string&, Tensor& a) {
                const int i = c++;
                if (i == 9 || i == 20 || i == 26 || i == 37 || i == 45)
                    for (int64_t r = 0; r < a.shape[0]; ++r) a.ptr()[r * a.shape[1]] += 0.05f;
            });
            GenerateOptions g = o;
            g.max_new_tokens = 60;
            g.speculative.draft_tokens = k;
            g.speculative.draft = k ? &m4 : nullptr;
            g.speculative.stats = stats;
            return generate(m4, prompt, g, &h);
        };
        SpeculativeStats s2;
        CHECK(run(6, &s2) == run(0, nullptr));
        CHECK(s2.reruns > 0 && s2.reruns <= 10 && s2.steps < 20);
    }
    // Sampler::greedy: whatever the logits, the most likely token.
    {
        GenerateOptions g;
        CHECK(Sampler(g).greedy());
        g.temperature = 0.8;
        CHECK(!Sampler(g).greedy());
        g.top_k = 1;
        CHECK(Sampler(g).greedy());
        g.top_k = 0;
        g.temperature = 1e-300;
        CHECK(Sampler(g).greedy());
    }
}

// A temperature whose reciprocal overflows a float: the largest logit alone, the lowest id among equal ones.
static void test_tiny_temperature() {
    const Model m = random_model(tiny(), 42), other = random_model(tiny(), 43);
    const Tokens prompt = {5, 9, 13, 5, 9, 13, 5, 9};
    const std::vector<float> logits = {1.f, 3.f, 2.f, 3.f};
    GenerateOptions o;
    o.max_new_tokens = 6;
    o.stop_at_eos = false;
    const Tokens greedy = generate(m, prompt, o);
    for (double temp : {2.9e-39, 1e-40, 1e-300, std::numeric_limits<double>::denorm_min()})
        for (double top_p : {1.0, 0.9}) {
            o.temperature = temp;
            o.top_p = top_p;
            std::vector<double> p;
            Sampler(o).distribution(std::span(logits.data(), 4), p);
            CHECK(p == (std::vector<double>{0, 1, 0, 0}));
            o.speculative.draft_tokens = 0;
            CHECK(generate(m, prompt, o) == greedy);
            o.speculative.draft_tokens = 3;
            for (const Model* d : {static_cast<const Model*>(nullptr), &other}) {
                o.speculative.draft = d;
                CHECK(generate(m, prompt, o) == greedy);
            }
            o.speculative.draft = nullptr;
        }
    // The smallest scale that still fits a float samples as usual: equal logits, equal chances.
    o = GenerateOptions{};
    o.temperature = 3e-39;
    std::vector<double> p;
    Sampler(o).distribution(std::span(logits.data(), 4), p);
    CHECK(p == (std::vector<double>{0, 0.5, 0, 0.5}));
}

// The sampler alone: frequencies follow softmax(logits / temperature), restricted by top_k and top_p;
// NaN and -inf are never picked.
static void test_sampler() {
    const std::vector<float> logits = {1.f, 2.f, 0.f, -INFINITY, std::nanf(""), 1.5f};
    auto freqs = [&](GenerateOptions o, int n) {
        Sampler s(o);
        std::vector<double> f(logits.size(), 0);
        for (int i = 0; i < n; ++i) f[s(logits)] += 1.0 / n;
        return f;
    };
    auto dist = [&](GenerateOptions o) {  // what speculative decoding's acceptance rule uses
        std::vector<double> d;
        Sampler(o).distribution(logits, d);
        return d;
    };
    auto softmax = [&](std::vector<int> keep, double temp) {
        std::vector<double> p(logits.size(), 0);
        double z = 0;
        for (int i : keep) z += std::exp(logits[i] / temp);
        for (int i : keep) p[i] = std::exp(logits[i] / temp) / z;
        return p;
    };
    const int n = 200000;
    GenerateOptions o;
    o.seed = 3;
    for (double temp : {1.0, 0.5}) {
        o.temperature = temp;
        const std::vector<double> f = freqs(o, n), p = softmax({0, 1, 2, 5}, temp);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
        const std::vector<double> d = dist(o);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(d[i], p[i], 1e-6);  // float weights
    }
    o.temperature = 1.0;
    o.top_k = 2;  // tokens 1 and 5
    {
        const std::vector<double> f = freqs(o, n), p = softmax({1, 5}, 1.0);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
        const std::vector<double> d = dist(o);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(d[i], p[i], 1e-6);  // float weights
    }
    o.top_k = 0;
    o.top_p = 0.6;  // token 1 has 0.47 of the mass, with token 5 0.76 >= 0.6
    {
        const std::vector<double> f = freqs(o, n), p = softmax({1, 5}, 1.0);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
        const std::vector<double> d = dist(o);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(d[i], p[i], 1e-6);  // float weights
    }
    o.top_p = 0.3;  // token 1 alone
    CHECK_NEAR(freqs(o, 1000)[1], 1.0, 1e-9);
    o.top_p = 1;
    o.temperature = 0;
    CHECK_NEAR(freqs(o, 10)[1], 1.0, 1e-9);
    CHECK((dist(o) == std::vector<double>{0, 1, 0, 0, 0, 0}));  // greedy: one-hot
    {  // draw: index i with probability w[i] / sum(w), never a zero weight
        Sampler s(o);
        const std::vector<double> w = {0.5, 0, 1.5, 2};
        std::vector<double> f(4, 0);
        for (int i = 0; i < n; ++i) f[static_cast<size_t>(s.draw(w))] += 1.0 / n;
        for (size_t i = 0; i < 4; ++i) CHECK_NEAR(f[i], w[i] / 4, 0.005);
        CHECK_THROWS(s.draw({0, 0}));
    }
    const std::vector<float> tie = {3.f, 5.f, 5.f};
    Sampler greedy(o);
    CHECK(greedy(tie) == 1);  // the lowest id among equal logits
    const std::vector<float> none = {-INFINITY, std::nanf("")};
    CHECK_THROWS(greedy(none));

    // The sampler finds top_k and the nucleus without sorting: it picks exactly what sorting every token
    // would, on a vocabulary-sized row (peaked and flat), pick for pick from the same seed. The rule: the k
    // largest logits (lower id first among equal ones), then the fewest most likely of those whose weights
    // reach top_p of their total, then a draw over the kept tokens in id order.
    std::mt19937_64 gen(9);
    for (const float spread : {3.f, 0.3f}) {
        std::normal_distribution<float> normal(0.f, spread);
        std::vector<float> row(20000);
        for (float& v : row) v = normal(gen);
        for (int64_t i = 0; i < 20000; i += 97)
            row[static_cast<size_t>(i)] = row[static_cast<size_t>(i + 1)];  // equal logits
        row[5] = -INFINITY;
        row[6] = std::nanf("");
        const float top = *std::ranges::max_element(
            row, [](float x, float y) { return std::isnan(y) ? false : std::isnan(x) || x < y; });
        for (const double top_p : {0.5, 0.9, 0.99, 1.0})
            for (const int64_t top_k : {int64_t(0), int64_t(50), int64_t(5000)})
                for (const double temp : {0.7, 1.5}) {
                    GenerateOptions so;
                    so.temperature = temp;
                    so.top_p = top_p;
                    so.top_k = top_k;
                    so.seed = 11;
                    Sampler fast(so);
                    std::mt19937_64 rng(11);
                    std::vector<int64_t> by_logit;  // usable tokens, largest logit first
                    for (int64_t i = 0; i < 20000; ++i)
                        if (!std::isnan(row[static_cast<size_t>(i)]) && row[static_cast<size_t>(i)] != -INFINITY)
                            by_logit.push_back(i);
                    std::ranges::stable_sort(by_logit, std::ranges::greater{},
                                             [&](int64_t i) { return row[static_cast<size_t>(i)]; });
                    if (top_k && static_cast<size_t>(top_k) < by_logit.size())
                        by_logit.resize(static_cast<size_t>(top_k));
                    const float scale = static_cast<float>(1.0 / temp);
                    std::vector<std::pair<float, int64_t>> c;  // (weight, id), most likely first
                    double mass = 0;
                    for (int64_t i : by_logit) {
                        const float d = row[static_cast<size_t>(i)] - top;
                        c.push_back({detail::exp_weight(d * scale), i});
                        mass += c.back().first;
                    }
                    std::ranges::stable_sort(c, std::ranges::greater{}, [](const auto& x) { return x.first; });
                    size_t n = c.size();
                    if (top_p < 1) {
                        double sum = 0;
                        for (n = 0; n < c.size();) {
                            sum += c[n++].first;
                            if (sum >= top_p * mass) break;
                        }
                    }
                    std::vector<std::pair<int64_t, float>> kept;  // id order
                    for (size_t i = 0; i < n; ++i) kept.push_back({c[i].second, c[i].first});
                    std::ranges::sort(kept);
                    for (int pick = 0; pick < 100; ++pick) {
                        double total = 0;
                        for (const auto& [id, weight] : kept) total += weight;
                        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * total;
                        int64_t want = kept.back().first;
                        double acc = 0;
                        for (const auto& [id, weight] : kept) {
                            acc += weight;
                            if (u < acc) {
                                want = id;
                                break;
                            }
                        }
                        CHECK(fast(row) == want);
                    }
                }
    }
    // Cuts at equal keys and at a bucket's edge: which tokens distribution() keeps.
    auto kept_ids = [](const std::vector<float>& lg, int64_t k, double p) {
        GenerateOptions so;
        so.temperature = 1;
        so.top_k = k;
        so.top_p = p;
        std::vector<double> d;
        Sampler(so).distribution(lg, d);
        std::vector<int64_t> ids;
        for (size_t i = 0; i < d.size(); ++i)
            if (d[i] > 0) ids.push_back(static_cast<int64_t>(i));
        return ids;
    };
    const std::vector<float> flat(10, 1.f);  // equal logits: lower ids first
    CHECK((kept_ids(flat, 3, 1) == std::vector<int64_t>{0, 1, 2}));
    CHECK((kept_ids(flat, 0, 0.25) == std::vector<int64_t>{0, 1, 2}));  // 0.3 is the first total reaching 0.25
    CHECK((kept_ids(flat, 6, 0.5) == std::vector<int64_t>{0, 1, 2}));   // half of the top 6
    CHECK((kept_ids({-0.f, 0.f, 0.f, -1.f}, 2, 1) == std::vector<int64_t>{0, 1}));  // -0 = +0
    // Each logit in a bucket of its own: the count is reached exactly at a bucket's end.
    CHECK((kept_ids({1.f, 16.f, 2.f, 8.f, 4.f}, 2, 1) == std::vector<int64_t>{1, 3}));
    CHECK((kept_ids({1.f, 16.f, 2.f, 8.f, 4.f}, 3, 1) == std::vector<int64_t>{1, 3, 4}));
    // exp_weight = exp to fp32 precision over its range, 0 below -87, for NaN and -inf; exp_weights (vector
    // and scalar parts) gives exp_weight's bits exactly.
    float worst = 0;
    for (float x = -87.f; x <= 0.f; x += 0.0137f) {
        const double want = std::exp(static_cast<double>(x));
        worst = std::max(worst, static_cast<float>(std::fabs(detail::exp_weight(x) - want) / want));
    }
    CHECK(worst < 3e-7f);
    CHECK(detail::exp_weight(0.f) == 1.f && detail::exp_weight(-0.f) == 1.f);
    CHECK(detail::exp_weight(-87.5f) == 0.f && detail::exp_weight(-INFINITY) == 0.f &&
          detail::exp_weight(std::nanf("")) == 0.f);
    std::vector<float> xs = {0.f,       -1e-30f,       -0.5f,  -87.f, -86.99f, -87.01f,
                             -INFINITY, std::nanf(""), -3.25f, -10.f, -50.f};
    for (int i = 0; i < 1000; ++i) xs.push_back(-90.f * static_cast<float>(i) / 1000.f);
    std::vector<float> ws(xs.size());
    detail::exp_weights(xs, 0.f, 1.f, ws);
    for (size_t i = 0; i < xs.size(); ++i) {
        const float d = xs[i] - 0.f;
        CHECK(ws[i] == detail::exp_weight(d * 1.f));
    }
    // The same picks on every platform: weights and draws use only exactly specified arithmetic (the logits
    // here come from the generator's raw output, not a library distribution).
    {
        std::mt19937_64 g(21);
        std::vector<float> lg(1000);
        for (float& v : lg) v = static_cast<float>(static_cast<double>(g() >> 40) * 0x1.0p-24 * 12.0 - 6.0);
        GenerateOptions go;
        go.temperature = 0.9;
        go.top_k = 200;
        go.top_p = 0.95;
        go.seed = 4;
        Sampler gs(go);
        std::vector<int64_t> picks;
        for (int i = 0; i < 12; ++i) picks.push_back(gs(std::span(lg.data(), 1000)));
        CHECK((picks == std::vector<int64_t>{762, 411, 537, 56, 470, 52, 831, 37, 704, 751, 618, 408}));
    }
}

#ifdef TFI_USE_ACCELERATE
// forward_cached and generate on the GPU (macOS): GPU-only mode, where every cached step runs there,
// and both mode with chunks of 2 rows or more on the GPU.
static void test_gpu_forward_cached(const Model& m) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    set_device(Device::Gpu);
    test_forward_cached(m);
    test_forward_cached_batch(m);
    test_generate(m);
    test_generate_trace(m);
    test_generate_batch(m);
    test_generate_speculative(m, random_model(m.cfg(), 43));  // steps of several rows, rolled back on the GPU
    test_runtime_hooks(m, random_model(m.cfg(), 43));
    test_runtime_mispredictions(m);
    set_device(Device::Both);
    detail::set_gpu_min_rows(2);
    test_forward_cached(m);
    test_forward_cached_batch(m);
    test_runtime_mispredictions(m);  // passes of several tokens on the GPU, single tokens computed again on the CPU
    detail::set_gpu_min_rows(0);
}
#endif

#ifdef TFI_USE_ACCELERATE
// Gpu::Seq::run commits its work in several command buffers (every few linear layers) so the GPU starts
// while the rest is encoded: the same kernels in the same order, so bit for bit what one command buffer
// gives, for cached decoding, a batch, batched patching passes and gradient passes.
static void test_gpu_chunked_commits(const Model& m) {
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    auto everything = [&] {
        std::vector<Tensor> out;
        set_device(Device::Gpu);
        KVCache c;
        out.push_back(m.forward_cached(seq(9, 64), c));
        for (int64_t t : {3, 7, 1}) out.push_back(m.forward_cached({t}, c));
        KVBatch b(2);
        for (const Tensor& t : m.forward_cached({seq(5, 65), seq(3, 66)}, b)) out.push_back(t);
        for (const Tensor& t : m.forward_cached({Tokens{4}, Tokens{9}}, b)) out.push_back(t);
        const Tokens clean = seq(8, 67), corrupt = seq(8, 68);
        out.push_back(patch_residual(m, clean, corrupt, "resid_pre", 7, 1, 2).effect);
        out.push_back(attribution_patch_residual(m, clean, corrupt, "resid_pre", 7, 1, 2).effect);
        set_device(Device::Both);
        return out;
    };
    detail::set_commit_linears(1);  // a command buffer per linear layer
    const std::vector<Tensor> chunked = everything();
    detail::set_commit_linears(int64_t(1) << 30);  // one command buffer
    const std::vector<Tensor> whole = everything();
    detail::set_commit_linears(0);
    CHECK(chunked.size() == whole.size());
    for (size_t i = 0; i < chunked.size() && i < whole.size(); ++i) CHECK(same_values(chunked[i], whole[i]));
}

// Random sequences of operations on KV caches whose positions live on the CPU, the GPU or both (GPU steps
// append on the GPU only): GPU and CPU steps in chunks, truncation, copies, keys(), clearing; single caches
// and batches. After every step the logits = the whole pass on the same history, keys() = its hook_k, and
// (set_check_kv) each cache's CPU and GPU copies agree wherever both are up to date.
static void test_kv_random(const Model& m, uint64_t seed) {
    const HalfTolerance half_tolerance(m);  // fp16 weights: the GPU rounds its multiplies' inputs
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    detail::set_check_kv(true);
    std::mt19937_64 rng(seed);
    auto pick = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };
    const int64_t V = m.cfg().vocab_size, L = m.cfg().n_layers;
    // fp16 weights: the cache holds fp16 keys and values, so the reference rounds them too. Over long chunks
    // many keys sit near an fp16 rounding boundary, and the GPU (whose keys differ from the CPU's in the last
    // float32 bits) rounds some to the neighbouring fp16 value: ~1e-3 of a key, which a squared activation
    // (glu relu2) amplifies to up to ~8e-3 of the logits' scale; with every step on the CPU nothing passes
    // 1e-3. A cache holding the wrong positions is off by the size of the values; test_fp16_kv checks the
    // rounding point exactly (clamped keys), and set_check_kv the CPU and GPU copies bit for bit.
    const bool half = m.weight_format() == WeightFormat::fp16;
    const Hooks round = round_kv_to_half(m);
    // Its GPU multiplies also take fp16 inputs (~1e-3 relative each): over long chunks of these random models that tips
    // near-ties in attention, up to ~6% of the logits' scale (measured). Still far from a cache holding the wrong
    // positions (off by the whole scale); set_check_kv checks the CPU and GPU copies bit for bit.
    const double rel = half ? 1e-1 : 1e-4, of_scale = half ? 1e-1 : 1e-4;
    auto reference = [&](const Tokens& t) {  // the whole pass on the CPU: logits and every layer's keys
        set_device(Device::Cpu);
        ActivationCache acts;
        Tensor lg = m.run_with_cache(t, acts, nullptr, half ? &round : nullptr);
        return std::make_pair(lg, acts);
    };
    // GPU against CPU, as check_close but with 1e-4 of the largest value: models without norms reach
    // logits in the thousands over long sequences, where the two add in different orders (the first step,
    // with no cache yet, already differs by ~1e-5 of the scale; long chunks by up to ~5e-5). A cache holding
    // the wrong positions is off by the size of the values themselves.
    auto near = [&](const Tensor& got, const Tensor& want) {
        CHECK(got.shape == want.shape);
        const float sc = max_abs(want);
        for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
            CHECK_NEAR(got.data[i], want.data[i], rel * std::max(1.f, std::fabs(want.data[i])) + of_scale * sc);
    };
    auto step = [&](const Tokens& history, int64_t n, bool gpu, auto&& run) {
        Tokens add(static_cast<size_t>(n));
        for (auto& x : add) x = pick(0, V - 1);
        set_device(gpu ? Device::Gpu : Device::Cpu);
        const Tensor got = run(add);
        Tokens full = history;
        full.insert(full.end(), add.begin(), add.end());
        const Tensor want = reference(full).first;
        near(got, rows(want, std::ssize(history), std::ssize(full)));
        return full;
    };
    // Histories stay under learned positions' limit (a step adds at most 4, a batch step 3).
    int64_t limit = m.cfg().positional == "learned" ? m.cfg().max_seq_len - 4 : int64_t(1) << 30;
    for (int64_t l = 0; l < L; ++l) {  // RoPE scalings whose frequencies change past original_max_position_embeddings
        const auto& rs = m.layer_cfg(l).rope_scaling;
        if (rs && (rs->type == "longrope" || rs->type == "dynamic"))
            limit = std::min(limit, rs->original_max_position_embeddings - 4);
    }
    // One cache.
    const Model twin = m;  // the same model, another GPU context
    KVCache cache;
    Tokens history;
    for (int op = 0; op < 40; ++op) {
        if (std::ssize(history) > limit) {
            cache.truncate(limit / 2);
            history.resize(static_cast<size_t>(limit / 2));
        }
        const int64_t kind = history.empty() ? 0 : pick(0, 7);
        if (kind <= 2) {  // a step of 1-4 tokens, on the GPU (twice as often) or the CPU
            history =
                step(history, pick(1, 4), kind != 2, [&](const Tokens& add) { return m.forward_cached(add, cache); });
        } else if (kind == 7) {  // a long chunk on the GPU: its buffers grow while it may hold the only copy of recent
                                 // positions
            if (limit > 1000)
                history = step(history, pick(150, 300), true,
                               [&](const Tokens& add) { return m.forward_cached(add, cache); });
        } else if (kind == 3) {
            const int64_t n = pick(0, std::ssize(history));
            cache.truncate(n);
            history.resize(static_cast<size_t>(n));
        } else if (kind == 4) {  // keys() brings the CPU side up to date
            const auto acts = reference(history).second;
            const int64_t l = pick(0, L - 1);
            near(cache.keys(l), acts.at(std::format("blocks.{}.attn.hook_k", l)));
            near(cache.values(l), acts.at(std::format("blocks.{}.attn.hook_v", l)));
        } else if (kind == 5) {  // continue on a copy (it has no GPU copy of its own yet)
            cache = KVCache(cache);
        } else if (pick(0, 1) ==
                   0) {  // a step through a copy of the model, with a GPU context of its own: a new GPU copy
            history =
                step(history, pick(1, 4), true, [&](const Tokens& add) { return twin.forward_cached(add, cache); });
        } else if (pick(0, 3) == 0) {
            cache.clear();
            history.clear();
        }
        CHECK(cache.size() == std::ssize(history));
    }
    // A batch of three: random subsets of sequences, each a random chunk, on the GPU or the CPU.
    KVBatch batch(3);
    std::vector<Tokens> hist(3);
    for (int op = 0; op < 25; ++op) {
        bool full_up = false;
        for (const Tokens& t : hist) full_up = full_up || std::ssize(t) > limit;
        if (full_up) {  // a batch can't be truncated: start a new one
            batch = KVBatch(3);
            for (Tokens& t : hist) t.clear();
        }
        const int64_t kind = pick(0, 6);
        if (kind <= 3 || kind == 6) {
            const bool gpu = kind != 3;
            const bool longer = kind == 6 && limit > 1000;  // one sequence's long chunk: every region grows
            std::vector<Tokens> add(3);
            for (int i = 0; i < 3; ++i)
                if (pick(0, 3) != 0 || (longer && i == 0))
                    for (int64_t k = longer && i == 0 ? pick(150, 300) : pick(1, 3); k > 0; --k)
                        add[i].push_back(pick(0, V - 1));
            set_device(gpu ? Device::Gpu : Device::Cpu);
            const std::vector<Tensor> got = m.forward_cached(add, batch);
            for (int i = 0; i < 3; ++i) {
                if (add[i].empty()) continue;
                Tokens full = hist[i];
                full.insert(full.end(), add[i].begin(), add[i].end());
                near(got[i], rows(reference(full).first, std::ssize(hist[i]), std::ssize(full)));
                hist[i] = full;
            }
        } else if (kind == 4) {
            const int i = static_cast<int>(pick(0, 2));
            if (!hist[i].empty()) {
                const auto acts = reference(hist[i]).second;
                near(batch[i].keys(L - 1), acts.at(std::format("blocks.{}.attn.hook_k", L - 1)));
            }
        } else {
            batch = KVBatch(batch);  // a copy: keys and values on the CPU, no GPU copy
        }
        for (int i = 0; i < 3; ++i) CHECK(batch[i].size() == std::ssize(hist[i]));
    }
    set_device(Device::Both);
    detail::set_check_kv(false);
}

// GPU cached passes with fp16 multiplies are one submission for every layer; when an input overflows
// fp16 the whole run repeats in fp32, so it must find its inputs intact (each layer writes its own
// residual slot). Embeddings scaled past fp16's range on a model without norms (the linear layers read
// the residual stream itself) force that repeat: the result = the fp32 pass. And without overflow,
// fp16 stays close to fp32.
static void test_gpu_fp16_cached_repeat() {
    const Model m = random_model(tiny([](ModelConfig& c) {
                                     c.mlp = "none";
                                     c.norm = "none";
                                 }),
                                 61, WeightFormat::fp16);
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens t = seq(14, 62);
    for (const float scale : {1e5f, 1.f}) {
        Hooks big;
        big.add("hook_embed", [scale](const std::string&, Tensor& a) {
            for (float& v : a.data) v *= scale;
        });
        auto run =
            [&](Device dev) {  // fp16 weights: the GPU's multiplies take fp16 inputs; the CPU's arithmetic is float32
                set_device(dev);
                KVCache cache;
                std::vector<Tensor> parts{m.forward_cached(Tokens(t.begin(), t.begin() + 8), cache, &big)};
                for (int64_t i = 8; i < 14; ++i) parts.push_back(m.forward_cached({t[i]}, cache, &big));
                set_device(Device::Both);
                return stack_rows(parts);
            };
        const Tensor cpu = run(Device::Cpu), gpu = run(Device::Gpu);
        if (scale > 1) {
            check_close(gpu, cpu);  // every GPU run overflowed and repeated with float32 inputs
        } else {
            CHECK(gpu.shape == cpu.shape);
            CHECK(max_abs_diff(gpu, cpu) <= 3e-2f * std::max(1.f, max_abs(cpu)));
            CHECK(max_abs_diff(gpu, cpu) > 0.f);  // really fp16 inputs
        }
    }
}

// A long cached sequence on the GPU (macOS): steps of 1-8 tokens over ~600 earlier positions use
// split-key attention (several chunks of keys merged), longer chunks the usual kernels; all = the CPU's
// whole pass, with windows, softcaps, ALiBi, GQA and head sizes of every kernel.
static void test_gpu_forward_cached_long() {
    for (auto [name, c] : long_attention_configs()) {
        if (c.positional == "learned") continue;  // positions past max_seq_len
        c.max_seq_len = 1024;
        const Model m = random_model(c, 58);
        if (!m.gpu()) {
            std::printf("     (no Metal device: GPU path not tested)\n");
            return;
        }
        const Tokens t = seq(620, 59);
        set_device(Device::Cpu);
        const Tensor full = m.forward(t);
        for (const Device dev : {Device::Gpu, Device::Both}) {
            set_device(dev);
            detail::set_gpu_min_rows(2);  // both mode: chunks of 2+ rows on the GPU, single tokens on the CPU
            KVCache cache;
            std::vector<Tensor> parts{m.forward_cached(Tokens(t.begin(), t.begin() + 560), cache)};
            int64_t at = 560;
            for (const int64_t n : {1, 1, 3, 8, 1, 40, 1, 5}) {
                parts.push_back(m.forward_cached(Tokens(t.begin() + at, t.begin() + at + n), cache));
                at += n;
            }
            check_close(stack_rows(parts), rows(full, 0, at));
            detail::set_gpu_min_rows(0);
        }
        set_device(Device::Both);
    }
}
#endif

static const char* const kGroups[] = {"core", "analysis", "grad", "fp16", "gpu"};

static int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--group G[,G...]] [--filter TEXT] [--list]\n"
                 "  --group   run only these groups: core, analysis, grad, fp16, gpu\n"
                 "  --filter  run only tests whose name contains TEXT\n"
                 "  --list    print the selected tests' groups and names, run nothing\n",
                 argv0);
    return 2;
}

int main(int argc, char** argv) {
    std::vector<std::string> only;
    std::string filter;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--group" || a == "--filter") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (a == "--filter") {
                filter = v;
                continue;
            }
            for (size_t p = 0; p <= v.size();) {
                const size_t q = std::min(v.find(',', p), v.size());
                const std::string g = v.substr(p, q - p);
                if (std::find_if(std::begin(kGroups), std::end(kGroups), [&](const char* k) { return g == k; }) ==
                    std::end(kGroups)) {
                    std::fprintf(stderr, "unknown group '%s'\n", g.c_str());
                    return usage(argv[0]);
                }
                only.push_back(g);
                p = q + 1;
            }
        } else if (a == "--list") {
            list = true;
        } else {
            return usage(argv[0]);
        }
    }
    int selected = 0;
    auto run = [&](const std::string& group, const std::string& name, const std::function<void()>& fn) {
        if ((!only.empty() && std::ranges::find(only, group) == only.end()) || name.find(filter) == std::string::npos)
            return;
        ++selected;
        if (list) {
            std::printf("%-9s %s\n", group.c_str(), name.c_str());
            return;
        }
        const int before = g_failures;
        try {
            fn();
        } catch (const std::exception& e) {
            ++g_failures;
            std::printf("  EXCEPTION: %s\n", e.what());
        }
        std::printf("%s %s\n", g_failures == before ? "ok  " : "FAIL", name.c_str());
    };
    run("core", "json", test_json);
    run("core", "reshape + linear", test_reshape_and_linear);
    run("core", "matmul", test_matmul);
    for (const auto& [name, cfg] : configs()) {
        const Model m = random_model(cfg, 42);
        run("analysis", "cache/hooks/causality: " + name, [&] { test_cache_and_hooks(name, m); });
        run("analysis", "logit lens: " + name, [&] { test_logit_lens(m); });
        run("analysis", "direct logit attribution: " + name, [&] { test_dla(m); });
        run("analysis", "activation patching: " + name, [&] { test_patching(m); });
        run("analysis", "patching edge cases (zero gap, missing sites): " + name, [&] { test_patching_edge_cases(m); });
        run("analysis", "optimized patching (batched, GPU) and ablation = naive: " + name,
            [&] { test_optimized_patching_matches_naive(m); });
        run("analysis", "ablation + head scores: " + name, [&] { test_ablation_and_scores(m); });
        run("analysis", "mean ablation has no future leak: " + name, [&] { test_mean_ablation(m); });
        run("grad", "gradients = finite differences: " + name, [&] { test_gradients(m); });
        run("grad", "attribution patching: " + name, [&] { test_attribution_patching(m); });
        run("grad", "mixed patching: " + name, [&] { test_mixed_patching(m); });
        if (cfg.kv_heads() == cfg.n_heads)
            run("analysis", "composition = direct: " + name, [&] { test_composition_matches_direct(m); });
        run("analysis", "circuits: factored, parallel = direct: " + name, [&] { test_circuits_fast_paths(m); });
        run("analysis", "KV cache: forward_cached = forward, hooks, copies: " + name, [&] { test_forward_cached(m); });
        run("analysis", "KV batch: several sequences in one pass = each alone: " + name,
            [&] { test_forward_cached_batch(m); });
        run("analysis", "generate_batch = generate on each, per-sequence hooks, traces: " + name,
            [&] { test_generate_batch(m); });
        run("analysis", "generate: greedy = argmax, stops, hooks, sampling: " + name, [&] { test_generate(m); });
        run("analysis", "generate's trace = full-pass analyses at each step: " + name, [&] { test_generate_trace(m); });
        run("analysis", "speculative decoding: greedy = plain greedy, hooks, trace, limits: " + name,
            [&] { test_generate_speculative(m, random_model(cfg, 43)); });
        run("analysis", "decode runtime: hooks see what they see without speculation: " + name,
            [&] { test_runtime_hooks(m, random_model(cfg, 43)); });
        run("analysis", "decode runtime: a wrong prediction is replaced by the real hook: " + name,
            [&] { test_runtime_mispredictions(m); });
    }
    run("analysis", "KV cache: 12 layers (two-digit hook names), float32 and fp16, clamped values",
        test_forward_cached_many_layers);
    run("core", "sampler: temperature, top_k, top_p", test_sampler);
    run("core", "sampler: temperature too small for a float's 1 / T", test_tiny_temperature);
    run("analysis", "speculative sampling = the model's distribution", test_speculative_sampling);
    run("analysis", "speculative decoding: draft with a smaller vocabulary", test_speculative_small_draft);
    run("analysis", "decode runtime: one token at a time, stops, failures, prediction, backoff",
        test_runtime_interface);
    run("fp16", "linear layers on 1-16 rows: direct from the weights, fp16 = rounded float32, rows independent",
        test_linear_few_rows);
    run("fp16", "fp16 conversion", test_half_conversion);
    run("fp16", "fp16 weights: reads, linear, matmul", test_fp16_weight);
    run("fp16", "fp16 weights: loading converts as it reads", test_fp16_load);
    for (const auto& [name, cfg] : configs()) {
        const Model m16 = random_model(cfg, 42, WeightFormat::fp16);
        run("fp16", "fp16 weights = float32 with rounded weights: " + name, [&] { test_fp16_model(m16); });
        run("fp16", "fp16 weights: KV cache in fp16 = whole pass with rounded keys and values: " + name, [&] {
            set_device(Device::Cpu);
            test_fp16_kv(m16, cfg);
            set_device(Device::Both);
        });
        run("fp16", "fp16 weights: lens, DLA, patching, optimized = naive: " + name, [&] {
            test_logit_lens(m16);
            test_dla(m16);
            test_patching(m16);
            test_optimized_patching_matches_naive(m16);
        });
        run("fp16", "fp16 weights: gradients, attribution, mixed, circuits: " + name, [&] {
            test_gradients(m16);
            test_attribution_patching(m16);
            test_mixed_patching(m16);
            test_circuits_fast_paths(m16);
        });
    }
    run("core", "weight shapes checked when a model is built", test_weight_shapes_checked);
    run("core", "bias parts: parsed, exactly the named biases, qkv-only models", test_bias_parts);
    run("core", "tokenizer: UTF-8, NFC, split regex, BPE rules", test_tokenizer_parts);
    run("analysis", "OV copying + positional QK", test_copying_and_positional);
    run("analysis", "post-norm DLA refused", test_post_norm_dla_refused);
    run("core", "vector math", test_vector_math);
    run("core", "safetensors: shapes that don't fit the file", test_safetensors_bad_shapes);
    run("core", "RoPE scaling: config checks, length limit", test_rope_scaling);
    run("core", "config: d_mlp 0 is the default width", test_d_mlp_zero_is_default);
    run("gpu", "chunked GPU work: packing, retries, failure", test_gpu_chunks);
    run("gpu", "which single passes go to the GPU", test_single_pass_rule);
#ifdef TFI_USE_ACCELERATE
    run("gpu", "GPU linear layer (macOS)", test_gpu_linear);
    run("gpu", "memory-mapped weights on the GPU = CPU (macOS)", test_gpu_mapped_weights);
    run("gpu", "heads too wide for GPU attention stay on the CPU in both mode (macOS)", test_gpu_wide_heads);
    run("gpu", "GPU used only with enough work (macOS)", test_gpu_work_threshold);
    run("gpu", "device modes: cpu, both, gpu (macOS)", test_device_modes);
    run("gpu", "GPU batches: multiplies, block sums, argmaxes (macOS)", test_gpu_batch);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "circuits on the GPU = CPU, chunked, retried, falling back (macOS): " + name,
            [&] { test_gpu_circuits(random_model(cfg, 46)); });
    run("gpu", "GPU block operations (macOS)", test_gpu_seq);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "GPU-resident batched passes = per-layer and plain (macOS): " + name,
            [&] { test_gpu_resident(random_model(cfg, 43)); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "attention on the GPU: batched passes = per-layer and plain (macOS): " + name, [&] {
            detail::set_gpu_attention(true);
            const Model pm = random_model(cfg, 47);
            for (bool tiled : {true, false}) {
                detail::set_gpu_attention_tiled(tiled);
                test_gpu_resident(random_model(cfg, 43));
                test_optimized_patching_matches_naive(pm, /*cpu_parts=*/tiled);  // the CPU's parts once
            }
            detail::set_gpu_attention_tiled(true);
            detail::set_gpu_attention(false);
        });
    run("gpu", "attention on the GPU: long sequences, both kernels = CPU (macOS)", test_gpu_attention_long);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu",
            "attention + its backward on the GPU: batched gradients = run_with_gradients, attribution = CPU (macOS): " +
                name,
            [&] {
                detail::set_gpu_attention(true);
                for (bool tiled : {true, false}) {
                    detail::set_gpu_attention_tiled(tiled);
                    test_gpu_gradients(random_model(cfg, 44));
                }
                detail::set_gpu_attention_tiled(true);
                test_gpu_attribution(random_model(cfg, 45));
                detail::set_gpu_attention(false);
            });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "single passes on the GPU in GPU-only mode = CPU, hooks included (macOS): " + name,
            [&] { test_gpu_single_pass(random_model(cfg, 53), 9); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "capture hooks on the GPU next to editing hooks = CPU (macOS): " + name,
            [&] { test_gpu_captures(random_model(cfg, 56), 9); });
    run("gpu", "single passes in both mode with the default thresholds (macOS)", test_gpu_single_pass_defaults);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "KV cache and generate on the GPU = forward (macOS): " + name,
            [&] { test_gpu_forward_cached(random_model(cfg, 57)); });
    run("gpu", "KV cache on the GPU: long sequences, split-key attention = CPU (macOS)", test_gpu_forward_cached_long);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "KV cache in fp16 on the GPU = whole pass with rounded keys and values (macOS): " + name, [&] {
            const Model m16 = random_model(cfg, 42, WeightFormat::fp16);
            if (!m16.gpu()) return;
            for (const Device dev : {Device::Gpu, Device::Both}) {
                set_device(dev);
                detail::set_gpu_min_rows(2);
                test_fp16_kv(m16, cfg);
                detail::set_gpu_min_rows(0);
            }
            set_device(Device::Gpu);  // speculative steps on the fp16 cache, rolled back on the GPU
            test_generate_speculative(m16, random_model(cfg, 43, WeightFormat::fp16));
            set_device(Device::Both);
        });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "GPU work committed in chunks = one command buffer, bit for bit (macOS): " + name,
            [&] { test_gpu_chunked_commits(random_model(cfg, 69)); });
    {
        uint64_t seed = 100;
        for (const auto& [name, cfg] : gpu_block_configs())
            run("gpu", "KV caches on CPU and GPU: random operations = whole passes, copies consistent (macOS): " + name,
                [&, s = seed++] {
                    test_kv_random(random_model(cfg, 63), s);
                    test_kv_random(random_model(cfg, 63, WeightFormat::fp16), s + 1000);  // the cache in fp16
                });
    }
    run("gpu", "KV cache on the GPU with fp16 multiplies: one run, its fp32 repeat finds the inputs intact (macOS)",
        test_gpu_fp16_cached_repeat);
    run("gpu", "single passes on the GPU in GPU-only mode: long sequences (macOS)", [&] {
        for (const auto& [name, cfg] : long_attention_configs()) test_gpu_single_pass(random_model(cfg, 54), 48);
    });
    run("gpu", "attention + its backward on the GPU: long sequences, gradients = run_with_gradients (macOS)", [&] {
        detail::set_gpu_attention(true);
        for (const auto& [name, cfg] : long_attention_configs())
            test_gpu_gradients(random_model(cfg, 52), {48, 17, 33, 1});
        detail::set_gpu_attention(false);
    });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "GPU batched gradients = run_with_gradients (macOS): " + name,
            [&] { test_gpu_gradients(random_model(cfg, 44)); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "attribution + mixed patching on the GPU = CPU (macOS): " + name,
            [&] { test_gpu_attribution(random_model(cfg, 45)); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "fp16 weights on the GPU: = rounded float32 weights; gradients, attribution = CPU (macOS): " + name,
            [&] {
                test_gpu_fp16_matches_rounded(random_model(cfg, 43, WeightFormat::fp16));
                test_gpu_gradients(random_model(cfg, 44, WeightFormat::fp16));
                test_gpu_attribution(random_model(cfg, 45, WeightFormat::fp16));
            });
#endif
    run("analysis", "NaN never matches", test_nan_never_matches);
    run("analysis", "patch gap threshold", test_patch_gap_threshold);
    run("analysis", "per-layer head counts", test_heterogeneous_heads);
    run("grad", "attribution patching exact when linear", test_attribution_exact_when_linear);
    if (selected == 0) {  // a typo in --filter must not pass as a clean run
        std::fprintf(stderr, "no tests match\n");
        return 2;
    }
    if (list) return 0;
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
