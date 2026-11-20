// Self-contained unit tests for tfi: invariants of the analysis tools on random models.
// The forward pass itself is checked against PyTorch by python/tests/test_golden.py (golden tests).
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "tfi/analysis.h"
#include "tfi/generate.h"
#include "tfi/json.h"
#include "tfi/model.h"
#include "tfi/safetensors.h"
#include "../src/vmath.h"  // internal: vectorizable exp / erf / tanh
#include "../src/gpu.h"    // internal (macOS): GPU matrix multiplies
#include "../src/ops.h"    // internal: patching batch sizes
#include "../src/half.h"   // internal: fp16 conversion
#include "../src/gpu_chunks.h"  // internal: chunked GPU work (portable half)
#include "../src/regex.h"       // internal: tokenizer split patterns
#include "../src/unicode.h"     // internal: UTF-8, categories, NFC
#include "tfi/tokenizer.h"

using namespace tfi;

static int g_failures = 0, g_checks = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_failures;                                                           \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(double(a) - double(b)) <= (tol))

#define CHECK_THROWS(expr)                     \
    do {                                       \
        bool threw = false;                    \
        try { (void)(expr); } catch (...) { threw = true; } \
        CHECK(threw);                          \
    } while (0)

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
        const std::string part = prefix.find("mlp.") != std::string::npos ? "mlp"
                                 : prefix.find("o_proj") != std::string::npos ? "o" : "qkv";
        if (c.bias.has(part)) randn(prefix + ".bias", {out}, 0.1f);
    };
    const int64_t d = cfg.d_model;
    randn("embed.weight", {cfg.vocab_size, d}, 1.0f);
    if (cfg.positional == "learned") randn("pos_embed.weight", {cfg.max_seq_len, d}, 0.5f);
    if (!cfg.tie_embeddings) randn("unembed.weight", {cfg.vocab_size, d}, 0.2f);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const ModelConfig c = cfg.layer_config(i);
        const std::string p = "blocks." + std::to_string(i) + ".";
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
             c.norm = "layernorm"; c.positional = "learned"; c.mlp = "mlp"; c.bias = true; c.tie_embeddings = false;
         })},
        {"alibi + GQA + window + q/k/v biases only", tiny([](ModelConfig& c) {
             c.positional = "alibi"; c.n_kv_heads = 2; c.sliding_window = 3; c.bias = Biases::from_json(json::parse(R"(["qkv"])"));
         })},
        {"parallel residual", tiny([](ModelConfig& c) { c.parallel_residual = true; c.norm = "layernorm"; c.bias = true; })},
        {"sandwich + softcaps", tiny([](ModelConfig& c) {
             c.norm_position = "sandwich"; c.attn_logit_softcap = 5.0; c.final_logit_softcap = 10.0;
         })},
        {"attention-only, no norm", tiny([](ModelConfig& c) { c.mlp = "none"; c.norm = "none"; })},
        {"RoPE scaling: yarn, longrope in layer 1 (cos/sin factors)", tiny([](ModelConfig& c) {
             c.rope_scaling = RopeScaling::from_json(json::parse(R"({"type": "yarn", "factor": 4, "original_max_position_embeddings": 8})"));
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
    CHECK_THROWS(parse(R"({"type": "llama3", "factor": 8, "low_freq_factor": 4, "high_freq_factor": 1, "original_max_position_embeddings": 8})"));
    const RopeScaling y = parse(R"({"type": "yarn", "factor": 4, "original_max_position_embeddings": 8, "beta_fast": null})");
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
        m.rope_scaling = parse(R"({"type": "longrope", "short_factor": [1, 2, 3, 4], "original_max_position_embeddings": 10})");
    });
    CHECK_THROWS(detail::rope_table(wrong));
#ifdef TFI_USE_ACCELERATE
    const Device before = device();
    set_device(Device::Gpu);  // single passes on the GPU: the GPU path checks the length too
    CHECK(m.run_with_cache(seq(10, 1), cache).shape == (Shape{10, 50}));
    CHECK_THROWS(m.run_with_cache(seq(11, 1), cache));
    set_device(before);
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
    for (const auto& n : m.hook_names()) CHECK(cache.count(n));
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
    zero.add("blocks.1.hook_attn_out", [](const std::string&, Tensor& a) { std::fill(a.data.begin(), a.data.end(), 0.f); });
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
    softmax_(last.ptr(), m.cfg().vocab_size);
    CHECK_NEAR(lens.back().target_prob, last.data[7], 1e-5);
    CHECK(lens.back().top_tokens[0] == topk(last.ptr(), m.cfg().vocab_size, 1)[0]);
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
        for (int64_t i = 0; i < r.values.numel(); ++i) CHECK(std::isnan(r.values.data[i]) && std::isnan(r.effect.data[i]));
    }
}

// The straightforward versions the optimized tools must agree with: full sequences, full forward
// passes, one at a time.
static Tensor naive_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                          int64_t pos, int64_t target, int64_t other) {
    const int64_t L = m.cfg().n_layers, T = static_cast<int64_t>(clean.size()), d = m.cfg().d_model;
    ActivationCache cc;
    m.run_with_cache(clean, cc);
    const float base = logit_metric(m.forward(corrupt), pos, target, other);
    Tensor out({L, T}, NAN);
    for (int64_t i = 0; i < L; ++i) {
        const std::string name = "blocks." + std::to_string(i) + ".hook_" + site;
        if (!cc.count(name)) continue;
        const Tensor& src = cc.at(name);
        for (int64_t t = 0; t < T; ++t) {
            Hooks h;
            h.add(name, [&src, t, d](const std::string&, Tensor& a) { std::copy(src.ptr() + t * d, src.ptr() + (t + 1) * d, a.ptr() + t * d); });
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
        const std::string name = "blocks." + std::to_string(i) + ".attn.hook_z";
        const Tensor& src = cc.at(name);
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        for (int64_t hd = 0; hd < Hl; ++hd) {
            Hooks h;
            h.add(name, [&src, hd, Hl, D](const std::string&, Tensor& a) {
                for (int64_t t = 0; t < a.shape[0]; ++t)
                    std::copy(src.ptr() + (t * Hl + hd) * D, src.ptr() + (t * Hl + hd + 1) * D, a.ptr() + (t * Hl + hd) * D);
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
static void check_close(const Tensor& got, const Tensor& want) {
    CHECK(got.shape == want.shape);
    float scale = 0;
    for (float v : want.data)
        if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
    for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i) {
        if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
        else CHECK_NEAR(got.data[i], want.data[i], 1e-4 * std::max(1.f, std::fabs(want.data[i])) + 1e-5 * scale);
    }
}

// Patching runs batched passes (src/analysis.cpp): stacked rows, recomputing only the rows from the
// patched position on, and on macOS GPU linear layers. Every batch size, with the GPU off and on,
// must agree with naive patching, and the thread count must never change a result.
// Batched reference means and ablation agree with one plain forward pass per reference and per head.
static void check_ablation_matches_naive(const Model& m, const Tokens& t) {
    const int64_t T = static_cast<int64_t>(t.size());
    for (const std::vector<Tokens>& refs : std::vector<std::vector<Tokens>>{
             {seq(T, 61), seq(T, 62), seq(T, 63)},  // one mean per position
             {seq(T, 61), seq(4, 62), seq(T + 2, 63)}}) {  // other lengths: one pooled mean
        const HeadMeans means = head_means(m, refs, T);
        double count = 0;
        std::vector<Tensor> naive;
        for (int64_t l = 0; l < m.cfg().n_layers; ++l) naive.emplace_back(means.layers[l].shape);
        for (const Tokens& r : refs) {
            ActivationCache c;
            m.run_with_cache(r, c, [](const std::string& n) { return n.find("attn.hook_z") != std::string::npos; });
            for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
                const Tensor& z = c.at("blocks." + std::to_string(l) + ".attn.hook_z");
                const int64_t row = z.numel() / z.shape[0];
                for (int64_t p = 0; p < z.shape[0]; ++p)
                    for (int64_t e = 0; e < row; ++e)
                        naive[l].data[(means.per_position() ? p : 0) * row + e] += z.data[p * row + e];
            }
            count += means.per_position() ? 1.0 : static_cast<double>(r.size());
        }
        for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
            for (float& v : naive[l].data) v = static_cast<float>(v / count);
            CHECK(max_abs_diff(means.layers[l], naive[l]) <= 1e-5f * std::max(1.f, max_abs(naive[l])));
        }
        if (!means.per_position()) continue;
        const float base = next_token_loss(m.forward(t), t);
        for (Ablation mode : {Ablation::Zero, Ablation::Mean}) {
            const Tensor got = head_ablation(m, t, mode, refs);
            for (int64_t l = 0; l < m.cfg().n_layers; ++l)
                for (int64_t h = 0; h < m.layer_cfg(l).n_heads; ++h) {
                    const Hooks hk = ablate_head(m, l, h, mode, mode == Ablation::Mean ? &means : nullptr);
                    // float rounding grows with the loss (models without norms reach losses in the hundreds)
                    CHECK_NEAR(got.at({l, h}), next_token_loss(m.forward(t, &hk), t) - base, 1e-5 * std::max(1.f, std::fabs(base)));
                }
        }
    }
}

static void test_optimized_patching_matches_naive(const Model& m) {
    // forward_from reproduces forward exactly from any layer's cached input.
    const Tokens t = seq(9, 51);
    ActivationCache cache;
    const Tensor full = m.run_with_cache(t, cache);
    for (int64_t l = 0; l <= m.cfg().n_layers; ++l) {
        const Tensor start = l < m.cfg().n_layers ? cache.at("blocks." + std::to_string(l) + ".hook_resid_pre")
                                                  : cache.at("blocks." + std::to_string(l - 1) + ".hook_resid_post");
        CHECK(max_abs_diff(m.forward_from(l, start), full) == 0.f);
    }
    CHECK_THROWS(m.forward_from(-1, cache.at("blocks.0.hook_resid_pre")));
    CHECK_THROWS(m.forward_from(0, Tensor({9, m.cfg().d_model + 1})));

    Tokens corrupt = t;
    corrupt[2] = (corrupt[2] + 5) % m.cfg().vocab_size;
    corrupt[6] = (corrupt[6] + 9) % m.cfg().vocab_size;
    const int64_t last = m.cfg().n_layers - 1;
    std::vector<bool> gpu_modes = {false};
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
        for (const auto& rows : std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {5, 9}, {0, 0}}) {
            detail::set_patch_batch_rows(rows.first, rows.second);
            for (int64_t pos : {8, 3}) {
                for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"}) {
                    const PatchResult r = patch_residual(m, t, corrupt, site, pos, 1, 2);
                    check_close(r.effect, naive_patch(m, t, corrupt, site, pos, 1, 2));
                    // After the last layer's attention, earlier positions cannot reach `pos`: exactly 0.
                    if (site != "resid_pre" && !std::isnan(r.effect.at({last, 0})))
                        for (int64_t q = 0; q < pos; ++q) CHECK(r.effect.at({last, q}) == 0.f);
                }
                check_close(patch_heads(m, t, corrupt, pos, 1, 2).effect, naive_patch_heads(m, t, corrupt, pos, 1, 2));
                check_close(patch_heads(m, t, corrupt, pos, 1, -1).effect, naive_patch_heads(m, t, corrupt, pos, 1, -1));
            }
            check_ablation_matches_naive(m, t);
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

        // fp16 GPU multiplies: really used, close to naive patching (errors around 1e-3 relative),
        // the same for any thread count, and no effect when the GPU is off.
        const Tensor gpu32 = patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect;
        set_gpu_fp16(true);
        CHECK(max_abs_diff(patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect, gpu32) > 0.f);
        auto check_fp16 = [](const Tensor& got, const Tensor& want) {
            CHECK(got.shape == want.shape);
            float scale = 1.f;
            for (float v : want.data) if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
            for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
                if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
                else CHECK_NEAR(got.data[i], want.data[i], 2e-2 * scale);
        };
        for (int64_t pos : {8, 3}) {
            for (const std::string site : {"resid_pre", "attn_out", "mlp_out"})
                check_fp16(patch_residual(m, t, corrupt, site, pos, 1, 2).effect, naive_patch(m, t, corrupt, site, pos, 1, 2));
            check_fp16(patch_heads(m, t, corrupt, pos, 1, 2).effect, naive_patch_heads(m, t, corrupt, pos, 1, 2));
        }
        set_num_threads(1);
        const Tensor f1 = patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect;
        set_num_threads(4);
        CHECK(same_values(patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect, f1));
        set_num_threads(0);
        set_use_gpu(false);
        CHECK(same_values(patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect, cpu));
        set_use_gpu(true);
        set_gpu_fp16(false);
    }
    detail::set_gpu_min_rows(0);
    detail::set_gpu_min_work(0);
#endif
    set_use_gpu(false);
    CHECK(!use_gpu());
    set_use_gpu(true);
    CHECK(use_gpu());
    CHECK(!gpu_fp16());
    set_gpu_fp16(true);
    CHECK(gpu_fp16());
    set_gpu_fp16(false);

}

static void test_patch_gap_threshold() {
    PatchResult r;
    r.clean_metric = 5.f, r.corrupt_metric = 5.f;
    CHECK(!r.normalized());
    r.corrupt_metric = 5.f - 1e-7f;  // float noise around a real metric
    CHECK(!r.normalized());
    r.corrupt_metric = 4.9f;           // small but genuine
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
    for (const auto& lh : path) {
        const HeadWeights w = head_weights(m, lh.first, lh.second);
        X = matmul(matmul(X, w.W_V, true), w.W_O, true);
    }
    Tensor U = m.unembed().to_tensor();
    if (m.cfg().norm != "none" && m.cfg().norm_position != "post")
        for (int64_t v = 0; v < V; ++v)
            for (int64_t j = 0; j < d; ++j) U.data[v * d + j] *= m.weight("ln_final.weight").f32().data[j];
    const Tensor logits = matmul(X, U, true);
    int64_t hits = 0;
    for (int64_t i = 0; i < n; ++i) hits += topk(logits.ptr() + i * V, V, 1)[0] == i * V / n;
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
            CHECK_NEAR(pair.data[a * N + b], direct_ov_path(m, {{la, ha}, {lb, hb}}, n), 1.5 / n);  // a near-tie may flip
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
    Model m = random_model(tiny([](ModelConfig& c) { c.mlp = "none"; c.norm = "none"; c.positional = "learned"; c.n_heads = 1; c.d_head = 16; }));
    std::map<std::string, Tensor> w;
    for (const auto& kv : m.weights()) w[kv.first] = kv.second.to_tensor();
    for (int64_t l = 0; l < 2; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".attn.";
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
    Model m = random_model(tiny([](ModelConfig& c) { c.norm_position = "post"; c.norm = "layernorm"; }));
    CHECK_THROWS(direct_logit_attribution(m, seq(5, 1), 4, 1));
    // A single post-norm layer in a pre-norm model breaks additivity just the same.
    Model mixed = random_model(tiny([](ModelConfig& c) {
        c.norm = "layernorm";
        c.layer_overrides[1] = json::parse(R"({"norm_position": "post"})");
    }));
    CHECK_THROWS(direct_logit_attribution(mixed, seq(5, 1), 4, 1));
}

// The vectorizable exp, erf and tanh (src/vmath.h) against double precision, at every 16th float
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
    CHECK(detail::fits_half(65504.f) && !detail::fits_half(65504.5f) && !detail::fits_half(NAN) && !detail::fits_half(INFINITY));

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
    CHECK(detail::floats_to_halves(f.data(), h.data(), static_cast<int64_t>(f.size())) == over);
    bool same = true, hw = true;
    for (size_t i = 0; i < f.size(); ++i) {
        if (i != 5) same &= h[i] == float_to_half(f[i]);
#if defined(__aarch64__)
        const __fp16 x = static_cast<__fp16>(f[i]);
        uint16_t bits;
        std::memcpy(&bits, &x, 2);
        if (i != 5) hw &= bits == float_to_half(f[i]);
#endif
    }
    CHECK(same);
    CHECK(hw);
    std::vector<float> back(h.size());
    detail::halves_to_floats(h.data(), back.data(), static_cast<int64_t>(h.size()));
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
    for (int64_t i = 0; i < W.numel(); ++i) rounded &= R.data[i] == detail::half_to_float(detail::float_to_half(W.data[i]));
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
    CHECK(linear(x, w32, &bw).data == linear(x, W, &b).data);
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
    bits[3] = 0x7C00;                                              // infinity
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
    for (const auto& kv : m.weights()) w[kv.first] = kv.second.to_tensor();
    return Model(m.cfg(), w);
}

// A model with fp16 weights computes what the float32 model with the rounded weights computes.
static void test_fp16_model(const Model& m16) {
    CHECK(m16.weight_format() == WeightFormat::fp16);
    int64_t bytes = 0, bytes32 = 0;
    for (const auto& kv : m16.weights()) {
        CHECK(kv.second.is_fp16() == (kv.second.dim() == 2));
        bytes += kv.second.numel() * (kv.second.dim() == 2 ? 2 : 4);
        bytes32 += kv.second.numel() * 4;
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
    for (const auto& kv : gr)
        CHECK(g16.count(kv.first) && max_abs_diff(g16.at(kv.first), kv.second) <= 1e-4f * std::max(1.f, max_abs(kv.second)));
    const Model copy = m16;  // copies share the fp16 values
    CHECK(copy.forward(t).data == a.data);
    CHECK(copy.weight("embed.weight").f16() == m16.weight("embed.weight").f16() && copy.weight("embed.weight").f16());
    // A weight with a value outside fp16's range stays float32.
    std::map<std::string, Tensor> w;
    for (const auto& kv : m16.weights()) w[kv.first] = kv.second.to_tensor();
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
    for (const auto& kv : tensors) {
        const std::string type = dtype(kv.first);
        if (aligned && kv.second.dim() == 2 && data.size() % kAlign) {
            const size_t at = data.size(), pad = kAlign - at % kAlign;
            data.append(pad, '\0');
            header += ", \"__padding" + std::to_string(pads++) + "\": {\"dtype\": \"U8\", \"shape\": [" + std::to_string(pad) +
                      "], \"data_offsets\": [" + std::to_string(at) + ", " + std::to_string(data.size()) + "]}";
        }
        const size_t begin = data.size();
        for (float v : kv.second.data) {
            if (type == "F32") {
                data.append(reinterpret_cast<const char*>(&v), 4);
                continue;
            }
            uint16_t hb;
            if (type == "F16") {
                hb = detail::float_to_half(v);
            } else {
                uint32_t x;
                std::memcpy(&x, &v, 4);
                hb = static_cast<uint16_t>((x + 0x7FFF + ((x >> 16) & 1)) >> 16);
            }
            data.append(reinterpret_cast<const char*>(&hb), 2);
        }
        std::string shape;
        for (int64_t d : kv.second.shape) shape += (shape.empty() ? "" : ", ") + std::to_string(d);
        header += ", \"" + kv.first + "\": {\"dtype\": \"" + type + "\", \"shape\": [" + shape + "], \"data_offsets\": [" +
                  std::to_string(begin) + ", " + std::to_string(data.size()) + "]}";
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
        uint32_t x;
        std::memcpy(&x, &v, 4);
        x = ((x + 0x7FFF + ((x >> 16) & 1)) >> 16) << 16;
        std::memcpy(&v, &x, 4);
    }
    return v;
}

// Model::load reads each 2-D tensor straight into its weight, converting as it reads when the file's
// type differs from the one kept in memory: the same model as building it from the file's values.
static void check_load_layout(const std::string& path, const std::string& cfg_json, const ModelConfig& cfg,
                              std::map<std::string, Tensor> w, const std::function<std::string(const std::string&)>& dtype,
                              bool aligned) {
    write_safetensors(path, cfg_json, w, dtype, aligned);
    for (auto& kv : w)
        for (float& v : kv.second.data) v = stored(v, dtype(kv.first));  // what the file holds
    const Model want16(cfg, w, WeightFormat::fp16), want32(cfg, w);
    const Model got16 = Model::load(path, WeightFormat::fp16), got32 = Model::load(path);
    std::remove(path.c_str());
    CHECK(got16.weight_format() == WeightFormat::fp16 && got32.weight_format() == WeightFormat::fp32);
    CHECK(got16.weights().size() == want16.weights().size() && got32.weights().size() == want32.weights().size());
    bool same = true;
    for (const auto& kv : want16.weights()) {
        const Weight* g = got16.weight_or_null(kv.first);
        same &= g && g->is_fp16() == kv.second.is_fp16() && g->to_tensor().data == kv.second.to_tensor().data;
        const Weight* g32 = got32.weight_or_null(kv.first);
        same &= g32 && !g32->is_fp16() && g32->to_tensor().data == want32.weight(kv.first).to_tensor().data;
    }
    CHECK(same);
    CHECK(got16.fp32_weights() == want16.fp32_weights());
    // Aligned files: a matrix stored in the type kept in memory (and, for fp16, all finite) is used in
    // place from the mapping; every other one is a copy. Unaligned files: always copies.
    bool borrowed_right = true;
    for (const auto& kv : got16.weights()) {
        if (kv.second.dim() != 2) continue;
        const std::string type = dtype(kv.first);
        borrowed_right &= kv.second.is_borrowed() == (aligned && type == "F16" && kv.second.is_fp16());
        borrowed_right &= got32.weight(kv.first).is_borrowed() == (aligned && type == "F32");
    }
    CHECK(borrowed_right);
    const Tokens t = seq(10, 2, cfg.vocab_size);  // the file is gone by now: the mapping outlives it
    CHECK(got16.forward(t).data == want16.forward(t).data);
    CHECK(got32.forward(t).data == want32.forward(t).data);
}

static void check_load(const std::string& path, const std::string& cfg_json, const ModelConfig& cfg,
                       const std::map<std::string, Tensor>& w, const std::function<std::string(const std::string&)>& dtype) {
    check_load_layout(path, cfg_json, cfg, w, dtype, false);
    check_load_layout(path, cfg_json, cfg, w, dtype, true);
}


static void test_fp16_load() {
    const std::string path = "/tmp/tfi_unit_fp16_" + std::to_string(::getpid()) + ".safetensors";
    const ModelConfig cfg = tiny([](ModelConfig& c) { c.bias = true; c.norm = "layernorm"; });
    const std::string cfg_json = R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                                 R"("mlp_multiple_of": 8, "bias": true, "norm": "layernorm"})";
    std::map<std::string, Tensor> w;
    const Model m = random_model(cfg, 9);
    for (const auto& kv : m.weights()) w[kv.first] = kv.second.to_tensor();
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
    const ModelConfig big = tiny([](ModelConfig& c) { c.vocab_size = 70000; c.n_layers = 1; });
    const std::string big_json = R"({"vocab_size": 70000, "d_model": 16, "n_layers": 1, "n_heads": 4, "max_seq_len": 32, )"
                                 R"("mlp_multiple_of": 8})";
    std::map<std::string, Tensor> wb;
    const Model mb = random_model(big, 3);
    for (const auto& kv : mb.weights()) wb[kv.first] = kv.second.to_tensor();
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
    CHECK(m.weights().count("blocks.0.attn.k_proj.bias") && !m.weights().count("blocks.0.attn.o_proj.bias") &&
          !m.weights().count("blocks.0.mlp.up.bias"));
    // q, k and v biases change the output (they are used), and match the computation spelled out
    std::map<std::string, Tensor> w;
    for (const auto& e : m.weights()) w[e.first] = e.second.to_tensor();
    auto zeroed = w;
    for (auto& e : zeroed)
        if (e.first.find("_proj.bias") != std::string::npos) std::fill(e.second.data.begin(), e.second.data.end(), 0.f);
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
    const ModelConfig per_layer = tiny([](ModelConfig& c) { c.layer_overrides[1] = json::parse(R"({"bias": ["o"]})"); });
    const Model pl = random_model(per_layer, 5);
    CHECK(pl.weights().count("blocks.1.attn.o_proj.bias") && !pl.weights().count("blocks.0.attn.o_proj.bias"));
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
    CHECK(std::string(unicode_category(U'A')) == "Lu" && unicode_letter(0x4E2D) && unicode_number(0x2163) && !unicode_letter(U'1'));
    CHECK(unicode_white_space(0x3000) && unicode_white_space(0x85) && !unicode_white_space(0x200B));
    // NFC / NFD: composition, Hangul, mark order, an excluded composition, and blocking
    CHECK(unicode_nfc(U"é") == U"é" && unicode_nfd(U"é") == U"é");
    CHECK(unicode_nfc(U"각") == U"각" && unicode_nfd(U"각") == U"각");
    CHECK(unicode_nfd(U"ạ́") == U"ạ́");            // dot below (220) before acute (230)
    CHECK(unicode_nfc(U"ạ́") == U"ạ́");             // ạ then the acute
    CHECK(unicode_nfc(U"क़") == U"क़");                     // excluded: stays decomposed
    CHECK(unicode_nfc(U"plain ascii") == U"plain ascii");

    // regex: the subset tokenizer patterns use
    auto spans = [](const char* pattern, const char32_t* text) {
        std::vector<std::u32string> out;
        const U32String s(text);
        for (const auto& m : Regex(pattern).find_all(s)) out.push_back(s.substr(m.first, m.second - m.first));
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
        return Tokenizer::from_json(R"({"model": {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "bc": 5, "abc": 6,
            "<0xC3>": 7, "<0xA9>": 8, "▁": 9, "<s>": 10, "aa": 11}, "merges": ["a b", "ab c", "b c", "a a"])" + model_extra + "}" + extra + "}");
    };
    auto ids = [](const Tokenizer& t, const std::string& s, bool special = true) { return t.encode(s, special); };
    const Tokenizer plain = tokenizer("");
    CHECK(ids(plain, "abc") == (Tokens{6}) && ids(plain, "bca") == (Tokens{5, 1}) && ids(plain, "aaa") == (Tokens{11, 1}));
    CHECK(ids(plain, "axz") == (Tokens{1}));  // no unk_token and no fallback: unknown characters dropped
    const Tokenizer unk = tokenizer("", R"(, "unk_token": "<unk>", "fuse_unk": true)");
    CHECK(ids(unk, "axza") == (Tokens{1, 0, 1}));
    const Tokenizer fallback = tokenizer("", R"(, "unk_token": "<unk>", "byte_fallback": true)");
    CHECK(ids(fallback, "a\xC3\xA9z") == (Tokens{1, 7, 8, 0}));  // é as its two bytes; z has no bytes in the vocabulary
    const Tokenizer templ = tokenizer(R"(, "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>"}},
        {"Sequence": {"id": "A"}}], "special_tokens": {"<s>": {"id": "<s>", "ids": [10], "tokens": ["<s>"]}}})");
    CHECK(ids(templ, "ab") == (Tokens{10, 4}) && ids(templ, "ab", false) == (Tokens{4}));
    // Metaspace "first": the replacement prepended only at the start of the input, not after an added token
    const Tokenizer meta = tokenizer(R"(, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first",
        "split": false}, "added_tokens": [{"id": 10, "content": "<s>", "special": true}])");
    CHECK(ids(meta, "a b") == (Tokens{9, 1, 9, 2}) && ids(meta, "<s>a") == (Tokens{10, 1}));
    // added tokens: leftmost-longest, rstrip eats the space after, single_word needs word boundaries
    const Tokenizer added = tokenizer(R"(, "added_tokens": [{"id": 20, "content": "XY"}, {"id": 21, "content": "XYZ", "rstrip": true},
        {"id": 22, "content": "Q", "single_word": true}, {"id": 99, "content": "abc"}])");
    // ids as tokenizers assigns them: past the vocabulary (12 entries) in order, whatever the file says
    CHECK(ids(added, "aXYZ  b") == (Tokens{1, 13, 2}) && ids(added, "XYa") == (Tokens{12, 1}));
    CHECK(ids(added, "aQ Q") == (Tokens{1, 14}));            // inside a word Q is not the token (and is dropped)
    CHECK(ids(added, "cabc") == (Tokens{3, 6}));             // text in the vocabulary: the vocabulary's id
    // ignore_merges: a word in the vocabulary is that token, even when the merges wouldn't reach it
    // (Llama 3); Digits: each digit alone, or runs of them
    auto small = [](const std::string& model_extra, const std::string& pre = "null") {
        return Tokenizer::from_json(R"({"pre_tokenizer": )" + pre + R"(, "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2,
            "bc": 3, "abc": 4, "1": 5, "2": 6, "12": 7}, "merges": ["b c", "1 2"])" + model_extra + "}}");
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
    for (const auto& kv : configs()) {
        const Model m = random_model(kv.second, 1);
        std::map<std::string, Tensor> w;
        for (const auto& e : m.weights()) w[e.first] = e.second.to_tensor();
        bool all = true;
        for (const auto& e : w) {
            auto shorter = w;
            Tensor& t = shorter[e.first];
            Shape s = t.shape;
            s[0] -= 1;
            t = Tensor(s, std::vector<float>(t.data.begin(), t.data.begin() + shape_numel(s)));
            bool threw = false;
            try {
                Model bad(kv.second, shorter);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            if (!threw) std::printf("     %s: accepted %s with shape %s\n", kv.first.c_str(), e.first.c_str(), shape_str(s).c_str());
            all &= threw;
            auto missing = w;
            missing.erase(e.first);
            threw = false;
            try {
                Model bad(kv.second, missing);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            if (!threw) std::printf("     %s: accepted a model without %s\n", kv.first.c_str(), e.first.c_str());
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
        std::vector<std::string> log;             // "s<first>" per submit, "w<first>" per wait
        std::vector<size_t> ticket_chunk;         // ticket -> chunk's first group
        std::map<size_t, int> fail;               // chunk's first group -> failures left
        size_t submit(const Chunk& c) override {
            log.push_back("s" + std::to_string(c.first));
            ticket_chunk.push_back(c.first);
            return ticket_chunk.size() - 1;
        }
        bool wait(size_t t) override {
            const size_t c = ticket_chunk.at(t);
            log.push_back("w" + std::to_string(c));
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
    {   // chunk 2 fails once, chunk 3 twice: each runs again after all the first waits, on its own
        Fake f;
        f.fail[2] = 1;
        f.fail[3] = 2;
        detail::run_chunks(f, chunks);
        CHECK((f.log == std::vector<std::string>{"s0", "s2", "s3", "w0", "w2", "w3", "s2", "w2", "s3", "w3", "s3", "w3"}));
    }
    {   // three failures in a row: GpuFailure
        Fake f;
        f.fail[3] = 3;
        bool threw = false;
        try {
            detail::run_chunks(f, chunks);
        } catch (const detail::GpuFailure&) {
            threw = true;
        }
        CHECK(threw);
    }
    {   // a test-injected failure counts like a GPU failure, once per request
        Fake f;
        detail::set_gpu_fail_chunks(1);
        detail::run_chunks(f, chunks);
        CHECK((f.log == std::vector<std::string>{"s0", "s2", "s3", "w0", "w2", "w3", "s0", "w0"}));
        detail::set_gpu_fail_chunks(0);
    }
    {   // nothing to run
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
        const std::string base = std::string(R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                                             R"("mlp_multiple_of": 8, "mlp": ")") + mlp + "\"";
        const ModelConfig zero = ModelConfig::from_json(base + R"(, "d_mlp": 0})");
        CHECK(!zero.d_mlp);
        const ModelConfig layer = ModelConfig::from_json(base + R"(, "d_mlp": 24, "layer_overrides": {"1": {"d_mlp": 0}}})");
        CHECK(layer.d_mlp && *layer.d_mlp == 24 && !layer.layer_config(1).d_mlp);
        const Model source = random_model(tiny([mlp](ModelConfig& c) { c.mlp = mlp; }), 3);  // default width
        std::map<std::string, Tensor> w;
        for (const auto& kv : source.weights()) w[kv.first] = kv.second.to_tensor();
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
    const std::string path = "/tmp/tfi_unit_shapes_" + std::to_string(::getpid()) + ".safetensors";
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
        } catch (const std::runtime_error& e) {
            error = e.what();
        }
        if (error.find(why) == std::string::npos) std::printf("     %s: got '%s'\n", tensor_json.c_str(), error.c_str());
        CHECK(error.find(why) != std::string::npos);
    };
    refused(R"("x": {"dtype": "F32", "shape": [4294967296, 4294967296], "data_offsets": [0, 0]})", 0, "larger than the file");
    refused(R"("x": {"dtype": "F16", "shape": [4611686018427387904, 4], "data_offsets": [0, 16]})", 16, "larger than the file");
    refused(R"("x": {"dtype": "F32", "shape": [1099511627776, 1099511627776, 0], "data_offsets": [0, 0]})", 0, "larger than the file");
    refused(R"("x": {"dtype": "I64", "shape": [3], "data_offsets": [0, 16]})", 16, "larger than the file");
    refused(R"("x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 20]})", 24, "byte size does not match");
    refused(R"("x": {"dtype": "F32", "shape": [2, -3], "data_offsets": [0, 24]})", 24, "negative dimension");
    write(R"("x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}, "e": {"dtype": "F32", "shape": [0, 1099511627776, 1099511627776], "data_offsets": [24, 24]})", 24);
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
        uint32_t b;
        std::memcpy(&b, &x, 4);
        return (b >> 31) ? -static_cast<int64_t>(b & 0x7FFFFFFFu) : static_cast<int64_t>(b);
    };
    auto from_order = [](int64_t k) {
        const uint32_t b = k <= 0 ? 0x80000000u | static_cast<uint32_t>(-k) : static_cast<uint32_t>(k);
        float x;
        std::memcpy(&x, &b, 4);
        return x;
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
    softmax_(row, 11);
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
    for (const auto& dims : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {7, 13, 5}, {300, 67, 129}, {64, 512, 2048}}) {
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
    {   // Activations: the same polynomials as the CPU (vmath.h), over a wide range and NaN.
        Tensor xs({4002});
        for (int64_t i = 0; i < 4001; ++i) xs.data[i] = -20.f + 0.01f * static_cast<float>(i);
        xs.data[4001] = NAN;
        const auto g = detail::Gpu::create();
        detail::Gpu::Seq s(*g);
        const char* names[] = {"relu", "gelu", "gelu_tanh", "silu", "tanh", "relu2"};
        for (int a = 0; a < 6; ++a) {
            Tensor want = xs;
            detail::activate_(want, names[a]);
            std::copy(xs.data.begin(), xs.data.end(), s.reserve(0, xs.numel()));
            s.activate(0, xs.numel(), static_cast<detail::GpuAct>(a));
            s.run(false);
            for (int64_t i = 0; i < xs.numel(); ++i) near(s.data(0)[i], want.data[i], 1e-6f);
        }
        // glu, add, copy.
        const int64_t n = 1000;
        Tensor gate({n}), up({n});
        for (float& v : gate.data) v = 3.f * nd(rng);
        for (float& v : up.data) v = nd(rng);
        std::copy(gate.data.begin(), gate.data.end(), s.reserve(1, n));
        std::copy(up.data.begin(), up.data.end(), s.reserve(2, n));
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
    {   // layernorm (with bias) and rmsnorm; rows wider than a threadgroup.
        const int64_t rows = 37, d = 300;
        Tensor x({rows, d}), w({d}), b({d});
        for (float& v : x.data) v = 3.f + 2.f * nd(rng);
        for (float& v : w.data) v = 1.f + 0.2f * nd(rng);
        for (float& v : b.data) v = 0.1f * nd(rng);
        const auto g = detail::Gpu::create();
        for (const bool layernorm : {true, false}) {
            detail::Gpu::Seq s(*g);
            std::copy(x.data.begin(), x.data.end(), s.reserve(0, rows * d));
            s.reserve(1, rows * d);
            s.norm(0, 1, rows, w, layernorm ? &b : nullptr, layernorm, 1e-5f);
            s.run(false);
            for (int64_t r = 0; r < rows; ++r) {
                double mean = 0, var = 0;
                for (int64_t j = 0; j < d; ++j) mean += layernorm ? x.data[r * d + j] / double(d) : 0.0;
                for (int64_t j = 0; j < d; ++j) var += (x.data[r * d + j] - mean) * (x.data[r * d + j] - mean) / double(d);
                for (int64_t j = 0; j < d; ++j) {
                    const double y = (x.data[r * d + j] - mean) / std::sqrt(var + 1e-5) * w.data[j] + (layernorm ? b.data[j] : 0.f);
                    near(s.data(1)[r * d + j], static_cast<float>(y), 1e-5f);
                }
            }
            CHECK_THROWS(s.norm(0, 1, rows, w, &b, false, 1e-5f));  // rmsnorm has no bias
        }
    }
    {   // linear (+ bias) -> activation -> linear -> add, in fp32 and fp16. An input outside fp16's
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
            std::copy(xin.data.begin(), xin.data.end(), s.reserve(0, R * k));
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
    {   // Slots must be reserved and large enough; a failed run records nothing further.
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
                     c.norm_position = "post"; c.norm = "layernorm"; c.bias = true; c.mlp = "mlp"; c.activation = "gelu_tanh";
                 })});
    v.push_back({"post-norm parallel, relu", tiny([](ModelConfig& c) {
                     c.norm_position = "post"; c.parallel_residual = true; c.mlp = "mlp"; c.activation = "relu";
                 })});
    v.push_back({"post-norm, no norm, attention-only", tiny([](ModelConfig& c) {
                     c.norm_position = "post"; c.norm = "none"; c.mlp = "none";
                 })});
    v.push_back({"geglu tanh + qk_norm", tiny([](ModelConfig& c) { c.mlp = "geglu"; c.activation = "gelu_tanh"; c.qk_norm = true; })});
    v.push_back({"glu relu2", tiny([](ModelConfig& c) { c.mlp = "glu"; c.activation = "relu2"; })});
    v.push_back({"sandwich layernorm, tanh mlp", tiny([](ModelConfig& c) {
                     c.norm_position = "sandwich"; c.norm = "layernorm"; c.bias = true; c.mlp = "mlp"; c.activation = "tanh";
                 })});
    v.push_back({"sandwich, no norm", tiny([](ModelConfig& c) { c.norm_position = "sandwich"; c.norm = "none"; })});
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
        {"D=24, rope, window 40, softcap", cfg(48, 2, [](ModelConfig& c) { c.sliding_window = 40; c.attn_logit_softcap = 3.0; })},
        {"D=64, alibi, GQA, window 20, softcap", cfg(128, 2, [](ModelConfig& c) {
             c.positional = "alibi"; c.n_kv_heads = 1; c.sliding_window = 20; c.attn_logit_softcap = 5.0;
         })},
        {"D=128, partial rope, qk_norm, GQA", cfg(512, 4, [](ModelConfig& c) { c.rope_fraction = 0.5; c.qk_norm = true; c.n_kv_heads = 2; })},
        {"D=160 (per-row kernel), learned", cfg(320, 2, [](ModelConfig& c) { c.positional = "learned"; c.tie_embeddings = false; })},
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
        auto lp = [](int64_t i) { return "blocks." + std::to_string(i) + "."; };
        detail::RowBatch rb;
        for (int64_t i = 0; i < L; ++i) {
            rb.prefix_k.push_back(&cache.at(lp(i) + "attn.hook_k"));
            rb.prefix_v.push_back(&cache.at(lp(i) + "attn.hook_v"));
        }
        rb.gpu = m.gpu();
        for (const auto& [s0, n] : std::vector<std::pair<int64_t, int64_t>>{{0, T}, {17, T - 17}, {40, 8}, {3, 21}, {33, 1}}) rb.add(s0, n);
        Tensor x({rb.rows(), d}), want({rb.rows(), d});
        const Tensor fin = m.final_norm(cache.at(lp(L - 1) + "hook_resid_post"));
        const Tensor& pre = cache.at(lp(0) + "hook_resid_pre");
        for (size_t s = 0; s < rb.start.size(); ++s) {
            std::copy(pre.ptr() + rb.start[s] * d, pre.ptr() + (rb.start[s] + rb.len[s]) * d, x.ptr() + rb.offset[s] * d);
            std::copy(fin.ptr() + rb.start[s] * d, fin.ptr() + (rb.start[s] + rb.len[s]) * d, want.ptr() + rb.offset[s] * d);
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
    auto near = [](const Tensor& got, const Tensor& want) {
        CHECK(got.shape == want.shape);
        if (got.shape == want.shape) CHECK(max_abs_diff(got, want) <= 1e-4f * std::max(1.f, max_abs(want)));
    };
    auto pass = [&](Device dev, bool resident, ActivationCache& cache, Tensor& edited, std::map<std::string, int>& n_calls) {
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
    CHECK(cpu.count("blocks.0.attn.hook_pattern") == 1);
    for (bool resident : {true, false}) {
        ActivationCache gpu;
        Tensor gpu_edited;
        std::map<std::string, int> gpu_calls;
        near(pass(Device::Gpu, resident, gpu, gpu_edited, gpu_calls), cpu_logits);
        CHECK(gpu.size() == cpu.size());
        for (const auto& [name, want] : cpu) {
            CHECK(gpu.count(name) == 1);
            if (gpu.count(name)) {
                CHECK(gpu.at(name).shape == want.shape);
                if (gpu.at(name).shape == want.shape) CHECK(max_abs_diff(gpu.at(name), want) <= 1e-3f * std::max(1.f, max_abs(want)));
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
    for (const auto& kv : m.weights())
        if (kv.second.dim() == 2 && kv.first.rfind("blocks.", 0) == 0) weights += static_cast<double>(kv.second.numel());
    const int64_t L = m.cfg().n_layers;
    if (!detail::single_pass_on_gpu(T, L, weights / static_cast<double>(L), true, detail::gpu_min_rows(), detail::gpu_min_work())) {
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
        h.add_filter([](const std::string& n) { return n.find("blocks.0.") == 0; }, nudge);  // edits in block 0 only
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
            CHECK(gpu.count(name) == 1);
            if (gpu.count(name) && gpu.at(name).shape == w.shape) CHECK(max_abs_diff(gpu.at(name), w) <= 1e-3f * std::max(1.f, max_abs(w)));
            else CHECK(false);
        }
        CHECK(max_abs_diff(got, want) > 0.f);  // the GPU really ran
    }
}

// Both mode with the default thresholds, once the GPU has started: a model with 3M weights per
// layer sends a single pass of 2 rows to the GPU, and keeps one of 1 row on the CPU, bit for bit.
static void test_gpu_single_pass_defaults() {
    const Model m = random_model(tiny([](ModelConfig& c) { c.d_model = 512; c.n_heads = 8; }), 55);
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    double weights = 0;
    for (const auto& kv : m.weights())
        if (kv.second.dim() == 2 && kv.first.rfind("blocks.0.", 0) == 0) weights += static_cast<double>(kv.second.numel());
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
    auto lp = [](int64_t i) { return "blocks." + std::to_string(i) + "."; };
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
            std::copy(full.ptr() + rb.start[s] * d, full.ptr() + (rb.start[s] + rb.len[s]) * d, x.ptr() + rb.offset[s] * d);
        return x;
    };
    std::map<std::string, int> calls;
    Hooks hooks;
    const std::string edited = m.cfg().has_mlp() ? "hook_mlp_out" : "hook_attn_out";
    hooks.add_filter([](const std::string& n) { return n.find("attn.hook_q") != std::string::npos || n.find("attn.hook_z") != std::string::npos; },
                     [&](const std::string& n, Tensor& a) { ++calls[n]; a.data[0] += 1.f; });
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
        CHECK(seen.count(lp(mid) + edited) == 1 && seen.count(lp(L - 1) + "attn.hook_z") == 1);
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
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const int64_t L = m.cfg().n_layers, V = m.cfg().vocab_size, d = m.cfg().d_model;
    std::vector<Tokens> seqs;
    for (size_t i = 0; i < lens.size(); ++i) seqs.push_back(seq(lens[i], 21 + i));
    detail::RowBatch rb;
    rb.gpu = m.gpu();
    for (const Tokens& t : seqs) rb.add(0, static_cast<int64_t>(t.size()));
    Tensor x0({rb.rows(), d});
    for (size_t g = 0; g < seqs.size(); ++g) {
        const Tensor e = m.embed(seqs[g]);
        std::copy(e.ptr(), e.ptr() + e.numel(), x0.ptr() + rb.offset[g] * d);
    }
    std::vector<std::string> names = {"hook_embed"};
    for (int64_t i = 0; i < L; ++i)
        for (const char* n : {"hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_attn_out", "hook_mlp_out", "attn.hook_z"})
            names.push_back("blocks." + std::to_string(i) + "." + n);
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
                CHECK(got[g].count(n) == want.count(n));
                if (!got[g].count(n) || !want.count(n)) continue;
                const Tensor &a = got[g].at(n), &b = want.at(n);
                CHECK(a.shape == b.shape);
                if (a.shape == b.shape) CHECK(max_abs_diff(a, b) <= 1e-4f * std::max(1.f, max_abs(b)));
            }
        }
    }
    // Only the filter's hook points, and every segment must start at 0.
    const auto only_embed = m.gradients_rows(x0, rb, [&](int64_t, const Tensor& l) { return Tensor(l.shape, 0.f); },
                                             [](const std::string& n) { return n == "hook_embed"; });
    CHECK(only_embed[0].size() == 1 && only_embed[0].count("hook_embed") == 1);
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
        c.layer_overrides[0] = json::parse(R"({"n_heads": 2})");                // 2 heads of size 8
        c.layer_overrides[2] = json::parse(R"({"n_heads": 8, "d_head": 2})");   // 8 heads of size 2
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
    CHECK(std::find(dla.labels.begin(), dla.labels.end(), "L2H7") != dla.labels.end());
    CHECK(std::find(dla.labels.begin(), dla.labels.end(), "L0H2") == dla.labels.end());
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
            CHECK(grads.count(n) == 1);
            if (!grads.count(n)) continue;
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
            const double an = dot(g.ptr(), v.ptr(), g.numel());
            const bool ok = std::fabs(fd - an) <= 2e-3 * std::max({1.0, std::fabs(fd), std::fabs(an)});
            CHECK(ok);
            if (!ok) std::printf("    %s (other %lld): backward %.6g, finite differences %.6g\n", n.c_str(),
                                 static_cast<long long>(other), an, fd);
        }
    }
}

static void test_attribution_patching(const Model& m) {
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
        CHECK(std::isnan(one.effect.data[i]) ? std::isnan(four.effect.data[i]) : one.effect.data[i] == four.effect.data[i]);
}

// With no final norm and no MLP, the logits are linear in the last layer's outputs, so attribution
// patching there must equal exact patching.
static void test_attribution_exact_when_linear() {
    const Model m = random_model(tiny([](ModelConfig& c) { c.mlp = "none"; c.norm = "none"; c.n_layers = 3; }), 5);
    const Tokens clean = seq(9, 2);
    Tokens corrupt = clean;
    corrupt[2] = (corrupt[2] + 11) % 50;
    corrupt[6] = (corrupt[6] + 3) % 50;
    const PatchResult exact = patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9);
    const PatchResult atp = attribution_patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9);
    for (int64_t p = 0; p < 9; ++p)
        CHECK_NEAR(atp.effect.at({2, p}), exact.effect.at({2, p}), 1e-4 * std::max(1.f, std::fabs(exact.effect.at({2, p}))));
    const PatchResult eh = patch_heads(m, clean, corrupt, 8, 4, 9), ah = attribution_patch_heads(m, clean, corrupt, 8, 4, 9);
    for (int64_t h = 0; h < 4; ++h)
        CHECK_NEAR(ah.effect.at({2, h}), eh.effect.at({2, h}), 1e-4 * std::max(1.f, std::fabs(eh.effect.at({2, h}))));
    // There the gradient is constant, so averaging it over interpolated inputs changes nothing.
    const PatchResult ig = attribution_patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9, 4);
    for (int64_t p = 0; p < 9; ++p)
        CHECK_NEAR(ig.effect.at({2, p}), exact.effect.at({2, p}), 1e-4 * std::max(1.f, std::fabs(exact.effect.at({2, p}))));
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
            CHECK_NEAR(mixed.effect.data[i], exact.effect.data[i], 1e-5 * std::max(1.f, std::fabs(exact.effect.data[i])));
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
            if (!std::isnan(mixed.effect.data[i])) CHECK_NEAR(mixed.values.data[i], mixed.effect.data[i] / mixed.gap(), 1e-5);
}

static void test_mixed_patching(const Model& m) {
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t pos = 6;
    for (const std::string site : {"resid_pre", "attn_out", "mlp_out"}) {
        const PatchResult atp = attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, 2);
        const PatchResult exact = patch_residual(m, clean, corrupt, site, pos, 1, 2);
        for (const int64_t verify : {int64_t(0), int64_t(3), int64_t(1000)})
            check_mixed(mixed_patch_residual(m, clean, corrupt, site, pos, 1, 2, verify, 2), atp, exact, verify, pos + 1);
        // Verifying every site reproduces exact patching.
        const PatchResult all = mixed_patch_residual(m, clean, corrupt, site, pos, 1, 2, 1000, 2);
        for (int64_t i = 0; i < all.effect.numel(); ++i)
            CHECK(std::isnan(exact.effect.data[i]) ? std::isnan(all.effect.data[i]) : all.effect.data[i] == exact.effect.data[i]);
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
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t pos = 6;
    auto close = [](const Tensor& got, const Tensor& want) {
        CHECK(got.shape == want.shape);
        float scale = 1.f;
        for (float v : want.data) if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
        for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
            if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
            else CHECK_NEAR(got.data[i], want.data[i], 1e-4 * scale);
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
    auto lp = [](int64_t i) { return "blocks." + std::to_string(i) + "."; };
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
            std::copy(full.ptr() + b16.start[s] * d, full.ptr() + (b16.start[s] + b16.len[s]) * d, x.ptr() + b16.offset[s] * d);
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
    std::copy(e.ptr(), e.ptr() + T * d, x0.ptr());
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
        for (const auto& kv : b[s]) CHECK(a[s].count(kv.first) && a[s].at(kv.first).data == kv.second.data);
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
    const Tokens clean = seq(12, 1), corrupt = [&] { Tokens t = clean; t[5] = (t[5] + 1) % 50; return t; }();
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
    std::copy(A.data.begin(), A.data.end(), batch.floats(a));
    std::copy(Bk.data.begin(), Bk.data.end(), batch.floats(bk));
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
                for (int64_t q = 0; q < 6; ++q) want += double(batch.floats(c3)[(bi * 6 + r) * 33 + bj * 6 + q]) * batch.floats(c3)[(bi * 6 + r) * 33 + bj * 6 + q];
            sums_ok &= std::fabs(batch.floats(sums)[bi * 5 + bj] - want) <= 1e-4 * std::max(1.0, want);
        }
    for (int64_t r = 0; r < 40; ++r) top_ok &= int64_t(batch.uints(top)[r]) == topk(batch.floats(c3) + r * 33, 33, 1)[0];
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
    CHECK_THROWS(batch.block_sumsq(Mat{c3, 0, 33}, 36, 30, 7, 6, sums, 0));                      // blocks don't tile
    CHECK_THROWS(batch.matmul(Mat{a, 0, 70}, Mat{bk, 0, 33}, Mat{w32, 0, 33}, 40, 33, 70, false));  // a weight as output
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
            if (std::isnan(a.data[i]) ? !std::isnan(b.data[i]) : !(std::fabs(a.data[i] - b.data[i]) <= tol)) return false;
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

    // fp16 inputs: close to the CPU, and really fp16 (composition differs from float32 in its last bits).
    set_gpu_fp16(true);
    const Composition ch = composition_scores(m);
    double comp_scale = 0, diff32 = 0;
    for (const float v : cc.q.data)
        if (!std::isnan(v)) comp_scale = std::max(comp_scale, double(std::fabs(v)));
    for (int64_t i = 0; i < ch.q.numel(); ++i)
        if (!std::isnan(cg.q.data[i])) diff32 = std::max(diff32, double(std::fabs(ch.q.data[i] - cg.q.data[i])));
    const double tol = 5e-3 * std::max(1.0, comp_scale);
    CHECK(close(ch.q, cc.q, tol) && close(ch.k, cc.k, tol) && close(ch.v, cc.v, tol));
    CHECK(diff32 > 0);
    CHECK(close(ov_copying_scores(m, n), oc, 4.0 / n) && close(ov_pair_copying_scores(m, n), pc, 4.0 / n));
    // Values too large for fp16: the work runs again in float32, with float32's results. First an
    // input that doesn't fit (W_O x 1e6), then products that don't while their inputs do
    // (embeddings and W_O x 1000).
    for (const float scale : {1e6f, 1e3f}) {
        std::map<std::string, Tensor> w;
        for (const auto& kv : m.weights()) w[kv.first] = kv.second.to_tensor();
        for (auto& kv : w)
            if (kv.first.find("o_proj.weight") != std::string::npos || (scale < 1e6f && kv.first == "embed.weight"))
                for (float& v : kv.second.data) v *= scale;
        const Model big(m.cfg(), w);
        set_gpu_fp16(false);
        const Composition b32 = composition_scores(big);
        const Tensor o32 = ov_copying_scores(big, n), p32 = ov_pair_copying_scores(big, n);
        set_gpu_fp16(true);
        const Composition b16 = composition_scores(big);
        if (scale == 1e6f) CHECK(same(b16.q, b32.q) && same(b16.k, b32.k) && same(b16.v, b32.v));
        CHECK(same(ov_copying_scores(big, n), o32) && same(ov_pair_copying_scores(big, n), p32));
    }
    set_gpu_fp16(false);
    detail::set_gpu_min_work(0);
}
#endif

// Which single passes go to the GPU (detail::single_pass_on_gpu), with the default thresholds and
// GPU-only mode's.
static void test_single_pass_rule() {
    using detail::single_pass_on_gpu;
    const int64_t rows = 256;
    const double work = 6e10, l = 12.6e6, xxs = 0.8e6;  // weights per layer of 24x1024 and 4x256 models
    CHECK(!single_pass_on_gpu(1, 24, l, true, rows, work));     // a matrix-vector pass: the CPU
    CHECK(single_pass_on_gpu(2, 24, l, true, rows, work));      // a big model, the GPU running
    CHECK(!single_pass_on_gpu(2, 24, l, false, rows, work));    // ... but not to start it
    CHECK(!single_pass_on_gpu(16, 4, xxs, true, rows, work));   // a small model, few rows: the CPU
    CHECK(single_pass_on_gpu(64, 4, xxs, true, rows, work));    // ... enough rows x weights
    CHECK(single_pass_on_gpu(300, 2, 1e3, true, rows, work));   // the batched passes' rule
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
    const Tokens clean = seq(12, 1), corrupt = [&] { Tokens t = clean; t[5] = (t[5] + 1) % 50; return t; }();
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
    } catch (const detail::GpuFailure&) {
        threw = true;
    }
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
    const std::string path = "/tmp/tfi_unit_mapped_" + std::to_string(::getpid()) + ".safetensors";
    const ModelConfig cfg = tiny([](ModelConfig& c) { c.bias = true; c.norm = "layernorm"; });
    const std::string cfg_json = R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                                 R"("mlp_multiple_of": 8, "bias": true, "norm": "layernorm"})";
    std::map<std::string, Tensor> w;
    const Model source = random_model(cfg, 11);
    for (const auto& kv : source.weights()) w[kv.first] = kv.second.to_tensor();
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
        const Model m = random_model(tiny([D](ModelConfig& c) { c.n_heads = 2; c.d_head = D; }), 62);
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
            CHECK(both.count("blocks.0.attn.hook_pattern") == 1);
            if (both.count("blocks.0.attn.hook_pattern"))
                CHECK(max_abs_diff(both.at("blocks.0.attn.hook_pattern"), cpu.at("blocks.0.attn.hook_pattern")) <= 1e-4f);
        }
        set_device(Device::Gpu);
        std::string error;
        try {
            got = m.forward(t);
        } catch (const std::exception& e) {
            error = e.what();
        }
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
// Linear layers on 1-4 rows multiply straight from the weights (src/weight.cpp): = a double-precision
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
        for (int64_t rows = 1; rows <= 5; ++rows) {
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
            if (rows <= 4) CHECK(linear(x, w16, &b).data == linear(x, rounded, &b).data);  // 5: the library's blocks
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
            check_close(cache.keys(l), acts.at("blocks." + std::to_string(l) + ".attn.hook_k"));
            check_close(cache.values(l), acts.at("blocks." + std::to_string(l) + ".attn.hook_v"));
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
        capture.add_filter([](const std::string&) { return true; }, [&](const std::string& n, Tensor& a) { seen[n] = a; });
        m.forward_cached(Tokens(t.begin() + 9, t.end()), c, &capture);
        for (int64_t l = 0; l < L; ++l) {
            const std::string p = "blocks." + std::to_string(l) + ".";
            check_close(seen.at(p + "attn.hook_pattern"), pattern_rows(acts.at(p + "attn.hook_pattern"), 9, 3));
            check_close(seen.at(p + "hook_resid_post"), rows(acts.at(p + "hook_resid_post"), 9, T));
        }
        check_close(seen.at("hook_final"), rows(acts.at("hook_final"), 9, T));
        if (m.cfg().embedding_positional()) check_close(seen.at("hook_pos_embed"), rows(acts.at("hook_pos_embed"), 9, T));
    }
    // Edits at every step = the same edits on the whole pass: zeroing a head's z, scaling keys (what
    // later tokens read from the cache) and editing the pattern.
    {
        Hooks edit;
        edit.add("blocks.0.attn.hook_z", [](const std::string&, Tensor& z) {
            const int64_t D = z.shape[2];
            for (int64_t r = 0; r < z.shape[0]; ++r) std::fill(z.ptr() + r * z.shape[1] * D, z.ptr() + r * z.shape[1] * D + D, 0.f);
        });
        edit.add("blocks." + std::to_string(L - 1) + ".attn.hook_k", [](const std::string&, Tensor& k) {
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
    zero.add("blocks.0.hook_attn_out", [](const std::string&, Tensor& a) { std::fill(a.data.begin(), a.data.end(), 0.f); });
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

// The sampler alone: frequencies follow softmax(logits / temperature), restricted by top_k and top_p;
// NaN and -inf are never picked.
static void test_sampler() {
    const std::vector<float> logits = {1.f, 2.f, 0.f, -INFINITY, std::nanf(""), 1.5f};
    auto freqs = [&](GenerateOptions o, int n) {
        Sampler s(o);
        std::vector<double> f(logits.size(), 0);
        for (int i = 0; i < n; ++i) f[s(logits.data(), static_cast<int64_t>(logits.size()))] += 1.0 / n;
        return f;
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
    }
    o.temperature = 1.0;
    o.top_k = 2;  // tokens 1 and 5
    {
        const std::vector<double> f = freqs(o, n), p = softmax({1, 5}, 1.0);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
    }
    o.top_k = 0;
    o.top_p = 0.6;  // token 1 has 0.47 of the mass, with token 5 0.76 >= 0.6
    {
        const std::vector<double> f = freqs(o, n), p = softmax({1, 5}, 1.0);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
    }
    o.top_p = 0.3;  // token 1 alone
    CHECK_NEAR(freqs(o, 1000)[1], 1.0, 1e-9);
    o.top_p = 1;
    o.temperature = 0;
    CHECK_NEAR(freqs(o, 10)[1], 1.0, 1e-9);
    const std::vector<float> tie = {3.f, 5.f, 5.f};
    Sampler greedy(o);
    CHECK(greedy(tie.data(), 3) == 1);  // the lowest id among equal logits
    const std::vector<float> none = {-INFINITY, std::nanf("")};
    CHECK_THROWS(greedy(none.data(), 2));
}

#ifdef TFI_USE_ACCELERATE
// forward_cached and generate on the GPU (macOS): GPU-only mode, where every cached step runs there,
// and both mode with chunks of 2 rows or more on the GPU.
static void test_gpu_forward_cached(const Model& m) {
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    set_device(Device::Gpu);
    test_forward_cached(m);
    test_generate(m);
    set_device(Device::Both);
    detail::set_gpu_min_rows(2);
    test_forward_cached(m);
    detail::set_gpu_min_rows(0);
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
                if (std::find_if(std::begin(kGroups), std::end(kGroups), [&](const char* k) { return g == k; }) == std::end(kGroups)) {
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
        if ((!only.empty() && std::find(only.begin(), only.end(), group) == only.end()) ||
            name.find(filter) == std::string::npos)
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
    for (const auto& kv : configs()) {
        const Model m = random_model(kv.second, 42);
        run("analysis", "cache/hooks/causality: " + kv.first, [&] { test_cache_and_hooks(kv.first, m); });
        run("analysis", "logit lens: " + kv.first, [&] { test_logit_lens(m); });
        run("analysis", "direct logit attribution: " + kv.first, [&] { test_dla(m); });
        run("analysis", "activation patching: " + kv.first, [&] { test_patching(m); });
        run("analysis", "patching edge cases (zero gap, missing sites): " + kv.first, [&] { test_patching_edge_cases(m); });
        run("analysis", "optimized patching (batched, GPU) and ablation = naive: " + kv.first, [&] { test_optimized_patching_matches_naive(m); });
        run("analysis", "ablation + head scores: " + kv.first, [&] { test_ablation_and_scores(m); });
        run("analysis", "mean ablation has no future leak: " + kv.first, [&] { test_mean_ablation(m); });
        run("grad", "gradients = finite differences: " + kv.first, [&] { test_gradients(m); });
        run("grad", "attribution patching: " + kv.first, [&] { test_attribution_patching(m); });
        run("grad", "mixed patching: " + kv.first, [&] { test_mixed_patching(m); });
        if (kv.second.kv_heads() == kv.second.n_heads)
            run("analysis", "composition = direct: " + kv.first, [&] { test_composition_matches_direct(m); });
        run("analysis", "circuits: factored, parallel = direct: " + kv.first, [&] { test_circuits_fast_paths(m); });
        run("analysis", "KV cache: forward_cached = forward, hooks, copies: " + kv.first, [&] { test_forward_cached(m); });
        run("analysis", "generate: greedy = argmax, stops, hooks, sampling: " + kv.first, [&] { test_generate(m); });
    }
    run("core", "sampler: temperature, top_k, top_p", test_sampler);
    run("fp16", "linear layers on 1-4 rows: direct from the weights, fp16 = rounded float32", test_linear_few_rows);
    run("fp16", "fp16 conversion", test_half_conversion);
    run("fp16", "fp16 weights: reads, linear, matmul", test_fp16_weight);
    run("fp16", "fp16 weights: loading converts as it reads", test_fp16_load);
    for (const auto& kv : configs()) {
        const Model m16 = random_model(kv.second, 42, WeightFormat::fp16);
        run("fp16", "fp16 weights = float32 with rounded weights: " + kv.first, [&] { test_fp16_model(m16); });
        run("fp16", "fp16 weights: lens, DLA, patching, optimized = naive: " + kv.first, [&] {
            test_logit_lens(m16);
            test_dla(m16);
            test_patching(m16);
            test_optimized_patching_matches_naive(m16);
        });
        run("fp16", "fp16 weights: gradients, attribution, mixed, circuits: " + kv.first, [&] {
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
    for (const auto& kv : gpu_block_configs())
        run("gpu", "circuits on the GPU = CPU, chunked, retried, falling back (macOS): " + kv.first,
            [&] { test_gpu_circuits(random_model(kv.second, 46)); });
    run("gpu", "GPU block operations (macOS)", test_gpu_seq);
    for (const auto& kv : gpu_block_configs())
        run("gpu", "GPU-resident batched passes = per-layer and plain (macOS): " + kv.first,
            [&] { test_gpu_resident(random_model(kv.second, 43)); });
    for (const auto& kv : gpu_block_configs())
        run("gpu", "attention on the GPU: batched passes = per-layer and plain (macOS): " + kv.first, [&] {
            detail::set_gpu_attention(true);
            for (bool tiled : {true, false}) {
                detail::set_gpu_attention_tiled(tiled);
                test_gpu_resident(random_model(kv.second, 43));
                test_optimized_patching_matches_naive(random_model(kv.second, 47));
            }
            detail::set_gpu_attention_tiled(true);
            detail::set_gpu_attention(false);
        });
    run("gpu", "attention on the GPU: long sequences, both kernels = CPU (macOS)", test_gpu_attention_long);
    for (const auto& kv : gpu_block_configs())
        run("gpu", "attention + its backward on the GPU: batched gradients = run_with_gradients, attribution = CPU (macOS): " + kv.first, [&] {
            detail::set_gpu_attention(true);
            for (bool tiled : {true, false}) {
                detail::set_gpu_attention_tiled(tiled);
                test_gpu_gradients(random_model(kv.second, 44));
            }
            detail::set_gpu_attention_tiled(true);
            test_gpu_attribution(random_model(kv.second, 45));
            detail::set_gpu_attention(false);
        });
    for (const auto& kv : gpu_block_configs())
        run("gpu", "single passes on the GPU in GPU-only mode = CPU, hooks included (macOS): " + kv.first,
            [&] { test_gpu_single_pass(random_model(kv.second, 53), 9); });
    for (const auto& kv : gpu_block_configs())
        run("gpu", "capture hooks on the GPU next to editing hooks = CPU (macOS): " + kv.first,
            [&] { test_gpu_captures(random_model(kv.second, 56), 9); });
    run("gpu", "single passes in both mode with the default thresholds (macOS)", test_gpu_single_pass_defaults);
    for (const auto& kv : gpu_block_configs())
        run("gpu", "KV cache and generate on the GPU = forward (macOS): " + kv.first,
            [&] { test_gpu_forward_cached(random_model(kv.second, 57)); });
    run("gpu", "single passes on the GPU in GPU-only mode: long sequences (macOS)", [&] {
        for (const auto& kv : long_attention_configs()) test_gpu_single_pass(random_model(kv.second, 54), 48);
    });
    run("gpu", "attention + its backward on the GPU: long sequences, gradients = run_with_gradients (macOS)", [&] {
        detail::set_gpu_attention(true);
        for (const auto& kv : long_attention_configs()) test_gpu_gradients(random_model(kv.second, 52), {48, 17, 33, 1});
        detail::set_gpu_attention(false);
    });
    for (const auto& kv : gpu_block_configs())
        run("gpu", "GPU batched gradients = run_with_gradients (macOS): " + kv.first,
            [&] { test_gpu_gradients(random_model(kv.second, 44)); });
    for (const auto& kv : gpu_block_configs())
        run("gpu", "attribution + mixed patching on the GPU = CPU (macOS): " + kv.first,
            [&] { test_gpu_attribution(random_model(kv.second, 45)); });
    for (const auto& kv : gpu_block_configs())
        run("gpu", "fp16 weights on the GPU: = rounded float32 weights; gradients, attribution = CPU (macOS): " + kv.first, [&] {
            test_gpu_fp16_matches_rounded(random_model(kv.second, 43, WeightFormat::fp16));
            test_gpu_gradients(random_model(kv.second, 44, WeightFormat::fp16));
            test_gpu_attribution(random_model(kv.second, 45, WeightFormat::fp16));
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
