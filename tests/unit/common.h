// What the unit tests share: the CHECK macros and their counters, random models and inputs, and the naive
// reference computations the optimized paths are held to. Each tests/unit/<group>.cpp holds one group's
// tests (main.cpp: --group); tests.h declares them.
#pragma once

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

#include "analysis/testing.h"    // internal: patching batch sizes
#include "generation/sampling.h"  // internal: the sampler's weights
#include "gpu/gpu.h"              // internal (macOS): GPU matrix multiplies
#include "gpu/gpu_chunks.h"       // internal: chunked GPU work (portable half)
#include "model/ops.h"            // internal: the passes' building blocks
#include "model/placement.h"      // internal: CPU or GPU, and the tests' switches
#include "support/half.h"         // internal: fp16 conversion
#include "support/vmath.h"        // internal: vectorizable exp / erf / tanh
#include "text/regex.h"            // internal: tokenizer split patterns
#include "text/unicode.h"          // internal: UTF-8, categories, NFC
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

inline int g_failures = 0, g_checks = 0;

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
inline double tol_for(const Model& m, double fp32) {
    return m.weight_format() == WeightFormat::fp16 ? std::max(fp32, 2e-2) : fp32;
}
// check_close's added tolerance while an fp16 model is tested (HalfTolerance); 0 for float32.
inline double g_half_tol = 0;
struct HalfTolerance {
    double prev;
    explicit HalfTolerance(const Model& m) : prev(g_half_tol) { g_half_tol = std::max(prev, tol_for(m, 0)); }
    ~HalfTolerance() { g_half_tol = prev; }
};

// Random weights with exactly the tensor names and shapes indranet's state_dict uses.
inline Model random_model(ModelConfig cfg, uint64_t seed = 0, WeightFormat format = WeightFormat::fp32) {
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

inline ModelConfig tiny(std::function<void(ModelConfig&)> edit = nullptr) {
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

inline Tokens seq(int64_t T, uint64_t seed, int64_t V = 50) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, V - 1);
    Tokens t(T);
    for (auto& x : t) x = dist(rng);
    return t;
}

inline std::vector<std::pair<std::string, ModelConfig>> configs() {
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

// The straightforward versions the optimized tools must agree with: full sequences, full forward
// passes, one at a time.
inline Tensor naive_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
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

inline Tensor naive_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
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
inline bool same_values(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) return false;
    for (int64_t i = 0; i < a.numel(); ++i)
        if (!(a.data[i] == b.data[i] || (std::isnan(a.data[i]) && std::isnan(b.data[i])))) return false;
    return true;
}

// Two computations of the same thing that add numbers in different orders (batched vs per pass, GPU
// vs CPU): close relative to each value, plus a little relative to the tensor's largest value, since
// rounding errors scale with the numbers summed (models without norms reach ~100 in a small output).
// An fp16 model's GPU multiplies round their inputs to fp16: g_half_tol then.
inline void check_close(const Tensor& got, const Tensor& want) {
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

// Patching runs batched passes (src/analysis/analysis.cpp): stacked rows, recomputing only the rows from the
// patched position on, and on macOS GPU linear layers. Every batch size, with the GPU off and on,
// must agree with naive patching, and the thread count must never change a result.
// Batched reference means and ablation agree with one plain forward pass per reference and per head.
// The reference sets ablation is checked with: equal lengths (a mean per position), then mixed lengths
// (one pooled mean).
inline std::vector<std::vector<Tokens>> ablation_reference_sets(int64_t T) {
    return {{seq(T, 61), seq(T, 62), seq(T, 63)}, {seq(T, 61), seq(4, 62), seq(T + 2, 63)}};
}

// Naive head means and ablation for one reference set: the means summed from each reference's own
// pass, and (per-position means only) each head's loss increase from a whole forward pass with it ablated.
struct NaiveAblation {
    HeadMeans means;
    std::map<int, Tensor> loss;  // per Ablation mode: (L, max heads); empty for pooled means
    float base = 0;
};

inline std::vector<NaiveAblation> naive_ablation(const Model& m, const Tokens& t) {
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
inline void check_ablation_matches_naive(const Model& m, const Tokens& t, const std::vector<NaiveAblation>& naive) {
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

inline NaiveRefs naive_refs(const Model& m, const Tokens& t, const Tokens& corrupt) {
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

// The OV path through d_model, as before the factored form: embed -> W_V^T W_O^T per head -> unembed.
inline float direct_ov_path(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& path, int64_t max_tokens) {
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

// The same model in float32, with every weight rounded to fp16 values.
inline Model rounded_copy(const Model& m) {
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    return Model(m.cfg(), w);
}

// A safetensors file of `tensors` with `cfg_json` as its config; dtype(name) picks "F32", "F16" or
// "BF16" (rounded to nearest even) for each tensor. aligned: as indranet_hf writes files, the data and
// every matrix start on a 16 KiB boundary, with __padding tensors filling the gaps.
inline void write_safetensors(const std::string& path, const std::string& cfg_json,
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
inline float stored(float v, const std::string& dtype) {
    if (dtype == "F16") return detail::half_to_float(detail::float_to_half(v));
    if (dtype == "BF16") {
        const auto x = std::bit_cast<uint32_t>(v);
        v = std::bit_cast<float>(((x + 0x7FFF + ((x >> 16) & 1)) >> 16) << 16);
    }
    return v;
}

// Model::load reads each 2-D tensor straight into its weight, converting as it reads when the file's
// type differs from the one kept in memory: the same model as building it from the file's values.
inline void check_load_layout(const std::string& path, const std::string& cfg_json, const ModelConfig& cfg,
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

inline void check_load(const std::string& path, const std::string& cfg_json, const ModelConfig& cfg,
                       const std::map<std::string, Tensor>& w,
                       const std::function<std::string(const std::string&)>& dtype) {
    check_load_layout(path, cfg_json, cfg, w, dtype, false);
    check_load_layout(path, cfg_json, cfg, w, dtype, true);
}

#ifdef TFI_USE_ACCELERATE

// Model types the GPU blocks must handle beyond configs(): post-norm (with and without the
// parallel residual), every activation, GLU variants, qk_norm, sandwich without a norm, and layers
// of different sizes.
inline std::vector<std::pair<std::string, ModelConfig>> gpu_block_configs() {
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
inline std::vector<std::pair<std::string, ModelConfig>> long_attention_configs() {
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
#endif

// Mixed patching: exact values at the top-K estimated sites, attribution estimates elsewhere.
inline void check_mixed(const PatchResult& mixed, const PatchResult& atp, const PatchResult& exact, int64_t verify,
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

// Rows of logits from several calls, stacked.
inline Tensor stack_rows(const std::vector<Tensor>& parts) {
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
inline Tensor pattern_rows(const Tensor& P, int64_t first, int64_t n) {
    const int64_t H = P.shape[0], T = P.shape[1], nk = first + n;
    Tensor out({H, n, nk});
    for (int64_t h = 0; h < H; ++h)
        for (int64_t r = 0; r < n; ++r)
            for (int64_t j = 0; j < nk; ++j) out.data[(h * n + r) * nk + j] = P.data[(h * T + first + r) * T + j];
    return out;
}

// Hooks rounding every layer's keys and values to fp16 values (clamped to fp16's range): what a model with
// fp16 weights stores in its KV cache, as a whole pass's reference.
inline Hooks round_kv_to_half(const Model& m) {
    Hooks h;
    for (int64_t l = 0; l < m.cfg().n_layers; ++l)
        for (const char* n : {".attn.hook_k", ".attn.hook_v"})
            h.add(std::format("blocks.{}{}", l, n), [](const std::string&, Tensor& a) {
                for (float& v : a.data)
                    v = detail::half_to_float(detail::float_to_half(std::clamp(v, -65504.f, 65504.f)));
            });
    return h;
}

// One hook setup for the runtime tests: `calls` receives each call's hook point and shape, `state` whatever the
// hooks remember, so a run's whole visible behavior can be compared with decoding without speculation.
struct RuntimeHookRun {
    std::vector<std::pair<std::string, Shape>> calls;
    std::vector<float> state;
    Hooks hooks;
};
using RuntimeHookSetup = std::function<void(RuntimeHookRun&)>;

inline std::vector<std::pair<std::string, RuntimeHookSetup>> runtime_hook_setups(const Model& m) {
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
