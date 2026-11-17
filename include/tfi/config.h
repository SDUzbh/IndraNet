// Mirror of indranet.config.ModelConfig, parsed from the JSON stored in exported models.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "tfi/json.h"

namespace tfi {

// RoPE frequency scaling, as Hugging Face's rope_scaling (indranet.config.ROPE_SCALING). Unused
// fields keep their defaults; rope_frequencies (src/ops.h) turns it into frequencies.
struct RopeScaling {
    std::string type;                             // linear | dynamic | llama3 | yarn | longrope
    std::optional<double> factor;
    int64_t original_max_position_embeddings = 0;
    double low_freq_factor = 1.0, high_freq_factor = 4.0;    // llama3
    double beta_fast = 32.0, beta_slow = 1.0;                 // yarn
    std::optional<double> mscale, mscale_all_dim;             // yarn
    std::optional<double> attention_factor;                   // yarn, longrope
    bool truncate = true;                                     // yarn
    std::vector<double> short_factor;                         // longrope

    static RopeScaling from_json(const json::Value& v);
};

// Which parts of a model have biases (indranet.config.BIAS_PARTS): "qkv" (q, k, v projections), "o"
// (attention output), "mlp" (every MLP linear), "norm" (LayerNorm, q/k norms included; RMSNorm has
// none). JSON: true (all), false (none) or a list of parts. Assigning a bool sets all or none.
struct Biases {
    static const std::set<std::string>& all_parts();
    std::set<std::string> parts;
    Biases& operator=(bool all);
    bool has(const std::string& part) const;
    std::string str() const;  // "all", "none" or the parts, comma-separated
    static Biases from_json(const json::Value& v);
};

struct ModelConfig {
    int64_t vocab_size = 32000;
    int64_t d_model = 512;
    int64_t n_layers = 8;
    int64_t n_heads = 8;
    std::optional<int64_t> n_kv_heads;
    std::optional<int64_t> d_head;
    std::optional<int64_t> d_mlp;  // unset (0 in the JSON, as indranet writes it too): the default width
    int64_t mlp_multiple_of = 64;
    int64_t max_seq_len = 2048;

    std::string norm = "rmsnorm";
    double norm_eps = 1e-5;
    std::string norm_position = "pre";
    std::string positional = "rope";
    double rope_theta = 10000.0;
    double rope_fraction = 1.0;
    std::optional<RopeScaling> rope_scaling;
    std::string attention = "causal";
    std::string mlp = "swiglu";
    std::string activation = "gelu";

    bool qk_norm = false;
    std::optional<int64_t> sliding_window;
    std::optional<double> attn_scale;
    std::optional<double> attn_logit_softcap;

    bool parallel_residual = false;
    Biases bias;
    bool tie_embeddings = true;
    bool embed_scale = false;
    std::optional<double> final_logit_softcap;

    std::map<int64_t, json::Value> layer_overrides;  // layer -> object of field overrides

    int64_t head_dim() const { return d_head ? *d_head : d_model / n_heads; }
    int64_t kv_heads() const { return n_kv_heads ? *n_kv_heads : n_heads; }
    bool has_mlp() const { return mlp != "none"; }
    bool embedding_positional() const { return positional == "learned" || positional == "sinusoidal"; }
    // The activation the MLP actually uses ("swiglu" forces silu, "geglu" a GELU variant).
    std::string mlp_activation() const;

    ModelConfig layer_config(int64_t layer) const;  // this config plus that layer's overrides
    void apply(const json::Value& fields);          // set every field present in a JSON object
    void validate() const;

    static ModelConfig from_json(const std::string& text);
};

}  // namespace tfi
