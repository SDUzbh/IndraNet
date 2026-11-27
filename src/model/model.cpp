#include "tfi/model.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <format>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

#include "gpu/gpu.h"
#include "model/model_impl.h"
#include "model/ops.h"
#include "model/placement.h"
#include "support/half.h"
#include "support/parallel.h"
#include "support/vmath.h"
#include "tfi/device.h"
#include "tfi/safetensors.h"

namespace tfi {

using detail::hook;
using detail::layer_prefix;
using namespace detail::names;

// --- hooks ---------------------------------------------------------------------

Hooks& Hooks::add(const std::string& name, HookFn fn) {
    entries_.push_back({[name](const std::string& n) { return n == name; }, std::move(fn), false});
    return *this;
}

Hooks& Hooks::add_filter(NameFilter filter, HookFn fn) {
    entries_.push_back({std::move(filter), std::move(fn), false});
    return *this;
}

Hooks& Hooks::add_capture(NameFilter filter, HookFn fn) {
    entries_.push_back({std::move(filter), std::move(fn), true});
    return *this;
}

Hooks& Hooks::extend(const Hooks& other) {
    entries_.insert(entries_.end(), other.entries_.begin(), other.entries_.end());
    return *this;
}

void Hooks::run(const std::string& name, Tensor& act) const {
    for (const auto& e : entries_)
        if (!e.filter || e.filter(name)) e.fn(name, act);
}

bool Hooks::matches(const std::string& name) const {
    for (const auto& e : entries_)
        if (!e.filter || e.filter(name)) return true;
    return false;
}

bool Hooks::changes(const std::string& name) const {
    for (const auto& e : entries_)
        if (!e.capture && (!e.filter || e.filter(name))) return true;
    return false;
}

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kSqrtHalf = std::numbers::sqrt2 / 2;  // exact: halving a double doesn't round

}  // namespace

namespace detail {

void activate_(Tensor& t, const std::string& act) {
    if (act == "relu") {
        for (float& v : t.data) v = v > 0 ? v : 0.f;
    } else if (act == "gelu") {
        for (float& v : t.data) v = 0.5f * v * (1.f + vm::erf(v * float(kSqrtHalf)));
    } else if (act == "gelu_tanh") {
        const float k = std::sqrt(2.f / float(kPi));
        for (float& v : t.data) v = 0.5f * v * (1.f + vm::tanh(k * (v + 0.044715f * v * v * v)));
    } else if (act == "silu") {
        for (float& v : t.data) v = v / (1.f + vm::exp(-v));
    } else if (act == "tanh") {
        for (float& v : t.data) v = vm::tanh(v);
    } else if (act == "relu2") {
        for (float& v : t.data) v = v > 0 ? v * v : 0.f;
    } else {
        throw std::runtime_error("unsupported activation " + act);
    }
}

void activation_grad_(Tensor& t, const std::string& act) {
    if (act == "relu") {
        for (float& v : t.data) v = v > 0 ? 1.f : 0.f;
    } else if (act == "gelu") {  // Phi(x) + x phi(x)
        const float inv_sqrt_2pi = static_cast<float>(1.0 / std::sqrt(2.0 * kPi));
        for (float& v : t.data)
            v = 0.5f * (1.f + vm::erf(v * float(kSqrtHalf))) + v * inv_sqrt_2pi * vm::exp(-0.5f * v * v);
    } else if (act == "gelu_tanh") {
        const float k = std::sqrt(2.f / float(kPi));
        for (float& v : t.data) {
            const float th = vm::tanh(k * (v + 0.044715f * v * v * v));
            v = 0.5f * (1.f + th) + 0.5f * v * (1.f - th * th) * k * (1.f + 3.f * 0.044715f * v * v);
        }
    } else if (act == "silu") {
        for (float& v : t.data) {
            const float s = 1.f / (1.f + vm::exp(-v));
            v = s * (1.f + v * (1.f - s));
        }
    } else if (act == "tanh") {
        for (float& v : t.data) {
            const float th = vm::tanh(v);
            v = 1.f - th * th;
        }
    } else if (act == "relu2") {
        for (float& v : t.data) v = v > 0 ? 2.f * v : 0.f;
    } else {
        throw std::runtime_error("unsupported activation " + act);
    }
}

// (T, H*D) -> (H, T, D)
Tensor split_heads(const float* x, int64_t T, int64_t H, int64_t D) {
    Tensor out({H, T, D});
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h)
            std::copy(x + (t * H + h) * D, x + (t * H + h + 1) * D, out.ptr() + (h * T + t) * D);
    return out;
}

Tensor split_heads(const Tensor& x, int64_t H, int64_t D) {
    if (x.dim() != 2 || x.shape[1] != H * D)
        throw std::invalid_argument(std::format("split_heads: expected (T, {}), got {}", H * D, shape_str(x.shape)));
    return split_heads(x.ptr(), x.shape[0], H, D);
}

Tensor merge_heads(const Tensor& x) {
    const int64_t H = x.shape[0], T = x.shape[1], D = x.shape[2];
    Tensor out({T, H * D});
    for (int64_t h = 0; h < H; ++h)
        for (int64_t t = 0; t < T; ++t)
            std::copy(x.ptr() + (h * T + t) * D, x.ptr() + (h * T + t + 1) * D, out.ptr() + t * H * D + h * D);
    return out;
}

RopeTable rope_table(const ModelConfig& c) {
    RopeTable t;
    t.rot = static_cast<int64_t>(c.head_dim() * c.rope_fraction);
    t.rot -= t.rot % 2;
    if (t.rot == 0) throw std::runtime_error("rope_fraction * head_dim leaves no dimensions to rotate");
    const int64_t half = t.rot / 2;
    const float rot = static_cast<float>(t.rot), theta = static_cast<float>(c.rope_theta);
    t.inv_freq.resize(half);
    for (int64_t i = 0; i < half; ++i) t.inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / rot);
    if (!c.rope_scaling) return t;
    // Hugging Face transformers' formulas (modeling_rope_utils), in float32 as there; the same as
    // indranet.layers.positional.rope_frequencies.
    const RopeScaling& s = *c.rope_scaling;
    const double orig = static_cast<double>(s.original_max_position_embeddings);
    const float factor = s.factor ? static_cast<float>(*s.factor) : 0.f;
    std::vector<float>& inv = t.inv_freq;
    if (s.type == "linear") {
        for (float& f : inv) f /= factor;
    } else if (s.type == "dynamic") {
        t.max_len = s.original_max_position_embeddings;
    } else if (s.type == "llama3") {
        const float lo = static_cast<float>(s.low_freq_factor), hi = static_cast<float>(s.high_freq_factor);
        const float low_wavelen = static_cast<float>(orig / s.low_freq_factor),
                    high_wavelen = static_cast<float>(orig / s.high_freq_factor);
        for (float& f : inv) {
            const float wavelen = 2.f * static_cast<float>(kPi) / f;
            const float scaled = wavelen > low_wavelen ? f / factor : f;
            if (!(wavelen < high_wavelen) && !(wavelen > low_wavelen)) {  // medium: smoothed between the two
                const float smooth = (static_cast<float>(orig) / wavelen - lo) / (hi - lo);
                f = (1.f - smooth) * scaled / factor + smooth * scaled;
            } else {
                f = scaled;
            }
        }
    } else if (s.type == "yarn") {
        auto mscale = [](double scale, double m) { return scale <= 1 ? 1.0 : 0.1 * m * std::log(scale) + 1.0; };
        double attn;
        if (s.attention_factor) attn = *s.attention_factor;
        else if (s.mscale && *s.mscale && s.mscale_all_dim && *s.mscale_all_dim)
            attn = mscale(*s.factor, *s.mscale) / mscale(*s.factor, *s.mscale_all_dim);
        else attn = mscale(*s.factor, 1.0);
        t.factor = static_cast<float>(attn);
        auto correction_dim = [&](double n_rot) {
            return (double(t.rot) * std::log(orig / (n_rot * 2 * kPi))) / (2 * std::log(c.rope_theta));
        };
        double low = correction_dim(s.beta_fast), high = correction_dim(s.beta_slow);
        if (s.truncate) {
            low = std::floor(low);
            high = std::ceil(high);
        }
        low = std::max(low, 0.0);
        high = std::min(high, double(t.rot - 1));
        if (low == high) high += 0.001;
        for (int64_t i = 0; i < half; ++i) {
            const float ramp = std::clamp(
                (static_cast<float>(i) - static_cast<float>(low)) / static_cast<float>(high - low), 0.f, 1.f);
            const float pos_freq = std::pow(theta, static_cast<float>(2 * i) / rot), extrapolation = 1.f - ramp;
            inv[i] = (1.f / (factor * pos_freq)) * (1.f - extrapolation) + (1.f / pos_freq) * extrapolation;
        }
    } else if (s.type == "longrope") {
        if (std::ssize(s.short_factor) != half)
            throw std::runtime_error(
                std::format("rope_scaling 'longrope' has {} short factors, but RoPE rotates {} pairs",
                            s.short_factor.size(), half));
        double attn = 1.0;
        if (s.attention_factor) attn = *s.attention_factor;
        else if (s.factor && *s.factor > 1.0) attn = std::sqrt(1 + std::log(*s.factor) / std::log(orig));
        t.factor = static_cast<float>(attn);
        for (int64_t i = 0; i < half; ++i)
            inv[i] = 1.f / (static_cast<float>(s.short_factor[i]) * std::pow(theta, static_cast<float>(2 * i) / rot));
        t.max_len = s.original_max_position_embeddings;
    }
    return t;
}

void check_rope_length(const ModelConfig& c, const RopeTable& t, int64_t n_positions) {
    if (t.max_len && n_positions > t.max_len)
        throw std::runtime_error(
            std::format("RoPE scaling '{}' is implemented for sequences of up to original_max_position_embeddings={} "
                        "tokens; this one has {}, where its frequencies change",
                        c.rope_scaling->type, t.max_len, n_positions));
}

void apply_rope_(Tensor& x, const ModelConfig& c, bool inverse, const std::vector<int64_t>* positions) {
    const int64_t H = x.shape[0], T = x.shape[1], D = x.shape[2];
    const RopeTable table = rope_table(c);
    const int64_t rot = table.rot, half = rot / 2;
    if (rot > D) throw std::runtime_error("RoPE rotates more values than a head has");
    const std::vector<float>& inv_freq = table.inv_freq;
    if (T == 0) return;
    // cos / sin per position, computed once even when many rows share a position, and only for the
    // positions the rows have (lo..hi): a generated token's step computes its own position's, not
    // those of every position before it.
    int64_t lo = 0, hi = T - 1;
    if (positions) {
        lo = hi = (*positions)[0];
        for (int64_t p : *positions) {
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
    }
    check_rope_length(c, table, hi + 1);
    const int64_t n_pos = hi - lo + 1;
    std::vector<float> cs(n_pos * half), sn(n_pos * half);
    for (int64_t t = 0; t < n_pos; ++t)
        for (int64_t i = 0; i < half; ++i) {
            const float a = static_cast<float>(lo + t) * inv_freq[i];
            float co = std::cos(a), si = std::sin(a);
            if (table.factor != 1.f) {
                co *= table.factor;
                si *= table.factor;
            }
            cs[t * half + i] = co;
            sn[t * half + i] = inverse ? -si : si;
        }
    for (int64_t t = 0; t < T; ++t) {
        const int64_t pt = (positions ? (*positions)[t] : t) - lo;
        for (int64_t i = 0; i < half; ++i) {
            const float c_ = cs[pt * half + i], s_ = sn[pt * half + i];
            for (int64_t h = 0; h < H; ++h) {
                float* v = x.ptr() + (h * T + t) * D;
                const float x1 = v[i], x2 = v[i + half];
                v[i] = x1 * c_ - x2 * s_;
                v[i + half] = x2 * c_ + x1 * s_;
            }
        }
    }
}

std::vector<float> alibi_slopes(int64_t n) {
    auto pow2 = [](int64_t m) {
        std::vector<double> s(m);
        const double start = std::pow(2.0, -8.0 / double(m));
        for (int64_t i = 0; i < m; ++i) s[i] = std::pow(start, double(i + 1));
        return s;
    };
    std::vector<double> s;
    const double lg = std::log2(double(n));
    if (lg == std::floor(lg)) s = pow2(n);
    else {
        const int64_t closest = int64_t(1) << int64_t(std::floor(lg));
        s = pow2(closest);
        const std::vector<double> extra = pow2(2 * closest);
        for (int64_t i = 0; i < n - closest; ++i) s.push_back(extra[2 * i]);
    }
    return std::vector<float>(s.begin(), s.end());
}

}  // namespace detail

using detail::activate_;
using detail::alibi_slopes;
using detail::apply_rope_;
using detail::split_heads;

// --- model -----------------------------------------------------------------------

namespace {
// One weight in `format`: with fp16, 2-D weights whose values fit become half precision.
Weight make_weight(Tensor t, WeightFormat format) {
    if (format == WeightFormat::fp16 && t.dim() == 2 && Weight::fits_fp16(t)) return Weight::fp16(t);
    return Weight(std::move(t));
}

// A float tensor's values (F32, F16 or BF16) as float32, a block at a time:
// fn(first index, values, count).
void for_each_block(const RawTensor& t, const std::function<void(int64_t, const float*, int64_t)>& fn) {
    constexpr int64_t kBlock = int64_t(1) << 20;  // values per block: 4 MB as float32
    const int64_t n = shape_numel(t.shape);
    const bool f32 = t.dtype == "F32";
    std::vector<float> values(static_cast<size_t>(std::min(kBlock, n)));
    std::vector<uint16_t> bits(f32 ? 0 : values.size());
    for (int64_t i = 0; i < n; i += kBlock) {
        const int64_t m = std::min(kBlock, n - i);
        if (f32) {
            t.read(static_cast<uint64_t>(i) * 4, static_cast<uint64_t>(m) * 4, values.data());
        } else {
            t.read(static_cast<uint64_t>(i) * 2, static_cast<uint64_t>(m) * 2, bits.data());
            if (t.dtype == "F16") detail::halves_to_floats(bits.data(), values.data(), m);
            else
                for (int64_t j = 0; j < m; ++j)  // BF16: the top half of a float32
                    values[j] = std::bit_cast<float>(uint32_t(bits[j]) << 16);
        }
        fn(i, values.data(), m);
    }
}
}  // namespace

Model::Model(ModelConfig cfg, std::map<std::string, Tensor> weights, WeightFormat format)
    : cfg_(std::move(cfg)), format_(format) {
    for (auto it = weights.begin(); it != weights.end(); it = weights.erase(it))
        w_.emplace(it->first, make_weight(std::move(it->second), format));
    check_shapes();
}

Model::Model(ModelConfig cfg, std::map<std::string, Weight> weights, WeightFormat format)
    : cfg_(std::move(cfg)), format_(format), w_(std::move(weights)) {
    check_shapes();
}

void Model::check_shapes() {
    cfg_.validate();
    for (int64_t i = 0; i < cfg_.n_layers; ++i) layer_cfgs_.push_back(cfg_.layer_config(i));
    for (int64_t i = 0; i < cfg_.n_layers; ++i)
        for (const char* suffix : kHookSuffix) hook_names_.push_back(layer_prefix(i) + suffix);
    for (int64_t i = 0; i < cfg_.n_layers; ++i)
        for (const char* param : kParamPrefix) {
            param_names_.push_back(layer_prefix(i) + param + ".weight");
            param_names_.push_back(layer_prefix(i) + param + ".bias");
        }
    // Every weight the forward pass reads must exist with the shape the config implies (as
    // indranet's modules create them); nothing is read unchecked later.
    auto expect = [&](const std::string& name, const Shape& s) {
        const Weight* w = weight_or_null(name);
        if (!w) throw std::runtime_error(std::format("model file is missing tensor '{}'", name));
        if (w->shape != s)
            throw std::runtime_error(
                std::format("{} has shape {}, config says {}", name, shape_str(w->shape), shape_str(s)));
    };
    // Exactly the biases the config names (bias: all, none or its parts): a bias it doesn't name would
    // otherwise be used unannounced, since the passes take every bias tensor present.
    auto expect_bias = [&](const std::string& name, int64_t n, bool declared, const ModelConfig& c, const char* part) {
        if (declared) expect(name, {n});
        else if (weight_or_null(name))
            throw std::runtime_error(std::format("model file has {}, but the config's bias ({}) doesn't include '{}'",
                                                 name, c.bias.str(), part));
    };
    auto expect_norm = [&](const std::string& prefix, const ModelConfig& c, int64_t dim) {
        if (c.norm == "none") return;
        expect(prefix + ".weight", {dim});
        expect_bias(prefix + ".bias", dim, c.norm == "layernorm" && c.bias.has("norm"), c, "norm");
    };
    auto expect_linear = [&](const std::string& prefix, const ModelConfig& c, int64_t out, int64_t in,
                             const char* part) {
        expect(prefix + ".weight", {out, in});
        expect_bias(prefix + ".bias", out, c.bias.has(part), c, part);
    };
    const int64_t d = cfg_.d_model;
    expect("embed.weight", {cfg_.vocab_size, d});
    if (!cfg_.tie_embeddings) expect("unembed.weight", {cfg_.vocab_size, d});
    else if (weight_or_null("unembed.weight")) expect("unembed.weight", {cfg_.vocab_size, d});
    if (cfg_.positional == "learned") expect("pos_embed.weight", {cfg_.max_seq_len, d});
    for (int64_t i = 0; i < cfg_.n_layers; ++i) {
        const ModelConfig& c = layer_cfgs_[i];
        const std::string p = layer_prefix(i);
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        expect_norm(p + "ln1", c, d);
        expect_linear(p + "attn.q_proj", c, H * D, d, "qkv");
        expect_linear(p + "attn.k_proj", c, Hkv * D, d, "qkv");
        expect_linear(p + "attn.v_proj", c, Hkv * D, d, "qkv");
        expect_linear(p + "attn.o_proj", c, d, H * D, "o");
        if (c.qk_norm) {
            ModelConfig qc = c;
            if (qc.norm == "none") qc.norm = "rmsnorm";
            expect_norm(p + "attn.q_norm", qc, D);
            expect_norm(p + "attn.k_norm", qc, D);
        }
        if (c.norm_position == "sandwich") expect_norm(p + "ln1_post", c, d);
        if (c.has_mlp()) {
            const bool gated = c.mlp != "mlp";
            int64_t d_mlp = c.d_mlp ? *c.d_mlp : 4 * d;
            if (gated && !c.d_mlp) {  // 8/3 d_model rounded up, as indranet's gated MLPs
                const int64_t base = static_cast<int64_t>(8.0 * double(d) / 3.0), mult = c.mlp_multiple_of;
                d_mlp = mult * ((base + mult - 1) / mult);
            }
            if (gated) expect_linear(p + "mlp.gate", c, d_mlp, d, "mlp");
            expect_linear(p + "mlp.up", c, d_mlp, d, "mlp");
            expect_linear(p + "mlp.down", c, d, d_mlp, "mlp");
            expect_norm(p + "ln2", c, d);
            if (c.norm_position == "sandwich") expect_norm(p + "ln2_post", c, d);
        }
    }
    if (cfg_.norm_position != "post") expect_norm("ln_final", cfg_, d);
}

Model Model::load(const std::string& path, WeightFormat format) {
    std::map<std::string, Weight> weights;
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    // Each 2-D tensor goes from the file straight into its weight's storage: read as it is when the file
    // holds the type kept in memory, otherwise converted a block at a time as it is read. A tensor of
    // that type starting on a page boundary of the memory-mapped file (as indranet_hf writes them) is
    // not even copied: the weight uses the mapping in place, and so does the GPU.
    const RawSink straight = [&](const std::string& name, const RawTensor& t) {
        if (t.shape.size() != 2) return false;
        const int64_t n = shape_numel(t.shape);
        const bool to_fp16 = format == WeightFormat::fp16;
        auto read_all = [&](void* dst) { t.read(0, t.bytes, dst); };
        if (t.dtype == (to_fp16 ? "F16" : "F32") && t.mapped && t.bytes > 0 &&
            reinterpret_cast<uintptr_t>(t.mapped) % page == 0 &&
            (!to_fp16 || detail::all_finite_halves(static_cast<const uint16_t*>(t.mapped), n))) {
            const size_t capacity =
                (static_cast<size_t>(t.bytes) + page - 1) / page * page;  // within the mapping's last page
            weights.emplace(name, Weight::borrowed(t.shape, to_fp16, t.mapped, capacity, t.mapping));
            return true;
        }
        if (t.dtype == (to_fp16 ? "F16" : "F32")) {
            Weight w = Weight::filled(t.shape, to_fp16, read_all);
            if (!to_fp16 || detail::all_finite_halves(w.f16(), n)) {
                weights.emplace(name, std::move(w));
                return true;
            }  // an fp16 infinity or NaN: kept as float32, below
        } else if (to_fp16) {  // F32 or BF16 -> fp16, unless a value doesn't fit (then float32, below)
            int64_t bad = 0;
            Weight w = Weight::filled(t.shape, true, [&](void* dst) {
                for_each_block(t, [&](int64_t first, const float* v, int64_t count) {
                    bad += detail::floats_to_halves(v, static_cast<uint16_t*>(dst) + first, count);
                });
            });
            if (bad == 0) {
                weights.emplace(name, std::move(w));
                return true;
            }
        }
        weights.emplace(name, Weight::filled(t.shape, false, [&](void* dst) {  // -> float32
                            if (t.dtype == "F32") return read_all(dst);
                            for_each_block(t, [&](int64_t first, const float* v, int64_t count) {
                                std::copy_n(v, count, static_cast<float*>(dst) + first);
                            });
                        }));
        return true;
    };
    SafeTensors st = load_safetensors(
        path, [&](const std::string& name, Tensor&& t) { weights.emplace(name, make_weight(std::move(t), format)); },
        straight);
    auto it = st.metadata.find("config");
    if (it == st.metadata.end())
        throw std::runtime_error(path + " has no IndraNet model config; export it with indranet.export.export_model");
    Model m(ModelConfig::from_json(it->second), std::move(weights), format);
    auto v = st.metadata.find("vocab");
    if (v != st.metadata.end()) {
        std::vector<std::string> vocab;
        for (const auto& s : json::parse(v->second).array) vocab.push_back(s.as_string());
        m.set_vocab(std::move(vocab));
    }
    if (auto tk = st.metadata.find("tokenizer"); tk != st.metadata.end()) m.set_tokenizer_json(std::move(tk->second));
    // A converted model's end-of-text tokens: eos_token_id of its Hugging Face config, an id or a list.
    if (auto hf = st.metadata.find("hf_config"); hf != st.metadata.end()) {
        const json::Value cfg = json::parse(hf->second);
        std::vector<int64_t> eos;
        if (const json::Value* e = cfg.find("eos_token_id")) {
            if (e->is_array())
                for (const json::Value& x : e->array) eos.push_back(x.as_int());
            else if (!e->is_null()) eos.push_back(e->as_int());
        }
        std::erase_if(eos, [&](int64_t id) { return id < 0 || id >= m.cfg().vocab_size; });
        m.set_eos_tokens(std::move(eos));
    }
    return m;
}

namespace detail {
struct TokenizerSlot {
    std::string json;
    std::once_flag once;
    std::unique_ptr<Tokenizer> tokenizer;
};
}  // namespace detail

bool Model::has_tokenizer() const { return tokenizer_ != nullptr; }

void Model::set_tokenizer_json(std::string tokenizer_json) {
    tokenizer_ = std::make_shared<detail::TokenizerSlot>();
    tokenizer_->json = std::move(tokenizer_json);
}

const Tokenizer& Model::tokenizer() const {
    if (!tokenizer_) throw std::runtime_error("this model file has no tokenizer");
    detail::TokenizerSlot& slot = *tokenizer_;
    std::call_once(slot.once, [&] {  // an exception leaves it unparsed, to throw again next time
        slot.tokenizer = std::make_unique<Tokenizer>(Tokenizer::from_json(slot.json));
        slot.json.clear();
        slot.json.shrink_to_fit();
    });
    return *slot.tokenizer;
}

void Model::set_vocab(std::vector<std::string> vocab) {
    if (!vocab.empty() && std::ssize(vocab) != cfg_.vocab_size)
        throw std::runtime_error(std::format("vocabulary has {} entries, model has {}", vocab.size(), cfg_.vocab_size));
    vocab_ = std::move(vocab);
}

std::string Model::token_str(int64_t id) const {
    if (id >= 0 && id < std::ssize(vocab_)) return vocab_[id];
    return std::to_string(id);
}

std::string Model::decode(const Tokens& ids, bool skip_special_tokens) const {
    if (tokenizer_) return tokenizer().decode(ids, skip_special_tokens);
    if (vocab_.empty())
        throw std::runtime_error("this model file has no vocabulary or tokenizer to decode tokens with");
    std::string out;
    for (int64_t id : ids) {
        if (id < 0 || id >= std::ssize(vocab_))
            throw std::out_of_range(std::format("token {} is outside the vocabulary", id));
        out += vocab_[id];
    }
    return out;
}

void Model::set_eos_tokens(std::vector<int64_t> ids) {
    for (int64_t id : ids)
        if (id < 0 || id >= cfg_.vocab_size)
            throw std::out_of_range(std::format("eos token {} is outside the vocabulary", id));
    eos_ = std::move(ids);
}

Tokens Model::tokenize(const std::string& text, bool add_special_tokens) const {
    if (tokenizer_) return tokenizer().encode(text, add_special_tokens);
    if (vocab_.empty()) throw std::runtime_error("this model file has no vocabulary; pass token ids instead");
    return tokenize_longest_match(vocab_, text);
}

std::vector<int64_t> tokenize_longest_match(const std::vector<std::string>& vocab, const std::string& text) {
    Tokens out;
    size_t i = 0;
    while (i < text.size()) {
        int64_t best = -1;
        size_t best_len = 0;
        for (size_t id = 0; id < vocab.size(); ++id) {
            const std::string& tok = vocab[id];
            if (tok.size() > best_len && text.compare(i, tok.size(), tok) == 0) {
                best = static_cast<int64_t>(id);
                best_len = tok.size();
            }
        }
        if (best < 0)
            throw std::runtime_error(
                std::format("no token matches the text at offset {}: '{}'", i, text.substr(i, 10)));
        out.push_back(best);
        i += best_len;
    }
    return out;
}

const Weight* Model::weight_or_null(const std::string& name) const {
    auto it = w_.find(name);
    return it == w_.end() ? nullptr : &it->second;
}

const Weight& Model::weight(const std::string& name) const {
    const Weight* t = weight_or_null(name);
    if (!t) throw std::runtime_error(std::format("model file is missing tensor '{}'", name));
    return *t;
}

const Weight& Model::unembed() const {
    if (const Weight* u = weight_or_null("unembed.weight")) return *u;
    if (!cfg_.tie_embeddings) throw std::runtime_error("model file is missing tensor 'unembed.weight'");
    return weight("embed.weight");
}

int64_t Model::max_heads() const {
    int64_t h = 0;
    for (const auto& c : layer_cfgs_) h = std::max(h, c.n_heads);
    return h;
}

int64_t Model::num_parameters() const {
    int64_t n = 0;
    for (const auto& [name, w] : w_) n += w.numel();
    return n;
}

int64_t Model::weight_bytes() const {
    int64_t n = 0;
    for (const auto& [name, w] : w_) n += w.bytes();
    return n;
}

std::vector<std::string> Model::fp32_weights() const {
    std::vector<std::string> out;
    if (format_ == WeightFormat::fp16)
        for (const auto& [name, w] : w_)
            if (w.dim() == 2 && !w.is_fp16()) out.push_back(name);
    return out;
}

std::vector<std::string> Model::hook_names() const {
    std::vector<std::string> names = {"hook_embed"};
    if (cfg_.embedding_positional()) names.push_back("hook_pos_embed");
    for (int64_t i = 0; i < cfg_.n_layers; ++i) {
        const std::string p = layer_prefix(i);
        for (const char* n : {"hook_resid_pre", "hook_attn_in", "attn.hook_q", "attn.hook_k", "attn.hook_v",
                              "attn.hook_pattern", "attn.hook_z", "hook_attn_out", "hook_resid_mid"})
            names.push_back(p + n);
        if (layer_cfgs_[i].has_mlp())
            for (const char* n : {"hook_mlp_in", "mlp.hook_pre", "mlp.hook_post", "hook_mlp_out"})
                names.push_back(p + n);
        names.push_back(p + "hook_resid_post");
    }
    names.push_back("hook_final");
    return names;
}

const std::string& Model::hook_name(int64_t layer, int point) const {
    return hook_names_[static_cast<size_t>(layer * kHookPoints + point)];
}

const std::string& Model::param_name(int64_t layer, int param, bool bias) const {
    return param_names_[static_cast<size_t>((layer * kParams + param) * 2 + (bias ? 1 : 0))];
}

Tensor Model::norm(const std::string& prefix, const ModelConfig& c, const Tensor& x) const {
    if (c.norm == "none") return x;
    return norm_named(prefix + ".weight", prefix + ".bias", c, x);
}

Tensor Model::norm_named(const std::string& weight_name, const std::string& bias_name, const ModelConfig& c,
                         const Tensor& x) const {
    if (c.norm == "none") return x;
    const Tensor& w = weight(weight_name).f32();
    const Weight* b = weight_or_null(bias_name);
    const int64_t d = x.size(-1), n = x.numel() / d;
    if (w.numel() != d) throw std::runtime_error(weight_name + " does not match the normalized size");
    const float eps = static_cast<float>(c.norm_eps);
    Tensor y(x.shape);
    const float* wp = w.ptr();
    const bool layernorm = c.norm == "layernorm";
    const float* bp = layernorm && b ? b->f32().ptr() : nullptr;  // rmsnorm has no bias
    for (int64_t r = 0; r < n; ++r) {
        const std::span<const float> xi = x.row_view(r);
        float* yi = y.ptr() + r * d;
        // Two passes (mean, then squared deviations), each summed in 8 lanes (see vmath.h).
        const float mean = layernorm ? vm::sum(xi) / static_cast<float>(d) : 0.f;
        const float inv = 1.0f / std::sqrt(vm::sum_sq_dev(xi, mean) / static_cast<float>(d) + eps);
        if (bp)
            for (int64_t j = 0; j < d; ++j) yi[j] = (xi[j] - mean) * inv * wp[j] + bp[j];
        else
            for (int64_t j = 0; j < d; ++j) yi[j] = (xi[j] - mean) * inv * wp[j];
    }
    return y;
}

Tensor Model::positional_embedding(int64_t start, int64_t T) const {
    std::vector<int64_t> pos(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) pos[t] = start + t;
    return positional_embedding_at(pos);
}

Tensor Model::positional_embedding_at(const std::vector<int64_t>& positions) const {
    const int64_t d = cfg_.d_model, T = std::ssize(positions);
    Tensor pe({T, d});
    if (cfg_.positional == "learned") {
        const Weight& P = weight("pos_embed.weight");
        for (int64_t t = 0; t < T; ++t) {
            if (positions[t] < 0 || positions[t] >= cfg_.max_seq_len)
                throw std::runtime_error(
                    std::format("sequence length {} exceeds max_seq_len {}", positions[t] + 1, cfg_.max_seq_len));
            P.read_rows(positions[t], positions[t] + 1, pe.ptr() + t * d);
        }
        return pe;
    }
    // sinusoidal: [sin(pos * f_i), cos(pos * f_i)], f_i = exp(-ln(10000) * i / half), zero-padded if d is odd
    const int64_t half = d / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float f =
            std::exp(static_cast<float>(-std::log(10000.0)) * static_cast<float>(i) / static_cast<float>(half));
        for (int64_t t = 0; t < T; ++t) {
            const float a = static_cast<float>(positions[t]) * f;
            pe.data[t * d + i] = std::sin(a);
            pe.data[t * d + half + i] = std::cos(a);
        }
    }
    return pe;
}

namespace {

// linear() with the weight named `weight_name` (and the bias `bias_name` if present); on the GPU for large batches.
Tensor project(const Model& m, const Tensor& x, const std::string& weight_name, const std::string& bias_name,
               const detail::RowBatch* batch) {
    const Weight& W = m.weight(weight_name);
    const Weight* b = m.weight_or_null(bias_name);
#ifdef TFI_USE_ACCELERATE
    if (detail::linear_on_gpu(batch, x)) return batch->gpu->linear(x, W, b, batch->fp16);
#else
    (void)batch;
#endif
    return linear(x, W, b);
}

// project() for several weights that read the same x; on the GPU they share one submission.
// (weight, bias) names: each pair as project()'s.
std::vector<Tensor> project_many(const Model& m, const Tensor& x,
                                 std::initializer_list<std::pair<const std::string*, const std::string*>> names,
                                 const detail::RowBatch* batch) {
    std::vector<const Weight*> Ws, bs;
    for (const auto& [w, b] : names) {
        Ws.push_back(&m.weight(*w));
        bs.push_back(m.weight_or_null(*b));
    }
#ifdef TFI_USE_ACCELERATE
    if (detail::linear_on_gpu(batch, x)) return batch->gpu->linear_many(x, Ws, bs, batch->fp16);
#else
    (void)batch;
#endif
    return linear_many(x, Ws, bs);  // few rows: one threaded pass over all the weights
}

// Earlier positions' keys or values of one head for HeadAttention, rows of D: float32, or fp16 bit patterns (a KV
// cache kept in fp16, read as it is).
struct Prefix {
    const float* f = nullptr;
    const uint16_t* h = nullptr;
};
// Head g's rows of a layer's earlier keys or values ((H, T, D)), from row `row` of the head.
Prefix prefix_of(const detail::KVRef* r, int64_t g, int64_t row = 0) {
    if (!r) return {};
    const int64_t at = (g * r->T + row) * r->D;
    return r->h ? Prefix{nullptr, r->h + at} : Prefix{r->f + at, nullptr};
}

// With fp16 earlier keys and values (a KV cache kept in fp16): up to kDirectQueryRows query rows (a generated
// token's, a speculative step's) read them straight from the cache, widening 8 at a time in registers, so attention
// reads half the bytes; more rows (a chunk continuing a cache) widen tiles of kPrefixTile positions into a buffer
// that stays in the cache and multiply in the library.
constexpr int64_t kDirectQueryRows = 16, kPrefixTile = 512;
// ... and the key / value heads go side by side from this many cached keys (of one layer): M5 Pro, 0.9B model,
// 1500 positions: 14.0 ms a token against 15.6 for float32; under it waking the threads costs more than it saves.
constexpr int64_t kSideBySideValues = int64_t(1) << 19;
std::atomic<int64_t> g_side_by_side{0};  // set_kv_side_by_side's; 0: kSideBySideValues
int64_t side_by_side_values() {
    const int64_t v = g_side_by_side.load();
    return v > 0 ? v : kSideBySideValues;
}

float* prefix_buffer(int64_t floats) {
    thread_local std::vector<float> buf;
    if (std::ssize(buf) < floats) buf.resize(static_cast<size_t>(floats));
    return buf.data();
}

// probs[r * ld + j] = q_r . k_j for R query rows and the `first` fp16 keys k (rows of D).
template <int R>
void scores_half_rows(const float* q, int64_t first, int64_t D, const uint16_t* k, float* probs, int64_t ld) {
    for (int64_t j = 0; j < first; ++j) {
        const uint16_t* kj = k + j * D;
        float sum[R];
        int64_t d = 0;
#if defined(__aarch64__)
        float32x4_t acc[R][2];
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = vdupq_n_f32(0.f);
        for (; d + 8 <= D; d += 8) {
            const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(kj + d));
            const float32x4_t lo = vcvt_f32_f16(vget_low_f16(h)), hi = vcvt_high_f32_f16(h);
            for (int r = 0; r < R; ++r) {
                acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(q + r * D + d), lo);
                acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(q + r * D + d + 4), hi);
            }
        }
        for (int r = 0; r < R; ++r) sum[r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
#else
        for (int r = 0; r < R; ++r) sum[r] = 0.f;
#endif
        for (; d < D; ++d) {
            const float kv = detail::half_to_float(kj[d]);
            for (int r = 0; r < R; ++r) sum[r] += q[r * D + d] * kv;
        }
        for (int r = 0; r < R; ++r) probs[r * ld + j] = sum[r];
    }
}

// out (R rows, stride ldo) = probs[:, :first] @ v for the `first` fp16 values v (rows of D).
template <int R>
void apply_half_rows(const float* probs, int64_t first, int64_t ld, int64_t D, const uint16_t* v, float* out,
                     int64_t ldo) {
    for (int r = 0; r < R; ++r) std::fill_n(out + r * ldo, D, 0.f);
    for (int64_t j = 0; j < first; ++j) {
        float p[R];
        bool any = false;
        for (int r = 0; r < R; ++r) any |= (p[r] = probs[r * ld + j]) != 0.f;
        if (!any) continue;  // masked for every row (or past the window)
        const uint16_t* vj = v + j * D;
        int64_t d = 0;
#if defined(__aarch64__)
        for (; d + 8 <= D; d += 8) {
            const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(vj + d));
            const float32x4_t lo = vcvt_f32_f16(vget_low_f16(h)), hi = vcvt_high_f32_f16(h);
            for (int r = 0; r < R; ++r) {
                float* o = out + r * ldo + d;
                vst1q_f32(o, vfmaq_n_f32(vld1q_f32(o), lo, p[r]));
                vst1q_f32(o + 4, vfmaq_n_f32(vld1q_f32(o + 4), hi, p[r]));
            }
        }
#endif
        for (; d < D; ++d) {
            const float vv = detail::half_to_float(vj[d]);
            for (int r = 0; r < R; ++r) out[r * ldo + d] += p[r] * vv;
        }
    }
}

using detail::side_by_side;

// R = 1..4 rows at a time over `rows` rows.
template <std::invocable<int64_t, std::integral_constant<int, 1>> F>
void in_row_groups(int64_t rows, F&& f) {
    for (int64_t r = 0; r < rows; r += 4) {
        switch (std::min<int64_t>(4, rows - r)) {
            case 1: f(r, std::integral_constant<int, 1>()); break;
            case 2: f(r, std::integral_constant<int, 2>()); break;
            case 3: f(r, std::integral_constant<int, 3>()); break;
            default: f(r, std::integral_constant<int, 4>()); break;
        }
    }
}

// probs[r * ld + j] = q_r . k_j for the `first` fp16 keys k (rows of D).
void scores_half(const float* q, int64_t rows, int64_t first, int64_t D, const uint16_t* k, float* probs, int64_t ld) {
    if (rows <= kDirectQueryRows) {
        in_row_groups(rows, [&](int64_t r, auto n) {
            scores_half_rows<decltype(n)::value>(q + r * D, first, D, k, probs + r * ld, ld);
        });
        return;
    }
    float* wide = prefix_buffer(std::min(first, kPrefixTile) * D);
    for (int64_t j = 0; j < first; j += kPrefixTile) {
        const int64_t n = std::min(kPrefixTile, first - j);
        detail::halves_to_floats(k + j * D, wide, n * D);
        detail::gemm(/*trans_b=*/true, rows, n, D, q, D, wide, D, probs + j, ld);
    }
}

// out (rows, stride ldo) = probs[:, :first] @ v for the `first` fp16 values v (rows of D).
void apply_half(const float* probs, int64_t rows, int64_t first, int64_t ld, int64_t D, const uint16_t* v, float* out,
                int64_t ldo) {
    if (rows <= kDirectQueryRows) {
        in_row_groups(rows, [&](int64_t r, auto n) {
            apply_half_rows<decltype(n)::value>(probs + r * ld, first, ld, D, v, out + r * ldo, ldo);
        });
        return;
    }
    float* wide = prefix_buffer(std::min(first, kPrefixTile) * D);
    for (int64_t j = 0; j < first; j += kPrefixTile) {
        const int64_t n = std::min(kPrefixTile, first - j);
        detail::halves_to_floats(v + j * D, wide, n * D);
        detail::gemm(/*trans_b=*/false, rows, D, n, probs + j, ld, wide, D, out, ldo, /*accumulate=*/j > 0);
    }
}

// Attention of n queries at positions first, first+1, ... over keys 0..first+n-1 for one head.
// Keys and values of positions before `first` come from (kp, vp), the rest from (kn, vn), all
// rows of D floats. `probs` (n rows, stride ld >= first + n) receives the attention pattern: each
// row is masked and softmaxed over the keys it may attend to (j <= its position, within the
// window); disallowed entries are exactly 0.
// Query rows per block of a whole sequence's attention on the CPU (Model::attend_into).
constexpr int64_t kCausalBlockRows = 128;

struct HeadAttention {
    int64_t D;
    float scale, cap;  // cap 0 = no softcap
    float slope;       // ALiBi slope, 0 = none
    int64_t window;    // 0 = none

    void pattern(const float* q, int64_t n, int64_t first, Prefix kp, const float* kn, float* probs, int64_t ld) const {
        scores(q, n, first, n, kp, kn, probs, ld);
        for (int64_t r = 0; r < n; ++r) finish(probs + r * ld, first + r, first + n);
    }

    // Raw scores q . k of `rows` query rows (stride D) against keys 0..first-1 (kp) and the n_new keys
    // after them (kn) into probs (stride ld). Rows may be several heads' queries sharing these keys.
    void scores(const float* q, int64_t rows, int64_t first, int64_t n_new, Prefix kp, const float* kn, float* probs,
                int64_t ld) const {
        if (first && kp.h) scores_half(q, rows, first, D, kp.h, probs, ld);
        else if (first) detail::gemm(/*trans_b=*/true, rows, first, D, q, D, kp.f, D, probs, ld);
        detail::gemm(/*trans_b=*/true, rows, n_new, D, q, D, kn, D, probs + first, ld);
    }

    // One row of scores into attention weights for the query at position i, over keys 0..end-1: scaled,
    // softcapped, ALiBi'd (this->slope), masked to j <= i (and the window) and softmaxed; disallowed
    // entries exactly 0.
    void finish(float* rowp, int64_t i, int64_t end) const {
        const int64_t j0 = window ? std::max<int64_t>(0, i - window + 1) : 0;
        float* seg = rowp + j0;
        const int64_t m = i + 1 - j0;
        for (int64_t j = 0; j < m; ++j) seg[j] *= scale;
        if (cap)
            for (int64_t j = 0; j < m; ++j) seg[j] = cap * vm::tanh(seg[j] / cap);
        if (slope != 0.f)
            for (int64_t j = 0; j < m; ++j) seg[j] += -slope * static_cast<float>(i - j0 - j);
        vm::softmax(std::span(seg, m));
        std::fill(rowp, seg, 0.f);
        std::fill(rowp + i + 1, rowp + end, 0.f);
    }

    // out (n rows, stride ldo) = probs @ values.
    void apply(const float* probs, int64_t n, int64_t first, int64_t ld, Prefix vp, const float* vn, float* out,
               int64_t ldo) const {
        apply_rows(probs, n, first, n, ld, vp, vn, out, ldo);
    }

    // apply for `rows` rows of weights over keys 0..first-1 (vp) and the n_new after them (vn).
    void apply_rows(const float* probs, int64_t rows, int64_t first, int64_t n_new, int64_t ld, Prefix vp,
                    const float* vn, float* out, int64_t ldo) const {
        if (first && vp.h) apply_half(probs, rows, first, ld, D, vp.h, out, ldo);
        else if (first) detail::gemm(/*trans_b=*/false, rows, D, first, probs, ld, vp.f, D, out, ldo);
        detail::gemm(/*trans_b=*/false, rows, D, n_new, probs + first, ld, vn, D, out, ldo, /*accumulate=*/first > 0);
    }
};

}  // namespace

Tensor Model::attention(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const {
    auto names = [&](int param) { return std::make_pair(&param_name(layer, param), &param_name(layer, param, true)); };
    std::vector<Tensor> qkv = project_many(*this, x, {names(kParamQ), names(kParamK), names(kParamV)}, batch);
    Tensor z = attend(layer, std::move(qkv[0]), std::move(qkv[1]), std::move(qkv[2]), hooks, batch);
    return project(*this, z, param_name(layer, kParamO), param_name(layer, kParamO, true), batch);
}

Tensor Model::attend(int64_t layer, Tensor q2, Tensor k2, Tensor v2, const Hooks* hooks,
                     const detail::RowBatch* batch) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const int64_t T = q2.shape[0], Hd = c.n_heads * c.head_dim(), KVd = c.kv_heads() * c.head_dim();
    if (q2.dim() != 2 || q2.shape[1] != Hd || k2.shape != Shape{T, KVd} || v2.shape != Shape{T, KVd})
        throw std::invalid_argument("attend: q, k, v must be (T, H*D), (T, Hkv*D), (T, Hkv*D)");
    Tensor z({T, Hd});
    attend_into(layer, q2.ptr(), k2.ptr(), v2.ptr(), T, hooks, batch, z.ptr());
    return z;
}

void Model::attend_into(int64_t layer, const float* q2, const float* k2, const float* v2, int64_t T, const Hooks* hooks,
                        const detail::RowBatch* batch, float* z_out) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim(), rep = H / Hkv;
    if (batch && batch->whole) batch = nullptr;  // one whole sequence: as a single pass, hook_pattern included

    Tensor q = split_heads(q2, T, H, D);
    Tensor k = split_heads(k2, T, Hkv, D);
    Tensor v = split_heads(v2, T, Hkv, D);
    if (c.qk_norm) {
        ModelConfig qc = c;
        if (qc.norm == "none") qc.norm = "rmsnorm";
        q = norm_named(param_name(layer, kParamQNorm), param_name(layer, kParamQNorm, true), qc, q);
        k = norm_named(param_name(layer, kParamKNorm), param_name(layer, kParamKNorm, true), qc, k);
    }
    if (c.positional == "rope") {
        std::vector<int64_t> positions;
        if (batch)
            for (size_t s = 0; s < batch->start.size(); ++s)
                for (int64_t r = 0; r < batch->len[s]; ++r) positions.push_back(batch->start[s] + r);
        apply_rope_(q, c, false, batch ? &positions : nullptr);
        apply_rope_(k, c, false, batch ? &positions : nullptr);
    }
    hook(hooks, hook_name(layer, kHookQ), q);
    hook(hooks, hook_name(layer, kHookK), k);
    hook(hooks, hook_name(layer, kHookV), v);
    // Attention reads T rows of each: a hook that resized one would make it read past the end.
    if (q.shape != Shape{H, T, D} || k.shape != Shape{Hkv, T, D} || v.shape != Shape{Hkv, T, D})
        throw std::runtime_error(layer_prefix(layer) +
                                 "attn.hook_q / hook_k / hook_v: a hook changed the activation's shape");

    const std::vector<float> slopes = c.positional == "alibi" ? alibi_slopes(H) : std::vector<float>();
    HeadAttention ha{D, static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(D))),
                     c.attn_logit_softcap ? static_cast<float>(*c.attn_logit_softcap) : 0.f, 0.f,
                     c.sliding_window ? *c.sliding_window : 0};
    // z goes straight to z_out, unless a hook needs it as a tensor first.
    const bool hook_z = hooks && hooks->matches(hook_name(layer, kHookZ));
    Tensor zt = hook_z ? Tensor({T, H, D}) : Tensor();
    float* z = hook_z ? zt.ptr() : z_out;
    if (!batch && hooks && hooks->matches(hook_name(layer, kHookPattern))) {  // the whole pattern, for its hook
        Tensor pattern({H, T, T});
        for (int64_t h = 0; h < H; ++h) {
            ha.slope = slopes.empty() ? 0.f : slopes[h];
            const float* kh = k.ptr() + (h / rep) * T * D;
            ha.pattern(q.ptr() + h * T * D, T, 0, Prefix{}, kh, pattern.ptr() + h * T * T, T);
        }
        hook(hooks, hook_name(layer, kHookPattern), pattern);
        for (int64_t h = 0; h < H; ++h) {
            const float* vh = v.ptr() + (h / rep) * T * D;
            ha.apply(pattern.ptr() + h * T * T, T, 0, T, Prefix{}, vh, z + h * D, H * D);
        }
    } else if (!batch) {
        // Blocks of query rows, each scored against the keys up to its last row only: the earlier ones as a prefix
        // (from the first row's window start), the block's own as new keys. So the masked triangle above the
        // diagonal blocks is skipped, and no T x T pattern is kept, only a block's rows of weights.
        // Heads go side by side, each with weights of its own (M5 Pro, a 0.9B model's prompt pass of 4096 tokens:
        // 1.6x faster than with the whole pattern, 1.1x with heads one after another; blocks of 64 to 512 rows
        // about as fast).
        const int64_t BQ = kCausalBlockRows;
        side_by_side(H, [&](int64_t h) {
            HeadAttention hh = ha;  // (its slope is the head's)
            hh.slope = slopes.empty() ? 0.f : slopes[h];
            const float* kh = k.ptr() + (h / rep) * T * D;
            const float* vh = v.ptr() + (h / rep) * T * D;
            std::vector<float> probs(static_cast<size_t>(std::min(BQ, T) * T));
            for (int64_t r0 = 0; r0 < T; r0 += BQ) {
                const int64_t n = std::min(BQ, T - r0);
                // Positions relative to lo, the first row's window start: what finish() masks and ALiBi weighs are
                // distances, the same either way.
                const int64_t lo = hh.window ? std::max<int64_t>(0, r0 + 1 - hh.window) : 0, first = r0 - lo;
                hh.pattern(q.ptr() + (h * T + r0) * D, n, first, Prefix{kh + lo * D, nullptr}, kh + r0 * D,
                           probs.data(), first + n);
                hh.apply(probs.data(), n, first, first + n, Prefix{vh + lo * D, nullptr}, vh + r0 * D,
                         z + (r0 * H + h) * D, H * D);
            }
        });
    } else if (batch->cached) {
        // One segment continuing a cached sequence (forward_cached). The rep query heads sharing a key /
        // value head are next to each other in q ((H, T, D): rep * T rows), so they are scored and
        // applied as one matrix: each key / value head's cached positions are read once, not rep times.
        // A hook on hook_pattern gets the new queries' pattern over every key, (H, T, first + T).
        const int64_t first = batch->start.at(0), nk = first + T, GR = rep * T;
        const detail::KVRef* pk = first ? &batch->prefix_k.at(layer) : nullptr;
        const detail::KVRef* pv = first ? &batch->prefix_v.at(layer) : nullptr;
        const bool want = hooks && hooks->matches(hook_name(layer, kHookPattern));
        Tensor pattern = want ? Tensor({H, T, nk}) : Tensor();
        std::vector<float> group_probs(want ? 0 : static_cast<size_t>(GR * nk)), out(static_cast<size_t>(GR * D));
        auto probs_of = [&](int64_t g) { return want ? pattern.ptr() + g * GR * nk : group_probs.data(); };
        auto score = [&](int64_t g) {
            float* P = probs_of(g);
            ha.scores(q.ptr() + g * GR * D, GR, first, T, prefix_of(pk, g), k.ptr() + g * T * D, P, nk);
            for (int64_t j = 0; j < rep; ++j) {
                ha.slope = slopes.empty() ? 0.f : slopes[g * rep + j];
                for (int64_t r = 0; r < T; ++r) ha.finish(P + (j * T + r) * nk, first + r, nk);
            }
        };
        auto apply = [&](int64_t g) {  // into `out` (head j, row r) = row j T + r, then to z (T, H, D)
            ha.apply_rows(probs_of(g), GR, first, T, nk, prefix_of(pv, g), v.ptr() + g * T * D, out.data(), D);
            for (int64_t j = 0; j < rep; ++j)
                for (int64_t r = 0; r < T; ++r)
                    std::copy(out.data() + (j * T + r) * D, out.data() + (j * T + r + 1) * D,
                              z + (r * H + g * rep + j) * D);
        };
        if (want) {
            for (int64_t g = 0; g < Hkv; ++g) score(g);
            hook(hooks, hook_name(layer, kHookPattern), pattern);
            if (pattern.numel() != H * T * nk)
                throw std::runtime_error(hook_name(layer, kHookPattern) + ": a hook changed the activation's size");
            for (int64_t g = 0; g < Hkv; ++g) apply(g);
        } else if (pk && pk->h && first * D * Hkv >= side_by_side_values()) {
            // An fp16 cache: its prefix is read by kernels running on one core each (the library's float32 multiply
            // uses the matrix units), so over a long prefix the key / value heads go side by side, each with buffers
            // of its own (a short one costs less than waking the threads). Each head's arithmetic is the same
            // whatever the threads.
            side_by_side(Hkv, [&](int64_t g) {
                HeadAttention hg = ha;  // (its slope changes per head)
                std::vector<float> P(static_cast<size_t>(GR * nk)), o(static_cast<size_t>(GR * D));
                hg.scores(q.ptr() + g * GR * D, GR, first, T, prefix_of(pk, g), k.ptr() + g * T * D, P.data(), nk);
                for (int64_t j = 0; j < rep; ++j) {
                    hg.slope = slopes.empty() ? 0.f : slopes[g * rep + j];
                    for (int64_t r = 0; r < T; ++r) hg.finish(P.data() + (j * T + r) * nk, first + r, nk);
                }
                hg.apply_rows(P.data(), GR, first, T, nk, prefix_of(pv, g), v.ptr() + g * T * D, o.data(), D);
                for (int64_t j = 0; j < rep; ++j)
                    for (int64_t r = 0; r < T; ++r)
                        std::copy(o.data() + (j * T + r) * D, o.data() + (j * T + r + 1) * D,
                                  z + (r * H + g * rep + j) * D);
            });
        } else {
            for (int64_t g = 0; g < Hkv; ++g) {
                score(g);
                apply(g);
            }
        }
    } else {  // stacked segments: no hook_pattern (each segment has its own pattern)
        std::vector<float> probs;
        for (size_t s = 0; s < batch->start.size(); ++s) {
            const int64_t first = batch->start[s], n = batch->len[s], off = batch->offset[s], nk = first + n;
            const detail::KVRef* pk = first ? &batch->prefix_k_of(s, layer) : nullptr;
            const detail::KVRef* pv = first ? &batch->prefix_v_of(s, layer) : nullptr;
            probs.resize(n * nk);
            for (int64_t h = 0; h < H; ++h) {
                const int64_t g = h / rep;
                ha.slope = slopes.empty() ? 0.f : slopes[h];
                ha.pattern(q.ptr() + (h * T + off) * D, n, first, prefix_of(pk, g), k.ptr() + (g * T + off) * D,
                           probs.data(), nk);
                ha.apply(probs.data(), n, first, nk, prefix_of(pv, g), v.ptr() + (g * T + off) * D,
                         z + (off * H + h) * D, H * D);
            }
        }
    }
    if (hook_z) {
        hook(hooks, hook_name(layer, kHookZ), zt);
        if (zt.numel() != T * H * D)
            throw std::runtime_error(hook_name(layer, kHookZ) + " changed the activation's size");
        std::copy_n(zt.ptr(), T * H * D, z_out);
    }
}

Tensor Model::mlp(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string act = c.mlp_activation();
    Tensor post;
    if (c.mlp == "mlp") {
        Tensor pre = project(*this, x, param_name(layer, kParamUp), param_name(layer, kParamUp, true), batch);
        hook(hooks, hook_name(layer, kHookPre), pre);
        post = std::move(pre);
        activate_(post, act);
    } else {  // glu, swiglu, geglu: down(act(gate(x)) * up(x))
        auto names = [&](int param) {
            return std::make_pair(&param_name(layer, param), &param_name(layer, param, true));
        };
        std::vector<Tensor> gate_up = project_many(*this, x, {names(kParamGate), names(kParamUp)}, batch);
        hook(hooks, hook_name(layer, kHookPre), gate_up[0]);
        post = std::move(gate_up[0]);
        activate_(post, act);
        post = mul(post, gate_up[1]);
    }
    hook(hooks, hook_name(layer, kHookPost), post);
    return project(*this, post, param_name(layer, kParamDown), param_name(layer, kParamDown, true), batch);
}

Tensor Model::block(int64_t layer, Tensor x, const Hooks* hooks, const detail::RowBatch* batch) const {
    const ModelConfig& c = layer_cfgs_[layer];
    auto norm_ = [&](int param, const Tensor& t) {
        return norm_named(param_name(layer, param), param_name(layer, param, true), c, t);
    };
    const bool post = c.norm_position == "post", sandwich = c.norm_position == "sandwich";
    hook(hooks, hook_name(layer, kHookResidPre), x);

    Tensor attn_in = post ? x : norm_(kParamLn1, x);
    hook(hooks, hook_name(layer, kHookAttnIn), attn_in);
    Tensor attn_out = attention(layer, attn_in, hooks, batch);
    if (sandwich) attn_out = norm_(kParamLn1Post, attn_out);
    hook(hooks, hook_name(layer, kHookAttnOut), attn_out);

    auto run_mlp = [&](Tensor mlp_in) {
        hook(hooks, hook_name(layer, kHookMlpIn), mlp_in);
        Tensor out = mlp(layer, mlp_in, hooks, batch);
        if (sandwich) out = norm_(kParamLn2Post, out);
        hook(hooks, hook_name(layer, kHookMlpOut), out);
        return out;
    };

    if (c.parallel_residual) {
        Tensor mid = add(x, attn_out);
        hook(hooks, hook_name(layer, kHookResidMid), mid);
        Tensor out = c.has_mlp() ? add(mid, run_mlp(norm_(kParamLn2, x))) : mid;
        hook(hooks, hook_name(layer, kHookResidPost), out);
        return out;
    }

    add_(x, attn_out);
    if (post) x = norm_(kParamLn1, x);
    hook(hooks, hook_name(layer, kHookResidMid), x);
    if (c.has_mlp()) {
        add_(x, run_mlp(post ? x : norm_(kParamLn2, x)));
        if (post) x = norm_(kParamLn2, x);
    }
    hook(hooks, hook_name(layer, kHookResidPost), x);
    return x;
}

Tensor Model::final_norm(const Tensor& resid) const {
    if (cfg_.norm_position == "post") return resid;  // post-norm blocks already end in a norm
    return norm("ln_final", cfg_, resid);
}

Tensor Model::logits_from_final(const Tensor& normed) const { return logits_from_final(normed, nullptr); }

Tensor Model::logits_from_final(const Tensor& normed, const detail::RowBatch* batch) const {
    Tensor logits;
#ifdef TFI_USE_ACCELERATE
    if (detail::unembed_on_gpu(batch, normed))
        logits = batch->gpu->linear(normed, unembed(), nullptr, batch->fp16);
    else
#else
    (void)batch;
#endif
        logits = linear(normed, unembed());
    if (cfg_.final_logit_softcap) {
        const float cap = static_cast<float>(*cfg_.final_logit_softcap);
        for (float& v : logits.data) v = cap * vm::tanh(v / cap);
    }
    return logits;
}

Tensor Model::forward(const Tokens& tokens, const Hooks* hooks) const {
    return forward_from(0, embed(tokens, hooks), hooks);
}

Tensor Model::embed(const Tokens& tokens, const Hooks* hooks, int64_t start) const {
    if (start < 0) throw std::invalid_argument("embed: negative start position");
    std::vector<int64_t> pos(tokens.size());
    for (size_t t = 0; t < tokens.size(); ++t) pos[t] = start + static_cast<int64_t>(t);
    return embed_at(tokens, pos, hooks);
}

Tensor Model::embed_at(const Tokens& tokens, const std::vector<int64_t>& positions, const Hooks* hooks) const {
    const int64_t T = std::ssize(tokens), d = cfg_.d_model;
    if (T == 0) throw std::invalid_argument("forward needs at least one token");
    const Weight& E = weight("embed.weight");
    Tensor x({T, d});
    for (int64_t t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= cfg_.vocab_size)
            throw std::out_of_range(std::format("token {} is outside the vocabulary", tokens[t]));
        E.read_rows(tokens[t], tokens[t] + 1, x.ptr() + t * d);
    }
    if (cfg_.embed_scale) x = scale(x, static_cast<float>(std::sqrt(double(d))));
    hook(hooks, "hook_embed", x);
    if (cfg_.embedding_positional()) {
        Tensor pe = positional_embedding_at(positions);
        hook(hooks, "hook_pos_embed", pe);
        add_(x, pe);
    }
    return x;
}

Tensor Model::forward_from(int64_t layer, Tensor x, const Hooks* hooks) const {
    return run_from(layer, std::move(x), hooks, false);
}

namespace {
Tensor last_row(const Tensor& x) {
    const int64_t d = x.shape[1];
    return Tensor({1, d}, std::vector<float>(x.ptr() + (x.shape[0] - 1) * d, x.ptr() + x.shape[0] * d));
}
}  // namespace

Tensor Model::run_from(int64_t layer, Tensor x, const Hooks* hooks, bool last_only) const {
    if (layer < 0 || layer > cfg_.n_layers) throw std::out_of_range("forward_from: layer out of range");
    if (x.dim() != 2 || x.shape[1] != cfg_.d_model)
        throw std::invalid_argument("forward_from: residual must be (T, d_model), got " + shape_str(x.shape));
    detail::RowBatch rb;  // the GPU's, if this pass goes to it: one whole sequence
#ifdef TFI_USE_ACCELERATE
    rb.gpu = layer < cfg_.n_layers ? detail::single_pass_gpu(*this, layer, x.shape[0]) : nullptr;
    if (rb.gpu) {
        rb.whole = true;
        rb.fp16 = format_ == WeightFormat::fp16;  // fp16 weights: fp16 multiplies on the GPU
        rb.add(0, x.shape[0]);
        x = run_rows(layer, std::move(x), rb, hooks);
    }
#endif
    if (!rb.gpu)
        for (int64_t i = layer; i < cfg_.n_layers; ++i) x = block(i, std::move(x), hooks);
    Tensor f = final_norm(x);
    hook(hooks, "hook_final", f);
    if (last_only) f = last_row(f);
    return logits_from_final(f, rb.gpu ? &rb : nullptr);
}

namespace {
// Keys or values rounded to fp16 values in place, clamped to fp16's finite range: what a KV cache kept in
// fp16 holds (as the GPU's round_half does).
void round_to_half_(Tensor& t) {
    for (float& v : t.data) v = detail::half_to_float(detail::float_to_half(std::clamp(v, -65504.f, 65504.f)));
}
// A hook rounding every layer's hook_k and hook_v so (on the CPU, after the caller's hooks).
// Which layer's keys or values a hook point name is ("blocks.<l>.attn.hook_k" / "hook_v"), read from the
// name without building strings (every pass of a cached step asks at each hook point); layer -1: neither.
struct KVHook {
    int64_t layer = -1;
    bool value = false;
};
KVHook kv_hook(std::string_view n) {
    constexpr std::string_view head = "blocks.", tail = ".attn.hook_";
    if (n.size() < head.size() + 1 + tail.size() + 1 || (n.back() != 'k' && n.back() != 'v')) return {};
    const bool value = n.back() == 'v';
    n.remove_suffix(1);  // "blocks.<l>.attn.hook_"
    if (!n.starts_with(head) || !n.ends_with(tail)) return {};
    int64_t layer = 0;
    for (char d : n.substr(head.size(), n.size() - head.size() - tail.size())) {
        if (d < '0' || d > '9' || layer > (int64_t(1) << 40)) return {};
        layer = layer * 10 + (d - '0');
    }
    return {layer, value};
}

void add_kv_rounding(Hooks& h, int64_t layers) {
    h.add_filter(
        [layers](const std::string& n) {
            const int64_t l = kv_hook(n).layer;
            return l >= 0 && l < layers;
        },
        [](const std::string&, Tensor& a) { round_to_half_(a); });
}
#ifdef TFI_USE_ACCELERATE
// Floats a KV slot needs for n values: n, or for fp16 values half that.
int64_t kv_floats(bool half, int64_t n) { return half ? (n + 1) / 2 : n; }
// Positions [from, to) of a CPU cache layer (H, c, D: float32 f, or for an fp16 cache fp16 bits h) into GPU slot
// `slot` (H, Tp, D) at position base + j, copied as they are (both copies hold the cache's type).
void upload_rows(detail::GpuKV& kv, int slot, const float* f, const uint16_t* bits, int64_t H, int64_t c, int64_t D,
                 int64_t Tp, int64_t base, int64_t from, int64_t to) {
    const int64_t n = (to - from) * D;
    for (int64_t h = 0; h < H; ++h) {
        const int64_t src = (h * c + from) * D, at = (h * Tp + base + from) * D;
        if (kv.half) std::copy_n(bits + src, n, reinterpret_cast<uint16_t*>(kv.seq->data(slot)) + at);
        else std::copy_n(f + src, n, kv.seq->data(slot) + at);
    }
}
#endif
}  // namespace

KVCache::KVCache(const KVCache& o) {
    o.sync_cpu();
    k_ = o.k_;
    v_ = o.v_;
    len_ = cpu_len_ = o.len_;
    half_ = o.half_;
}

KVCache& KVCache::operator=(const KVCache& o) {
    if (this == &o) return *this;
    o.sync_cpu();
    k_ = o.k_;
    v_ = o.v_;
    len_ = cpu_len_ = o.len_;
    half_ = o.half_;
    gpu_.reset();
    region_ = gpu_len_ = 0;
    return *this;
}

void KVCache::clear() {
    k_.clear();
    v_.clear();
    len_ = cpu_len_ = gpu_len_ = 0;  // the GPU's buffers stay, for the next sequence
}

void KVCache::truncate(int64_t n) {
    if (n < 0 || n > len_) throw std::out_of_range(std::format("KVCache::truncate: {} is not in 0..{}", n, len_));
    if (n == 0) return clear();
    len_ = n;  // later positions are overwritten as the sequence continues, on either side
    cpu_len_ = std::min(cpu_len_, n);
    gpu_len_ = std::min(gpu_len_, n);
}

void KVCache::shapes(const std::vector<int64_t>& heads, const std::vector<int64_t>& dims) {
    if (!k_.empty()) return;
    for (size_t l = 0; l < heads.size(); ++l) {
        k_.push_back(Store{.H = heads[l], .D = dims[l]});
        v_.push_back(Store{.H = heads[l], .D = dims[l]});
    }
}

void KVCache::reserve_cpu(int64_t n) const {
    for (int pass = 0; pass < 2; ++pass)
        for (Store& t : pass ? v_ : k_) {
            if (n <= t.cap) continue;
            // geometrically, with room after a prompt, so that appending one position at a time stays linear
            const int64_t grown = std::max({n + std::max<int64_t>(16, n / 2), 2 * t.cap});
            auto regrow = [&](auto& old) {
                std::remove_reference_t<decltype(old)> g(static_cast<size_t>(t.H * grown * t.D));
                for (int64_t h = 0; h < t.H; ++h)
                    std::copy(old.begin() + h * t.cap * t.D, old.begin() + (h * t.cap + cpu_len_) * t.D,
                              g.begin() + h * grown * t.D);
                old = std::move(g);
            };
            if (half_) regrow(t.h);
            else regrow(t.f);
            t.cap = grown;
        }
}

void KVCache::put(Store& s, const float* src, int64_t n, int64_t at) const {
    for (int64_t h = 0; h < s.H; ++h) {
        const float* from = src + h * n * s.D;
        const int64_t to = (h * s.cap + at) * s.D;
        if (half_) detail::floats_to_halves(from, s.h.data() + to, n * s.D);  // fp16 values already (add_kv_rounding)
        else std::copy_n(from, n * s.D, s.f.data() + to);
    }
}

detail::KVRef KVCache::ref(int64_t layer, bool values) const {
    const Store& s = (values ? v_ : k_).at(static_cast<size_t>(layer));
    detail::KVRef r;
    if (half_) r.h = s.h.data();
    else r.f = s.f.data();
    r.H = s.H;
    r.T = s.cap;
    r.D = s.D;
    return r;
}

int64_t KVCache::clamped() const {
    if (!half_ || len_ == 0) return 0;
    sync_cpu();
    int64_t n = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (const Store& s : pass ? v_ : k_)
            for (int64_t h = 0; h < s.H; ++h) {
                const uint16_t* p = s.h.data() + h * s.cap * s.D;
                for (int64_t i = 0; i < len_ * s.D; ++i) n += (p[i] & 0x7fff) == 0x7bff;  // +-65504
            }
    return n;
}

void KVCache::sync_cpu() const {
    if (cpu_len_ >= len_) return;
#ifdef TFI_USE_ACCELERATE
    if (!gpu_ || gpu_len_ < len_) throw std::logic_error("KVCache: positions held on neither side");
    reserve_cpu(len_);
    const int64_t Tp = gpu_->cap * gpu_->regions, base = region_ * gpu_->cap;
    auto gpu_halves = [&](int slot) { return reinterpret_cast<const uint16_t*>(gpu_->seq->data(slot)); };
    for (size_t l = 0; l < k_.size(); ++l) {
        const int64_t H = k_[l].H, c = k_[l].cap, D = k_[l].D;
        for (int64_t h = 0; h < H; ++h) {
            const int64_t from = (h * Tp + base + cpu_len_) * D, n = (len_ - cpu_len_) * D, to = (h * c + cpu_len_) * D;
            if (gpu_->half) {  // fp16 on both sides: copied as they are
                const uint16_t* gk = gpu_halves(detail::GpuKV::keys(static_cast<int64_t>(l))) + from;
                const uint16_t* gv = gpu_halves(detail::GpuKV::values(static_cast<int64_t>(l))) + from;
                std::copy_n(gk, n, k_[l].h.data() + to);
                std::copy_n(gv, n, v_[l].h.data() + to);
            } else {
                float* dk = k_[l].f.data() + to;
                float* dv = v_[l].f.data() + to;
                const float* gk = gpu_->seq->data(detail::GpuKV::keys(static_cast<int64_t>(l)));
                const float* gv = gpu_->seq->data(detail::GpuKV::values(static_cast<int64_t>(l)));
                std::copy_n(gk + from, n, dk);
                std::copy_n(gv + from, n, dv);
            }
        }
    }
    cpu_len_ = len_;
#else
    throw std::logic_error("KVCache: positions held on neither side");
#endif
}

namespace {
std::atomic<int> g_check_kv{-1};  // -1: TFI_CHECK_KV decides
}  // namespace

void detail::set_check_kv(bool on) { g_check_kv = on ? 1 : 0; }
void detail::set_kv_side_by_side(int64_t values) { g_side_by_side = values; }

void KVCache::check() const {
    if (g_check_kv.load() < 0) {
        const char* e = std::getenv("TFI_CHECK_KV");
        g_check_kv = e && *e && std::string(e) != "0" ? 1 : 0;
    }
    if (g_check_kv.load() == 0) return;
    if (cpu_len_ > len_ || gpu_len_ > len_ || std::max(cpu_len_, gpu_len_) != len_)
        throw std::logic_error(
            std::format("KVCache: up-to-date prefixes {} (CPU), {} (GPU) for {} positions", cpu_len_, gpu_len_, len_));
#ifdef TFI_USE_ACCELERATE
    const int64_t both = std::min(cpu_len_, gpu_len_);
    if (!gpu_ || both == 0) return;
    const int64_t Tp = gpu_->cap * gpu_->regions, base = region_ * gpu_->cap;
    auto gpu_halves = [&](int slot) { return reinterpret_cast<const uint16_t*>(gpu_->seq->data(slot)); };
    for (size_t l = 0; l < k_.size(); ++l) {
        const int64_t H = k_[l].H, c = k_[l].cap, D = k_[l].D;
        for (int64_t h = 0; h < H; ++h) {
            const int64_t from = (h * Tp + base) * D, to = h * c * D, n = both * D;
            bool same;
            if (gpu_->half) {  // bit for bit
                const uint16_t* gk = gpu_halves(detail::GpuKV::keys(static_cast<int64_t>(l))) + from;
                const uint16_t* gv = gpu_halves(detail::GpuKV::values(static_cast<int64_t>(l))) + from;
                same = std::equal(gk, gk + n, k_[l].h.data() + to) && std::equal(gv, gv + n, v_[l].h.data() + to);
            } else {
                const float* gk = gpu_->seq->data(detail::GpuKV::keys(static_cast<int64_t>(l))) + from;
                const float* gv = gpu_->seq->data(detail::GpuKV::values(static_cast<int64_t>(l))) + from;
                same = std::equal(gk, gk + n, k_[l].f.data() + to) && std::equal(gv, gv + n, v_[l].f.data() + to);
            }
            if (!same) throw std::logic_error(std::format("KVCache: the CPU and GPU copies differ in layer {}", l));
        }
    }
#endif
}

Tensor KVCache::keys(int64_t layer) const {
    sync_cpu();
    if (layer < 0 || layer >= std::ssize(k_))
        throw std::out_of_range(
            std::format("KVCache: no layer {}{}", layer, (k_.empty() ? " (the cache is empty)" : "")));
    const detail::KVRef r = ref(layer, false);
    Tensor out({r.H, len_, r.D});
    for (int64_t h = 0; h < r.H; ++h) {  // positions [0, len_) of each head, as float32 (fp16 values widened exactly)
        if (r.h) detail::halves_to_floats(r.h + h * r.T * r.D, out.ptr() + h * len_ * r.D, len_ * r.D);
        else std::copy(r.f + h * r.T * r.D, r.f + (h * r.T + len_) * r.D, out.ptr() + h * len_ * r.D);
    }
    return out;
}
Tensor KVCache::values(int64_t layer) const {
    sync_cpu();
    if (layer < 0 || layer >= std::ssize(v_))
        throw std::out_of_range(
            std::format("KVCache: no layer {}{}", layer, (v_.empty() ? " (the cache is empty)" : "")));
    const detail::KVRef r = ref(layer, true);
    Tensor out({r.H, len_, r.D});
    for (int64_t h = 0; h < r.H; ++h) {
        if (r.h) detail::halves_to_floats(r.h + h * r.T * r.D, out.ptr() + h * len_ * r.D, len_ * r.D);
        else std::copy(r.f + h * r.T * r.D, r.f + (h * r.T + len_) * r.D, out.ptr() + h * len_ * r.D);
    }
    return out;
}

void KVCache::append(const std::vector<Tensor>& k, const std::vector<Tensor>& v, int64_t n) {
    if (cpu_len_ != len_) throw std::logic_error("KVCache::append: the CPU side is not up to date");
    if (k_.empty())
        for (size_t l = 0; l < k.size(); ++l) {
            k_.push_back(Store{.H = k[l].shape[0], .D = k[l].shape[2]});
            v_.push_back(Store{.H = v[l].shape[0], .D = v[l].shape[2]});
        }
    reserve_cpu(len_ + n);
    for (size_t l = 0; l < k.size(); ++l) {
        put(k_[l], k[l].ptr(), n, len_);
        put(v_[l], v[l].ptr(), n, len_);
    }
    len_ += n;
    cpu_len_ = len_;
}

void Model::check_cache(const KVCache& cache) const {
    const int64_t L = cfg_.n_layers;
    if (cache.len_) {  // the cache must be this model's: a tensor per layer, each with the layer's heads
        bool ok = std::ssize(cache.k_) == L;
        for (int64_t l = 0; ok && l < L; ++l)
            ok = cache.k_[l].H == layer_cfgs_[l].kv_heads() && cache.k_[l].D == layer_cfgs_[l].head_dim();
        if (!ok) throw std::invalid_argument("forward_cached: the cache holds another model's keys and values");
    }
    // fp16 values for a model with fp16 weights; a cache keeps the precision of its first step.
    const bool half = format_ == WeightFormat::fp16;
    if ((cache.len_ || !cache.k_.empty()) && cache.half_ != half)
        throw std::invalid_argument(
            std::format("forward_cached: the cache holds {} keys and values; this model's are {}",
                        (cache.half_ ? "fp16" : "float32"), (half ? "fp16" : "float32")));
}

void Model::use_gpu_cache(KVCache& cache, detail::RowBatch& rb, int64_t T, int64_t need, int64_t upload_to,
                          const Hooks* hooks) const {
    (void)cache, (void)rb, (void)T, (void)need, (void)upload_to, (void)hooks;
#ifdef TFI_USE_ACCELERATE
    const int64_t L = cfg_.n_layers;
    const bool half = format_ == WeightFormat::fp16;
    // The GPU, with the cache kept there (detail::GpuKV), for: the prompt when a single pass of it would
    // go to the GPU (its keys and values then go straight into the GPU's copy); later chunks of tokens big
    // enough; single tokens of models large enough that reading their weights takes the CPU longer than
    // the GPU's pass (kDecodeWeights); and every step in GPU-only mode. Not when a hook reads the
    // pattern, which only the CPU's cached attention (and a whole pass) hands over, or when a layer's heads
    // are too wide for the GPU's attention.
    bool pattern_hook = false;
    for (int64_t l = 0; l < L && hooks; ++l)
        pattern_hook = pattern_hook || hooks->matches(layer_prefix(l) + "attn.hook_pattern");
    detail::Gpu* g = pattern_hook ? nullptr : detail::cached_pass_gpu(*this, T, cache.len_ == 0);
    if (!g) return;
    if (!cache.gpu_ || cache.gpu_->gpu.get() != g || cache.gpu_->regions != 1) {  // first use, or another model's GPU
        cache.sync_cpu();  // the old GPU copy goes: the CPU must hold everything first
        auto kv = std::make_shared<detail::GpuKV>();
        kv->gpu = gpu_.gpu;
        kv->seq = std::make_unique<detail::Gpu::Seq>(*g);
        kv->half = half;
        cache.gpu_ = std::move(kv);
        cache.region_ = 0;
        cache.gpu_len_ = 0;
    }
    detail::GpuKV& kv = *cache.gpu_;
    if (need > kv.cap) {  // grow, which loses the buffers' contents: the CPU must hold everything first
        cache.sync_cpu();
        kv.cap = std::max({need + std::max<int64_t>(256, need / 2), 2 * kv.cap});
        for (int64_t l = 0; l < L; ++l) {
            const int64_t n = kv_floats(half, layer_cfgs_[l].kv_heads() * kv.cap * layer_cfgs_[l].head_dim());
            kv.seq->reserve(detail::GpuKV::keys(l), n);
            kv.seq->reserve(detail::GpuKV::values(l), n);
        }
        cache.gpu_len_ = 0;
    }
    // Positions only the CPU holds (CPU steps, a copy, the buffers growing): up to the GPU. The CPU
    // holds them all whenever the GPU lacks any (the invariant).
    for (int64_t l = 0; l < L && cache.gpu_len_ < upload_to; ++l) {
        const KVCache::Store &ck = cache.k_[l], &cv = cache.v_[l];
        upload_rows(kv, detail::GpuKV::keys(l), ck.f.data(), ck.h.data(), ck.H, ck.cap, ck.D, kv.cap, 0, cache.gpu_len_,
                    upload_to);
        upload_rows(kv, detail::GpuKV::values(l), cv.f.data(), cv.h.data(), cv.H, cv.cap, cv.D, kv.cap, 0,
                    cache.gpu_len_, upload_to);
    }
    cache.gpu_len_ = std::max(cache.gpu_len_, upload_to);
    rb.gpu = g;
    rb.fp16 = format_ == WeightFormat::fp16;  // fp16 weights: fp16 multiplies on the GPU
    rb.kv = &kv;
#endif
}

Tensor Model::forward_cached(const Tokens& tokens, KVCache& cache, const Hooks* hooks, bool last_only) const {
    const int64_t L = cfg_.n_layers, T = std::ssize(tokens), start = cache.len_;
    if (T == 0) throw std::invalid_argument("forward_cached needs at least one token");
    check_cache(cache);
    const bool half = format_ == WeightFormat::fp16;
    cache.half_ = half;
    detail::RowBatch rb;
    rb.add(start, T);
    rb.cached = true;
    use_gpu_cache(cache, rb, T, start + T, start, hooks);
    const bool on_gpu = rb.kv != nullptr;  // the new positions go to the GPU's copy only
    if (!on_gpu && start) {
        cache.sync_cpu();
        for (int64_t l = 0; l < L; ++l) {
            rb.prefix_k.push_back(cache.ref(l, false));
            rb.prefix_v.push_back(cache.ref(l, true));
        }
    }
    // On the CPU: the new positions' keys and values, as the pass leaves them after any hooks, appended at the end.
    std::vector<Tensor> new_k(L), new_v(L);
    auto slot = [&](const std::string& n) -> Tensor* {
        const KVHook kv = kv_hook(n);
        return kv.layer < 0 || kv.layer >= L ? nullptr : kv.value ? &new_v[kv.layer] : &new_k[kv.layer];
    };
    Hooks h;
    if (hooks) h.extend(*hooks);
    if (half && !on_gpu) add_kv_rounding(h, L);  // (the GPU's pass rounds them itself)
    if (!on_gpu)
        h.add_capture([&slot](const std::string& n) { return slot(n) != nullptr; },
                      [&slot](const std::string& n, Tensor& act) { *slot(n) = act; });

    Tensor logits;
    if (start == 0 && !on_gpu) {
        logits = run_from(0, embed(tokens, &h), &h, last_only);  // the CPU, or a whole pass on the GPU
    } else if (on_gpu) {
        // Layers, final norm and unembedding in one GPU submission; the keys and values appended there.
        run_rows(0, embed(tokens, &h, start), rb, &h, false, &logits, last_only);
        if (cfg_.final_logit_softcap) {
            const float cap = static_cast<float>(*cfg_.final_logit_softcap);
            for (float& v : logits.data) v = cap * vm::tanh(v / cap);
        }
    } else {
        Tensor x = run_rows(0, embed(tokens, &h, start), rb, &h);
        Tensor f = final_norm(x);
        hook(&h, "hook_final", f);
        if (last_only) f = last_row(f);
        logits = logits_from_final(f);
    }
    if (on_gpu) {
        std::vector<int64_t> heads, dims;
        for (int64_t l = 0; l < L; ++l) {
            heads.push_back(layer_cfgs_[l].kv_heads());
            dims.push_back(layer_cfgs_[l].head_dim());
        }
        cache.shapes(heads, dims);
        cache.len_ = cache.gpu_len_ = start + T;  // the CPU side catches up only when something reads it
    } else {
        for (int64_t l = 0; l < L; ++l) {
            const Shape want{layer_cfgs_[l].kv_heads(), T, layer_cfgs_[l].head_dim()};
            if (new_k[l].shape != want || new_v[l].shape != want)
                throw std::runtime_error(std::format("forward_cached: {}attn's keys or values are not {}",
                                                     layer_prefix(l), shape_str(want)));
        }
        cache.append(new_k, new_v, T);
    }
    cache.check();
    return logits;
}

Tensor Model::rerun_final(int64_t pos, const Tensor& final_out, const KVCache& cache) const {
    if (pos < 0 || pos >= cache.len_)
        throw std::invalid_argument(std::format("rerun_final: position {} is not in the cache", pos));
    if (final_out.shape != Shape{1, cfg_.d_model})
        throw std::invalid_argument("rerun_final: final_out must be (1, d_model)");
    check_cache(cache);
    detail::RowBatch rb;
#ifdef TFI_USE_ACCELERATE
    if (cache.gpu_ && cache.gpu_len_ > pos) {
        rb.gpu = cache.gpu_->gpu.get();
        rb.fp16 = format_ == WeightFormat::fp16;  // fp16 weights: fp16 multiplies on the GPU
        rb.whole = true;                          // on the GPU whatever the rows, as the pass was
    }
#endif
    return logits_from_final(final_out, rb.gpu ? &rb : nullptr);
}

Tensor Model::rerun_cached(int64_t layer, int64_t pos, Tensor resid, KVCache& cache, const Hooks* hooks) const {
    const int64_t L = cfg_.n_layers, d = cfg_.d_model;
    if (layer < 0 || layer >= L)
        throw std::invalid_argument(std::format("rerun_cached: layer {} is not in 0..{}", layer, L - 1));
    if (pos < 0 || pos >= cache.len_)
        throw std::invalid_argument(std::format("rerun_cached: position {} is not in the cache", pos));
    if (resid.shape != Shape{1, d}) throw std::invalid_argument("rerun_cached: resid must be (1, d_model)");
    check_cache(cache);
    const bool half = format_ == WeightFormat::fp16;
    detail::RowBatch rb;
    rb.add(pos, 1);
    rb.cached = true;
    // Every position on the GPU (the blocks before `layer` read their own at pos there), or every one on the CPU.
    use_gpu_cache(cache, rb, 1, cache.len_, cache.len_, hooks);
    const bool on_gpu = rb.kv != nullptr;
    if (!on_gpu) {
        cache.sync_cpu();
        for (int64_t l = 0; l < L; ++l) {
            rb.prefix_k.push_back(cache.ref(l, false));
            rb.prefix_v.push_back(cache.ref(l, true));
        }
    }
    std::vector<Tensor> new_k(L), new_v(L);
    auto slot = [&](const std::string& n) -> Tensor* {
        const KVHook kv = kv_hook(n);
        return kv.layer < layer || kv.layer >= L ? nullptr : kv.value ? &new_v[kv.layer] : &new_k[kv.layer];
    };
    Hooks h;
    if (hooks) h.extend(*hooks);
    if (half && !on_gpu) add_kv_rounding(h, L);
    if (!on_gpu)
        h.add_capture([&slot](const std::string& n) { return slot(n) != nullptr; },
                      [&slot](const std::string& n, Tensor& act) { *slot(n) = act; });
    Tensor logits;
    if (on_gpu) {
        run_rows(layer, std::move(resid), rb, &h, false, &logits, true);
        if (cfg_.final_logit_softcap) {
            const float cap = static_cast<float>(*cfg_.final_logit_softcap);
            for (float& v : logits.data) v = cap * vm::tanh(v / cap);
        }
        cache.cpu_len_ = std::min(cache.cpu_len_, pos);  // the GPU's copy has the new values at pos
    } else {
        Tensor f = final_norm(run_rows(layer, std::move(resid), rb, &h));
        hook(&h, "hook_final", f);
        logits = logits_from_final(f);
        for (int64_t l = layer; l < L; ++l) {
            const Shape want{layer_cfgs_[l].kv_heads(), 1, layer_cfgs_[l].head_dim()};
            if (new_k[l].shape != want || new_v[l].shape != want)
                throw std::runtime_error(std::format("forward_cached: {}attn's keys or values are not {}",
                                                     layer_prefix(l), shape_str(want)));
        }
        for (int64_t l = layer; l < L; ++l)
            for (int pass = 0; pass < 2; ++pass)
                cache.put(pass ? cache.v_[l] : cache.k_[l], (pass ? new_v[l] : new_k[l]).ptr(), 1, pos);
        cache.gpu_len_ = std::min(cache.gpu_len_, pos);  // the CPU's copy has the new values at pos
    }
    cache.check();
    return logits;
}

std::vector<Tensor> Model::forward_cached(const std::vector<Tokens>& tokens, KVBatch& kb, const Hooks* hooks,
                                          bool last_only) const {
    const int64_t L = cfg_.n_layers, n_seq = kb.size();
    if (std::ssize(tokens) != n_seq)
        throw std::invalid_argument(
            std::format("forward_cached: {} token lists for a batch of {}", tokens.size(), n_seq));
    for (int64_t l = 0; l < L && hooks; ++l)
        if (hooks->matches(layer_prefix(l) + "attn.hook_pattern"))
            throw std::invalid_argument(
                "forward_cached: hook_pattern is not available in a batched pass (each sequence has its own)");
    // The sequences given tokens, as the batch's segments, rows stacked in order.
    std::vector<int64_t> seq_of;
    detail::RowBatch rb;
    Tokens all;
    std::vector<int64_t> positions;
    for (int64_t i = 0; i < n_seq; ++i) {
        if (tokens[i].empty()) continue;
        const KVCache& c = kb.caches_[i];
        if (c.len_) {
            bool ok = std::ssize(c.k_) == L;
            for (int64_t l = 0; ok && l < L; ++l)
                ok = c.k_[l].H == layer_cfgs_[l].kv_heads() && c.k_[l].D == layer_cfgs_[l].head_dim();
            if (!ok) throw std::invalid_argument("forward_cached: the batch holds another model's keys and values");
        }
        const int64_t T = std::ssize(tokens[i]);
        seq_of.push_back(i);
        rb.add(c.len_, T);
        all.insert(all.end(), tokens[i].begin(), tokens[i].end());
        for (int64_t t = 0; t < T; ++t) positions.push_back(c.len_ + t);
    }
    std::vector<Tensor> out(static_cast<size_t>(n_seq));
    if (seq_of.empty()) return out;
    const int64_t S = std::ssize(seq_of), R = rb.rows();
    const bool half = format_ == WeightFormat::fp16;  // as for one cache
    for (int64_t i : seq_of) {
        KVCache& c = kb.caches_[i];
        if ((c.len_ || !c.k_.empty()) && c.half_ != half)
            throw std::invalid_argument(
                std::format("forward_cached: the batch holds {} keys and values; this model's are {}",
                            (c.half_ ? "fp16" : "float32"), (half ? "fp16" : "float32")));
        c.half_ = half;
    }

#ifdef TFI_USE_ACCELERATE
    // The GPU under forward_cached's rules (rows big enough, models large enough, GPU-only mode), with every
    // sequence's keys and values kept there, a region each (each cache's GPU copy is its region).
    detail::Gpu* g = detail::cached_pass_gpu(*this, R, false);
    if (g) {
        int64_t need = 0;
        for (int64_t sg = 0; sg < S; ++sg) need = std::max(need, rb.start[sg] + rb.len[sg]);
        const bool fresh = !kb.gpu_ || kb.gpu_->gpu.get() != g || kb.gpu_->regions != n_seq || kb.gpu_->half != half;
        if (fresh || need > kb.gpu_->cap) {
            // A new or grown GPU copy loses what the old one held: every sequence's CPU side catches up first.
            for (const KVCache& c : kb.caches_) c.sync_cpu();
            if (fresh) {
                auto kv = std::make_shared<detail::GpuKV>();
                kv->gpu = gpu_.gpu;
                kv->seq = std::make_unique<detail::Gpu::Seq>(*g);
                kv->regions = n_seq;
                kv->half = half;
                kb.gpu_ = std::move(kv);
            }
            detail::GpuKV& kv = *kb.gpu_;
            if (need > kv.cap) {
                kv.cap = std::max({need + std::max<int64_t>(256, need / 2), 2 * kv.cap});
                for (int64_t l = 0; l < L; ++l) {
                    const int64_t n =
                        kv_floats(half, layer_cfgs_[l].kv_heads() * kv.cap * n_seq * layer_cfgs_[l].head_dim());
                    kv.seq->reserve(detail::GpuKV::keys(l), n);
                    kv.seq->reserve(detail::GpuKV::values(l), n);
                }
            }
            for (int64_t i = 0; i < n_seq; ++i) {
                KVCache& c = kb.caches_[i];
                c.gpu_ = kb.gpu_;
                c.region_ = i;
                c.gpu_len_ = 0;
            }
        }
        detail::GpuKV& kv = *kb.gpu_;
        const int64_t Tp = kv.cap * n_seq;
        for (int64_t sg = 0; sg < S; ++sg) {  // positions only the CPU holds, per sequence, up to its region
            const int64_t i = seq_of[sg], start = rb.start[sg];
            KVCache& c = kb.caches_[i];
            for (int64_t l = 0; l < L && c.gpu_len_ < start; ++l) {
                const KVCache::Store &ck = c.k_[l], &cv = c.v_[l];
                upload_rows(kv, detail::GpuKV::keys(l), ck.f.data(), ck.h.data(), ck.H, ck.cap, ck.D, Tp, i * kv.cap,
                            c.gpu_len_, start);
                upload_rows(kv, detail::GpuKV::values(l), cv.f.data(), cv.h.data(), cv.H, cv.cap, cv.D, Tp, i * kv.cap,
                            c.gpu_len_, start);
            }
            c.gpu_len_ = start;
            rb.kbase.push_back(i * kv.cap);
        }
        rb.gpu = g;
        rb.fp16 = format_ == WeightFormat::fp16;  // fp16 weights: fp16 multiplies on the GPU
        rb.kv = &kv;
    }
#endif
    const bool on_gpu = rb.kv != nullptr;  // the new positions go to the GPU's copy only
    if (!on_gpu)
        for (int64_t sg = 0; sg < S; ++sg) {  // each sequence's CPU side, up to date, as its prefix
            const KVCache& c = kb.caches_[seq_of[sg]];
            c.sync_cpu();
            std::vector<detail::KVRef> pk, pv;
            for (int64_t l = 0; l < L && c.len_; ++l) {
                pk.push_back(c.ref(l, false));
                pv.push_back(c.ref(l, true));
            }
            rb.seg_prefix_k.push_back(std::move(pk));
            rb.seg_prefix_v.push_back(std::move(pv));
        }
    // On the CPU: the new positions' keys and values, (Hkv, R, D) per layer, after any hooks, split per sequence at the
    // end.
    std::vector<Tensor> new_k(L), new_v(L);
    auto slot = [&](const std::string& n) -> Tensor* {
        const KVHook kv = kv_hook(n);
        return kv.layer < 0 || kv.layer >= L ? nullptr : kv.value ? &new_v[kv.layer] : &new_k[kv.layer];
    };
    Hooks h;
    if (hooks) h.extend(*hooks);
    if (half && !on_gpu) add_kv_rounding(h, L);  // (the GPU's pass rounds them itself)
    if (!on_gpu)
        h.add_capture([&slot](const std::string& n) { return slot(n) != nullptr; },
                      [&slot](const std::string& n, Tensor& act) { *slot(n) = act; });

    // Each sequence's rows of a (rows, width) tensor, (len, width).
    auto seg_rows = [&](const Tensor& t, int64_t sg, bool last) {
        const int64_t w = t.shape[1], r0 = rb.offset[sg] + (last ? rb.len[sg] - 1 : 0), n = last ? 1 : rb.len[sg];
        return Tensor({n, w}, std::vector<float>(t.ptr() + r0 * w, t.ptr() + (r0 + n) * w));
    };
    Tensor x = embed_at(all, positions, &h);
    if (on_gpu) {
        Tensor logits;  // each sequence's last row, or every row
        run_rows(0, std::move(x), rb, &h, false, &logits, last_only);
        if (cfg_.final_logit_softcap) {
            const float cap = static_cast<float>(*cfg_.final_logit_softcap);
            for (float& v : logits.data) v = cap * vm::tanh(v / cap);
        }
        for (int64_t sg = 0; sg < S; ++sg) {
            const int64_t V = logits.shape[1], r0 = last_only ? sg : rb.offset[sg], n = last_only ? 1 : rb.len[sg];
            out[seq_of[sg]] = Tensor({n, V}, std::vector<float>(logits.ptr() + r0 * V, logits.ptr() + (r0 + n) * V));
        }
        std::vector<int64_t> heads, dims;
        for (int64_t l = 0; l < L; ++l) {
            heads.push_back(layer_cfgs_[l].kv_heads());
            dims.push_back(layer_cfgs_[l].head_dim());
        }
        for (int64_t sg = 0; sg < S;
             ++sg) {  // appended on the GPU; the CPU side catches up only when something reads it
            KVCache& c = kb.caches_[seq_of[sg]];
            c.shapes(heads, dims);
            c.len_ = c.gpu_len_ = rb.start[sg] + rb.len[sg];
        }
    } else {
        x = run_rows(0, std::move(x), rb, &h);
        Tensor f = final_norm(x);
        hook(&h, "hook_final", f);
        for (int64_t sg = 0; sg < S; ++sg) out[seq_of[sg]] = logits_from_final(seg_rows(f, sg, last_only));
        for (int64_t l = 0; l < L; ++l) {
            const Shape want{layer_cfgs_[l].kv_heads(), R, layer_cfgs_[l].head_dim()};
            if (new_k[l].shape != want || new_v[l].shape != want)
                throw std::runtime_error(std::format("forward_cached: {}attn's keys or values are not {}",
                                                     layer_prefix(l), shape_str(want)));
        }
        for (int64_t sg = 0; sg < S; ++sg) {
            const int64_t i = seq_of[sg], n = rb.len[sg], off = rb.offset[sg];
            std::vector<Tensor> k(L), v(L);
            for (int64_t l = 0; l < L; ++l) {
                const int64_t Hkv = layer_cfgs_[l].kv_heads(), D = layer_cfgs_[l].head_dim();
                k[l] = Tensor({Hkv, n, D});
                v[l] = Tensor({Hkv, n, D});
                for (int64_t hh = 0; hh < Hkv; ++hh) {
                    std::copy(new_k[l].ptr() + (hh * R + off) * D, new_k[l].ptr() + (hh * R + off + n) * D,
                              k[l].ptr() + hh * n * D);
                    std::copy(new_v[l].ptr() + (hh * R + off) * D, new_v[l].ptr() + (hh * R + off + n) * D,
                              v[l].ptr() + hh * n * D);
                }
            }
            kb.caches_[i].append(k, v, n);
        }
    }
    for (int64_t sg = 0; sg < S; ++sg) kb.caches_[seq_of[sg]].check();
    return out;
}

namespace {
std::mutex g_gpu_mutex;  // guards each model's one-time GPU context creation; Gpu itself is thread-safe
}  // namespace

detail::Gpu* Model::gpu() const {
#ifdef TFI_USE_ACCELERATE
    std::scoped_lock lock(g_gpu_mutex);
    if (!gpu_.checked) {
        gpu_.gpu = detail::Gpu::create();
        gpu_.checked = true;
    }
    return gpu_.gpu.get();
#else
    return nullptr;
#endif
}

bool Model::has_gpu() const {
    std::scoped_lock lock(g_gpu_mutex);
    return gpu_.gpu != nullptr;
}

namespace {
void check_rows(const Model& m, int64_t layer, const Tensor& x, const detail::RowBatch& batch) {
    if (layer < 0 || layer >= m.cfg().n_layers) throw std::out_of_range("forward_rows: layer out of range");
    if (x.dim() != 2 || x.shape[1] != m.cfg().d_model || x.shape[0] != batch.rows())
        throw std::invalid_argument("forward_rows: residual must be (rows, d_model), got " + shape_str(x.shape));
}
}  // namespace

Tensor Model::forward_rows_final(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks) const {
    check_rows(*this, layer, x, batch);
    return final_norm(run_rows(layer, std::move(x), batch, hooks));
}

Tensor Model::forward_rows(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks) const {
    check_rows(*this, layer, x, batch);
    return logits_from_final(final_norm(run_rows(layer, std::move(x), batch, hooks, /*last_rows=*/true)));
}

Tensor Model::run_with_cache(const Tokens& tokens, ActivationCache& cache, NameFilter filter,
                             const Hooks* hooks) const {
    Hooks all;
    if (hooks) all.extend(*hooks);
    all.add_capture(std::move(filter), [&cache](const std::string& name, Tensor& act) { cache[name] = act; });
    return forward(tokens, &all);
}

}  // namespace tfi
