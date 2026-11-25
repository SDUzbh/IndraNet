// Python bindings for the tfi engine (nanobind). Built by CMake with -DTFI_PYTHON=ON (make python),
// into python/tfi/, and imported through python/tfi/__init__.py.
//
// Arrays: activations, logits and results come back as float32 numpy arrays that take over the
// engine's buffers (no copy). Inputs (residuals, hook replacements) may be any float array; they are
// converted to C-contiguous float32.
//
// Threads: every engine call runs with the GIL released, so other Python threads (a web server's)
// keep running; Python hook functions take it back while they run.
#include <format>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "tfi/analysis.h"
#include "tfi/generate.h"
#include "tfi/model.h"
#include "tfi/safetensors.h"
#include "tfi/tokenizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace tfi;

namespace {

using FloatArray = nb::ndarray<nb::numpy, float>;
using IntArray = nb::ndarray<nb::numpy, int64_t>;
using FloatIn = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

// A tensor as a numpy array that owns the tensor's buffer.
nb::object to_numpy(Tensor&& t) {
    auto* data = new std::vector<float>(std::move(t.data));
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    std::vector<size_t> shape(t.shape.begin(), t.shape.end());
    return nb::cast(FloatArray(data->data(), shape.size(), shape.data(), owner));
}

nb::object to_numpy(std::vector<float>&& v, std::vector<size_t> shape) {
    auto* data = new std::vector<float>(std::move(v));
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    return nb::cast(FloatArray(data->data(), shape.size(), shape.data(), owner));
}

nb::object to_numpy(std::vector<int64_t>&& v, std::vector<size_t> shape) {
    auto* data = new std::vector<int64_t>(std::move(v));
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<int64_t>*>(p); });
    return nb::cast(IntArray(data->data(), shape.size(), shape.data(), owner));
}

Tensor from_numpy(const FloatIn& a) {
    Shape shape(a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) shape[i] = static_cast<int64_t>(a.shape(i));
    return Tensor(shape, std::vector<float>(a.data(), a.data() + a.size()));
}

nb::dict cache_to_dict(ActivationCache&& cache) {
    nb::dict d;
    for (auto& [name, t] : cache) d[nb::str(name.c_str())] = to_numpy(std::move(t));
    return d;
}

// A filter accepting exactly `names` (all hook points when names is None), checked against the model's.
NameFilter names_filter(const Model& m, const std::optional<std::vector<std::string>>& names) {
    if (!names) return nullptr;
    const std::vector<std::string> all = m.hook_names();
    const std::set<std::string> known(all.begin(), all.end());
    for (const auto& n : *names)
        if (!known.contains(n))
            throw std::invalid_argument(std::format("no hook point named '{}' (Model.hook_names() lists them)", n));
    auto wanted = std::make_shared<std::set<std::string>>(names->begin(), names->end());
    return [wanted](const std::string& n) { return wanted->count(n) > 0; };
}

// A RuntimeWarning when an fp16 KV cache cut values off at fp16's limit (KVCache::clamped), as the CLI's warning.
void warn_kv_clamped(int64_t n) {
    if (n == 0) return;
    const std::string msg =
        std::format("{} numbers in the KV cache were beyond fp16's range and were cut off at +-65504, so the output "
                    "may differ from the model's; loading with fp16=False keeps them",
                    n);
    if (PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1) != 0) throw nb::python_error();
}

// Engine hooks calling Python functions fn(activation, name). Each gets a float32 copy of the
// activation. A hook's return value (an array of the same shape), or else the copy as it left it,
// replaces the activation. A capture only reads: its return value is ignored, and a pass on the GPU
// doesn't stop for it (it may be called when the pass has moved on, with the activation as it was).
// Hooks run first, then captures, so captures see what the hooks left. The dicts must outlive the
// hooks (they hold the references; the hooks only borrow them, so copying the hooks never touches
// reference counts without the GIL). None when neither is given.
std::optional<Hooks> python_hooks(const Model& m, const std::optional<nb::dict>& hooks,
                                  const std::optional<nb::dict>& captures = std::nullopt) {
    if (!hooks && !captures) return std::nullopt;
    Hooks h;
    const std::vector<std::string> all = m.hook_names();
    const std::set<std::string> known(all.begin(), all.end());
    auto each = [&](const nb::dict& fns, const char* what, auto&& add) {
        for (auto [key, value] : fns) {
            const std::string name = nb::cast<std::string>(key);
            if (!known.contains(name))
                throw std::invalid_argument(
                    std::format("no hook point named '{}' (Model.hook_names() lists them)", name));
            if (!PyCallable_Check(value.ptr()))
                throw std::invalid_argument(std::format("the {} for '{}' is not callable", what, name));
            add(name, nb::handle(value));
        }
    };
    if (hooks)
        each(*hooks, "hook", [&](const std::string& name, nb::handle fn) {
            h.add(name, [fn](const std::string& n, Tensor& act) {
                nb::gil_scoped_acquire gil;
                const Shape shape = act.shape;
                nb::object arr = to_numpy(Tensor(act));
                nb::object out = fn(arr, n);
                FloatIn result = nb::cast<FloatIn>(out.is_none() ? arr : out);
                Tensor t = from_numpy(result);
                if (t.shape != shape)
                    throw std::invalid_argument(std::format("the hook for '{}' returned shape {}, expected {}", n,
                                                            shape_str(t.shape), shape_str(shape)));
                act = std::move(t);
            });
        });
    if (captures)
        each(*captures, "capture", [&](const std::string& name, nb::handle fn) {
            h.add_capture([name](const std::string& n) { return n == name; },
                          [fn](const std::string& n, Tensor& act) {
                              nb::gil_scoped_acquire gil;
                              fn(to_numpy(Tensor(act)), n);
                          });
        });
    return h;
}

// Decoded bytes as a Python str; bytes that aren't valid UTF-8 (a character cut off) become U+FFFD.
nb::str utf8_str(const std::string& bytes) {
    PyObject* s = PyUnicode_DecodeUTF8(bytes.data(), static_cast<Py_ssize_t>(bytes.size()), "replace");
    if (!s) throw nb::python_error();
    return nb::steal<nb::str>(s);
}

int64_t resolve_pos(const Tokens& t, int64_t pos) {
    const int64_t T = std::ssize(t);
    if (pos < 0) pos += T;
    if (pos < 0 || pos >= T) throw std::out_of_range(std::format("pos is out of range for {} tokens", T));
    return pos;
}

Device parse_device(const std::string& s) {
    if (s == "cpu") return Device::Cpu;
    if (s == "both") return Device::Both;
    if (s == "gpu") return Device::Gpu;
    throw std::invalid_argument(std::format("device must be 'cpu', 'both' or 'gpu', not '{}'", s));
}

const char* device_name(Device d) { return d == Device::Cpu ? "cpu" : d == Device::Gpu ? "gpu" : "both"; }

// A generate() trace as Python: a dict per new token (see tfi.generate).
nb::list trace_list(std::vector<TraceStep>& steps, const TraceOptions& opt) {
    nb::list list;
    for (TraceStep& st : steps) {
        nb::dict d;
        d["token"] = st.token;
        d["position"] = st.position;
        d["prob"] = st.prob;
        d["logprob"] = st.logprob;
        d["top_tokens"] = st.top_tokens;
        d["top_probs"] = st.top_probs;
        if (opt.lens) {
            const size_t S = st.lens.size();
            std::vector<std::string> stages;
            std::vector<float> prob;
            std::vector<int64_t> rank, top;
            for (const LensEntry& e : st.lens) {
                stages.push_back(e.stage);
                prob.push_back(e.target_prob);
                rank.push_back(e.target_rank);
                top.push_back(e.top_tokens.empty() ? -1 : e.top_tokens[0]);
            }
            nb::dict lens;
            lens["stages"] = stages;
            lens["prob"] = to_numpy(std::move(prob), {S});
            lens["rank"] = to_numpy(std::move(rank), {S});
            lens["top"] = to_numpy(std::move(top), {S});
            d["lens"] = lens;
        }
        if (opt.dla) {
            nb::dict dla;
            const size_t n = st.dla.values.size();
            dla["labels"] = st.dla.labels;
            dla["values"] = to_numpy(std::move(st.dla.values), {n});
            dla["sum"] = st.dla.sum;
            dla["actual"] = st.dla.actual;
            d["dla"] = dla;
        }
        if (opt.attention_top > 0) {
            nb::dict att;
            for (const TraceStep::Attention& a : st.attention) {
                nb::dict one;
                one["positions"] = a.positions;
                one["weights"] = a.weights;
                att[nb::str(a.head.c_str())] = one;
            }
            d["attention"] = att;
        }
        list.append(d);
    }
    return list;
}

// The logit-lens stages, as logit_lens defines them: (label, hook point of the residual stream).
std::vector<std::pair<std::string, std::string>> lens_stages(const Model& m) {
    std::vector<std::pair<std::string, std::string>> stages = {{"embed", "blocks.0.hook_resid_pre"}};
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) {
        const std::string L = std::format("L{}", i), p = std::format("blocks.{}.", i);
        if (m.layer_cfg(i).has_mlp()) {
            stages.push_back({L + ".attn", p + "hook_resid_mid"});
            stages.push_back({L + ".mlp", p + "hook_resid_post"});
        } else {
            stages.push_back({L + ".attn", p + "hook_resid_post"});
        }
    }
    return stages;
}

nb::dict patch_dict(PatchResult&& r) {
    nb::dict d;
    d["row_labels"] = r.row_labels;
    d["col_labels"] = r.col_labels;
    d["effect"] = to_numpy(std::move(r.effect));
    d["values"] = to_numpy(std::move(r.values));
    d["clean_metric"] = r.clean_metric;
    d["corrupt_metric"] = r.corrupt_metric;
    d["normalized"] = r.normalized();
    if (!r.estimate.empty()) {
        const size_t rows = r.row_labels.size(), cols = r.col_labels.size();
        d["estimate"] = to_numpy(std::move(r.estimate));
        std::vector<int64_t> exact(r.exact.begin(), r.exact.end());
        d["exact"] = to_numpy(std::move(exact), {rows, cols});
    }
    return d;
}

}  // namespace

NB_MODULE(_tfi, mod) {
    mod.doc() = "Python bindings for the tfi interpretability engine (see the tfi package).";

    // --- settings ------------------------------------------------------------------------
    mod.def(
        "set_device", [](const std::string& d) { set_device(parse_device(d)); }, "device"_a,
        "Where computations run: 'cpu', 'both' (default: the GPU for large batched work) or 'gpu'.");
    mod.def("device", [] { return std::string(device_name(device())); });
    mod.def("set_num_threads", &set_num_threads, "n"_a,
            "Threads for patching, ablation and means (0: TFI_THREADS, else every core).");
    mod.def("num_threads", &num_threads);
    mod.def("read_metadata", &read_safetensors_metadata, "path"_a, nb::call_guard<nb::gil_scoped_release>(),
            "The string metadata of a safetensors file's header (no tensor is read).");

    // --- tokenizer -----------------------------------------------------------------------
    nb::class_<Tokenizer>(mod, "Tokenizer", "A Hugging Face tokenizer.json, as the engine runs it.")
        .def_static("from_json", &Tokenizer::from_json, "tokenizer_json"_a)
        .def("encode", &Tokenizer::encode, "text"_a, "add_special_tokens"_a = true,
             nb::call_guard<nb::gil_scoped_release>())
        .def(
            "decode",
            [](const Tokenizer& t, const Tokens& ids, bool skip_special_tokens) {
                return utf8_str(t.decode(ids, skip_special_tokens));
            },
            "ids"_a, "skip_special_tokens"_a = false, "Text for token ids.");

    // --- KV cache ------------------------------------------------------------------------
    nb::class_<KVCache>(mod, "KVCache",
                        "Every layer's keys and values for the positions run so far (Model.forward_cached). "
                        "copy() continues separately, e.g. several continuations of one prompt.")
        .def(nb::init<>())
        .def_prop_ro("size", &KVCache::size, "Positions held.")
        .def("__len__", &KVCache::size)
        .def("clear", &KVCache::clear)
        .def("truncate", &KVCache::truncate, "n"_a, "Keep positions 0..n-1.")
        .def_prop_ro("clamped", &KVCache::clamped,
                     "Keys and values at fp16's limit, +-65504: with fp16 weights the cache cuts values beyond fp16's "
                     "range off to it (above 0: some changed by far more than rounding). Always 0 with float32.")
        .def(
            "keys", [](const KVCache& c, int64_t layer) { return to_numpy(c.keys(layer)); }, "layer"_a,
            "Layer's keys (kv_heads, size, d_head), as hook_k gave them.")
        .def(
            "values", [](const KVCache& c, int64_t layer) { return to_numpy(c.values(layer)); }, "layer"_a,
            "Layer's values (kv_heads, size, d_head), as hook_v gave them.")
        .def("copy", [](const KVCache& c) { return KVCache(c); })
        .def("__copy__", [](const KVCache& c) { return KVCache(c); })
        .def("__deepcopy__", [](const KVCache& c, nb::handle) { return KVCache(c); }, "memo"_a);

    // --- model ---------------------------------------------------------------------------
    nb::class_<Model>(mod, "Model", "A model loaded from a safetensors file (indranet.export or indranet_hf).")
        .def_static(
            "load",
            [](const std::string& path, bool fp16) {
                nb::gil_scoped_release release;
                return Model::load(path, fp16 ? WeightFormat::fp16 : WeightFormat::fp32);
            },
            "path"_a, "fp16"_a = false, "Load a model; fp16 keeps the matrices in half precision (half the memory).")
        .def_prop_ro("n_layers", [](const Model& m) { return m.cfg().n_layers; })
        .def_prop_ro("d_model", [](const Model& m) { return m.cfg().d_model; })
        .def_prop_ro("vocab_size", [](const Model& m) { return m.cfg().vocab_size; })
        .def_prop_ro("max_seq_len", [](const Model& m) { return m.cfg().max_seq_len; })
        .def_prop_ro("max_heads", &Model::max_heads)
        .def_prop_ro(
            "layer_heads",
            [](const Model& m) {
                std::vector<int64_t> h;
                for (int64_t i = 0; i < m.cfg().n_layers; ++i) h.push_back(m.layer_cfg(i).n_heads);
                return h;
            },
            "Number of attention heads of each layer.")
        .def_prop_ro("num_parameters", &Model::num_parameters)
        .def_prop_ro("weight_bytes", &Model::weight_bytes)
        .def_prop_ro("fp16", [](const Model& m) { return m.weight_format() == WeightFormat::fp16; })
        .def_prop_ro("vocab", &Model::vocab, "One string per token id, or empty.")
        .def_prop_ro("has_tokenizer", &Model::has_tokenizer)
        .def("token_str", &Model::token_str, "id"_a, "The token's text, or its id.")
        .def("tokenize", &Model::tokenize, "text"_a, "add_special_tokens"_a = true,
             nb::call_guard<nb::gil_scoped_release>(),
             "Token ids: the model's tokenizer, or the longest vocabulary match.")
        .def(
            "decode",
            [](const Model& m, const Tokens& ids, bool skip_special_tokens) {
                return utf8_str(m.decode(ids, skip_special_tokens));
            },
            "ids"_a, "skip_special_tokens"_a = false,
            "Text for token ids: the model's tokenizer, or its vocabulary's strings joined.")
        .def_prop_rw("eos_tokens", &Model::eos_tokens, &Model::set_eos_tokens,
                     "Token ids that end a text (Hugging Face's eos_token_id for converted models).")
        .def("hook_names", &Model::hook_names)
        .def("weight_names",
             [](const Model& m) {
                 std::vector<std::string> n;
                 for (const auto& [name, w] : m.weights()) n.push_back(name);
                 return n;
             })
        .def(
            "weight",
            [](const Model& m, const std::string& name) {
                const Weight& w = m.weight(name);
                Tensor t = w.dim() == 2 ? w.to_tensor() : w.f32();
                return to_numpy(std::move(t));
            },
            "name"_a, "A weight's values as float32 (a copy).")
        .def(
            "forward",
            [](const Model& m, const Tokens& tokens, std::optional<nb::dict> hooks, std::optional<nb::dict> captures) {
                std::optional<Hooks> h = python_hooks(m, hooks, captures);
                Tensor logits;
                {
                    nb::gil_scoped_release release;
                    logits = m.forward(tokens, h ? &*h : nullptr);
                }
                return to_numpy(std::move(logits));
            },
            "tokens"_a, "hooks"_a = nb::none(), "captures"_a = nb::none(),
            "Logits (T, vocab). hooks: {hook name: fn(activation, name)}; fn may edit the array or return a new one. "
            "captures: the same, only reading (the return value is ignored; a GPU pass doesn't stop for them).")
        .def(
            "forward_cached",
            [](const Model& m, const Tokens& tokens, KVCache& cache, std::optional<nb::dict> hooks, bool last_only,
               std::optional<nb::dict> captures) {
                std::optional<Hooks> h = python_hooks(m, hooks, captures);
                Tensor logits;
                {
                    nb::gil_scoped_release release;
                    logits = m.forward_cached(tokens, cache, h ? &*h : nullptr, last_only);
                }
                return to_numpy(std::move(logits));
            },
            "tokens"_a, "cache"_a, "hooks"_a = nb::none(), "last_only"_a = false, "captures"_a = nb::none(),
            "Logits for tokens continuing the sequence in cache, which then holds them too: (T, vocab), or (1, vocab) "
            "with last_only. Hooks see only the new rows; hook_pattern is (H, T, cache.size + T).")
        .def(
            "run_with_cache",
            [](const Model& m, const Tokens& tokens, std::optional<std::vector<std::string>> names,
               std::optional<nb::dict> hooks, std::optional<nb::dict> captures) {
                NameFilter filter = names_filter(m, names);
                std::optional<Hooks> h = python_hooks(m, hooks, captures);
                ActivationCache cache;
                Tensor logits;
                {
                    nb::gil_scoped_release release;
                    logits = m.run_with_cache(tokens, cache, filter, h ? &*h : nullptr);
                }
                return nb::make_tuple(to_numpy(std::move(logits)), cache_to_dict(std::move(cache)));
            },
            "tokens"_a, "names"_a = nb::none(), "hooks"_a = nb::none(), "captures"_a = nb::none(),
            "(logits, {hook name: activation}) for the named hook points (all when names is None).")
        .def(
            "embed",
            [](const Model& m, const Tokens& tokens) {
                Tensor t;
                {
                    nb::gil_scoped_release release;
                    t = m.embed(tokens);
                }
                return to_numpy(std::move(t));
            },
            "tokens"_a, "The input to block 0, (T, d_model).")
        .def(
            "forward_from",
            [](const Model& m, int64_t layer, const FloatIn& resid) {
                Tensor r = from_numpy(resid), logits;
                {
                    nb::gil_scoped_release release;
                    logits = m.forward_from(layer, std::move(r));
                }
                return to_numpy(std::move(logits));
            },
            "layer"_a, "resid"_a, "Run blocks layer.. from their input resid (T, d_model); returns the logits.")
        .def(
            "final_norm",
            [](const Model& m, const FloatIn& resid) {
                Tensor r = from_numpy(resid), out;
                {
                    nb::gil_scoped_release release;
                    out = m.final_norm(r);
                }
                return to_numpy(std::move(out));
            },
            "resid"_a)
        .def(
            "logits_from_final",
            [](const Model& m, const FloatIn& normed) {
                Tensor r = from_numpy(normed), out;
                {
                    nb::gil_scoped_release release;
                    out = m.logits_from_final(r);
                }
                return to_numpy(std::move(out));
            },
            "normed"_a, "Unembedding (and final softcap) of final-normed rows (T, d_model).");

    // --- generation ----------------------------------------------------------------------
    mod.def(
        "generate",
        [](const Model& m, nb::object prompt, int64_t max_new_tokens, double temperature, int64_t top_k, double top_p,
           uint64_t seed, std::optional<Tokens> stop_tokens, bool stop_at_eos, std::optional<nb::dict> hooks,
           nb::object on_token, std::optional<nb::dict> captures, bool trace, int64_t trace_top_k, bool trace_lens,
           bool trace_dla, int64_t trace_attention, int64_t speculate, const Model* draft, int64_t ngram,
           std::optional<nb::dict> stats) -> nb::object {
            Tokens tokens;
            if (nb::isinstance<nb::str>(prompt)) {
                const std::string text = nb::cast<std::string>(prompt);
                nb::gil_scoped_release release;
                tokens = m.tokenize(text);
            } else {
                tokens = nb::cast<Tokens>(prompt);
            }
            GenerateOptions o;
            o.max_new_tokens = max_new_tokens;
            o.temperature = temperature;
            o.top_k = top_k;
            o.top_p = top_p;
            o.seed = seed;
            if (stop_tokens) o.stop_tokens = *stop_tokens;
            o.stop_at_eos = stop_at_eos;
            o.trace.top_k = trace_top_k;
            o.trace.lens = trace_lens;
            o.trace.dla = trace_dla;
            o.trace.attention_top = trace_attention;
            if (draft && speculate == 0) throw std::invalid_argument("draft needs speculate > 0");
            SpeculativeStats st;
            o.speculative.draft_tokens = speculate;
            o.speculative.draft = draft;
            o.speculative.ngram = ngram;
            o.speculative.stats = &st;
            int64_t clamped = 0;
            o.kv_clamped = &clamped;
            if (!on_token.is_none()) {
                if (!PyCallable_Check(on_token.ptr())) throw std::invalid_argument("on_token is not callable");
                nb::handle fn = on_token;  // borrowed: on_token outlives the call
                o.on_token = [fn](int64_t t) {
                    nb::gil_scoped_acquire gil;
                    nb::object r = fn(t);
                    return r.is_none() || nb::cast<bool>(r);
                };
            }
            std::optional<Hooks> h = python_hooks(m, hooks, captures);
            std::vector<TraceStep> steps;
            Tokens out;
            {
                nb::gil_scoped_release release;
                out = generate(m, tokens, o, h ? &*h : nullptr, trace ? &steps : nullptr);
            }
            warn_kv_clamped(clamped);
            if (stats) {
                (*stats)["steps"] = st.steps;
                (*stats)["drafted"] = st.drafted;
                (*stats)["accepted"] = st.accepted;
                (*stats)["reruns"] = st.reruns;
            }
            if (!trace) return nb::cast(out);
            nb::list list = trace_list(steps, o.trace);
            return nb::make_tuple(out, list);
        },
        "model"_a, "prompt"_a, "max_new_tokens"_a = 32, "temperature"_a = 0.0, "top_k"_a = 0, "top_p"_a = 1.0,
        "seed"_a = 0, "stop_tokens"_a = nb::none(), "stop_at_eos"_a = true, "hooks"_a = nb::none(),
        "on_token"_a = nb::none(), "captures"_a = nb::none(), "trace"_a = false, "trace_top_k"_a = 5,
        "trace_lens"_a = true, "trace_dla"_a = true, "trace_attention"_a = 0, "speculate"_a = 0,
        "draft"_a.none() = nb::none(), "ngram"_a = 3, "stats"_a = nb::none(),
        "New token ids continuing prompt (token ids, or text for the model's tokenizer), with a KV cache. Greedy at "
        "temperature 0; otherwise sampled from the top_k most likely and the fewest reaching top_p, reproducibly from "
        "seed. Stops at stop_tokens, the model's eos_tokens (unless stop_at_eos is False), or when on_token(token) "
        "returns False. hooks (editing) and captures (reading only) apply at every step: the prompt's pass, then each "
        "new token's. With trace, returns (tokens, steps): per new token a dict of its probability, the top tokens, "
        "and (trace_lens) the logit lens at its position, (trace_dla) direct logit attribution of its logit, "
        "(trace_attention > 0) each head's most attended positions. speculate=K: speculative decoding, each step "
        "proposing up to K tokens that the model checks in one pass, from draft (a smaller model with the same "
        "tokenizer) or else from earlier in the text (the tokens after the last occurrence of the last ngram tokens). "
        "Only the speed changes: greedy output is the same up to rounding, sampled output has the same distribution "
        "(not the same tokens for a seed). Hooks and captures see what they see without speculation: one row per "
        "call, kept tokens only, in order, so they can keep state or pick rows. Hooks whose edits can't be predicted "
        "get little or no speedup. A draft "
        "with a smaller vocabulary stops proposing once the text holds a token past it. A stats dict receives "
        "steps, drafted, accepted and reruns.");

    mod.def(
        "generate_batch",
        [](const Model& m, nb::list prompts, int64_t max_new_tokens, double temperature, int64_t top_k, double top_p,
           uint64_t seed, std::optional<Tokens> stop_tokens, bool stop_at_eos, nb::object hooks, nb::object captures,
           bool trace, int64_t trace_top_k, bool trace_lens, bool trace_dla) -> nb::object {
            std::vector<Tokens> ps;
            for (nb::handle p : prompts) {
                if (nb::isinstance<nb::str>(p)) {
                    const std::string text = nb::cast<std::string>(p);
                    nb::gil_scoped_release release;
                    ps.push_back(m.tokenize(text));
                } else {
                    ps.push_back(nb::cast<Tokens>(p));
                }
            }
            const size_t n = ps.size();
            // hooks / captures: one dict for every sequence, or a list with a dict (or None) per sequence.
            auto per_seq = [&](nb::object x, const char* what) {
                std::vector<std::optional<nb::dict>> v(n);
                if (x.is_none()) return v;
                if (nb::isinstance<nb::dict>(x)) {
                    for (auto& d : v) d = nb::cast<nb::dict>(x);
                    return v;
                }
                nb::list l = nb::cast<nb::list>(x);
                if (l.size() != n)
                    throw std::invalid_argument(std::string(what) + ": one per prompt (or one dict for all)");
                for (size_t i = 0; i < n; ++i)
                    if (!l[i].is_none()) v[i] = nb::cast<nb::dict>(l[i]);
                return v;
            };
            const std::vector<std::optional<nb::dict>> hs = per_seq(hooks, "hooks"), cs = per_seq(captures, "captures");
            std::vector<std::optional<Hooks>> built(n);
            std::vector<const Hooks*> ptrs(n, nullptr);
            bool any = false;
            for (size_t i = 0; i < n; ++i) {
                built[i] = python_hooks(m, hs[i], cs[i]);
                if (built[i]) {
                    ptrs[i] = &*built[i];
                    any = true;
                }
            }
            GenerateOptions o;
            o.max_new_tokens = max_new_tokens;
            o.temperature = temperature;
            o.top_k = top_k;
            o.top_p = top_p;
            o.seed = seed;
            if (stop_tokens) o.stop_tokens = *stop_tokens;
            o.stop_at_eos = stop_at_eos;
            o.trace.top_k = trace_top_k;
            o.trace.lens = trace_lens;
            o.trace.dla = trace_dla;
            int64_t clamped = 0;
            o.kv_clamped = &clamped;
            std::vector<std::vector<TraceStep>> traces;
            std::vector<Tokens> out;
            {
                nb::gil_scoped_release release;
                out = generate_batch(m, ps, o, any ? ptrs : std::vector<const Hooks*>{}, trace ? &traces : nullptr);
            }
            warn_kv_clamped(clamped);
            if (!trace) return nb::cast(out);
            nb::list tl;
            for (auto& t : traces) tl.append(trace_list(t, o.trace));
            return nb::make_tuple(out, tl);
        },
        "model"_a, "prompts"_a, "max_new_tokens"_a = 32, "temperature"_a = 0.0, "top_k"_a = 0, "top_p"_a = 1.0,
        "seed"_a = 0, "stop_tokens"_a = nb::none(), "stop_at_eos"_a = true, "hooks"_a = nb::none(),
        "captures"_a = nb::none(), "trace"_a = false, "trace_top_k"_a = 5, "trace_lens"_a = true, "trace_dla"_a = true,
        "Several sequences at once, one stacked pass per step (the weights read once for all): sequence i continues "
        "prompts[i] (token ids, or text) as generate() would with seed + i. hooks / captures: one dict for every "
        "sequence, or a list with a dict (or None) per sequence, each seeing only its sequence's rows (hook_pattern is "
        "not available). Returns a list of new token lists, with trace=True (lists, per-sequence traces).");

    // --- analyses ------------------------------------------------------------------------
    mod.def(
        "logit_metric",
        [](const FloatIn& logits, int64_t pos, int64_t target, int64_t other) {
            return logit_metric(from_numpy(logits), pos, target, other);
        },
        "logits"_a, "pos"_a, "target"_a, "other"_a = -1);
    mod.def(
        "next_token_loss",
        [](const FloatIn& logits, const Tokens& tokens) { return next_token_loss(from_numpy(logits), tokens); },
        "logits"_a, "tokens"_a);

    mod.def(
        "logit_lens",
        [](const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t k) {
            std::vector<LensEntry> entries;
            {
                nb::gil_scoped_release release;
                entries = logit_lens(m, tokens, resolve_pos(tokens, pos), target, k);
            }
            nb::list out;
            for (auto& e : entries) {
                nb::dict d;
                d["stage"] = e.stage;
                d["top_tokens"] = e.top_tokens;
                d["top_probs"] = e.top_probs;
                if (target >= 0) {
                    d["target_prob"] = e.target_prob;
                    d["target_rank"] = e.target_rank;
                }
                out.append(d);
            }
            return out;
        },
        "model"_a, "tokens"_a, "pos"_a = -1, "target"_a = -1, "k"_a = 5,
        "The residual stream after every sublayer decoded at pos with the final norm and unembedding.");

    mod.def(
        "lens_grid",
        [](const Model& m, const Tokens& tokens, int64_t k) {
            const auto stages = lens_stages(m);
            const int64_t T = std::ssize(tokens), V = m.cfg().vocab_size, S = std::ssize(stages);
            if (T == 0) throw std::invalid_argument("no tokens");
            k = std::max<int64_t>(1, std::min(k, V));
            std::vector<int64_t> top(S * T * k), rank(S * T, -1);
            std::vector<float> prob(S * T * k), next(S * T, std::numeric_limits<float>::quiet_NaN());
            {
                nb::gil_scoped_release release;
                std::set<std::string> needed;
                for (const auto& [label, hook] : stages) needed.insert(hook);
                ActivationCache cache;
                m.run_with_cache(tokens, cache, [&needed](const std::string& n) { return needed.contains(n); });
                const int64_t block = 64;  // rows unembedded at once: (block, V) logits at a time
                for (int64_t s = 0; s < S; ++s) {
                    const Tensor normed = m.final_norm(cache.at(stages[s].second));
                    for (int64_t r0 = 0; r0 < T; r0 += block) {
                        const int64_t r1 = std::min(T, r0 + block);
                        Tensor probs = m.logits_from_final(rows(normed, r0, r1));
                        for (int64_t p = r0; p < r1; ++p) {
                            const std::span<float> v = probs.row_view(p - r0);
                            softmax_(v);
                            const std::vector<int64_t> best = topk(v, k);
                            for (int64_t j = 0; j < k; ++j) {
                                top[(s * T + p) * k + j] = best[j];
                                prob[(s * T + p) * k + j] = v[best[j]];
                            }
                            if (p + 1 < T) {
                                next[s * T + p] = v[tokens[p + 1]];
                                rank[s * T + p] = rank_of(v, tokens[p + 1]);
                            }
                        }
                    }
                }
            }
            nb::dict d;
            std::vector<std::string> labels;
            for (const auto& [label, hook] : stages) labels.push_back(label);
            const size_t s = static_cast<size_t>(S), t = static_cast<size_t>(T), kk = static_cast<size_t>(k);
            d["stages"] = labels;
            d["top_tokens"] = to_numpy(std::move(top), {s, t, kk});
            d["top_probs"] = to_numpy(std::move(prob), {s, t, kk});
            d["next_prob"] = to_numpy(std::move(next), {s, t});
            d["next_rank"] = to_numpy(std::move(rank), {s, t});
            return d;
        },
        "model"_a, "tokens"_a, "k"_a = 5,
        "The logit lens at every position from one forward pass: stages, top_tokens and top_probs (stages, T, k), "
        "and next_prob / next_rank (stages, T): the probability and rank of the actual next token (NaN / -1 at the "
        "last position).");

    mod.def(
        "direct_logit_attribution",
        [](const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t other) {
            DLAResult r;
            {
                nb::gil_scoped_release release;
                r = direct_logit_attribution(m, tokens, resolve_pos(tokens, pos), target, other);
            }
            nb::dict d;
            d["labels"] = r.labels;
            const size_t n = r.values.size();
            d["values"] = to_numpy(std::move(r.values), {n});
            d["sum"] = r.sum;
            d["actual"] = r.actual;
            return d;
        },
        "model"_a, "tokens"_a, "pos"_a, "target"_a, "other"_a = -1,
        "The logit difference split into embeddings, heads, MLPs and norm bias; values sum to 'actual'.");

    mod.def(
        "head_scores",
        [](const Model& m, int64_t segment_len, int64_t n_samples, uint64_t seed) {
            HeadScores s;
            {
                nb::gil_scoped_release release;
                s = head_scores(m, segment_len, n_samples, seed);
            }
            nb::dict d;
            d["previous_token"] = to_numpy(std::move(s.previous_token));
            d["induction"] = to_numpy(std::move(s.induction));
            d["duplicate_token"] = to_numpy(std::move(s.duplicate_token));
            d["first_token"] = to_numpy(std::move(s.first_token));
            d["self"] = to_numpy(std::move(s.self));
            nb::list hops;
            for (const auto& h : s.top_two_hop) hops.append(nb::make_tuple(h.writer, h.reader, h.score));
            d["top_two_hop"] = hops;
            return d;
        },
        "model"_a, "segment_len"_a = 24, "n_samples"_a = 8, "seed"_a = 0,
        "Attention-head pattern scores on random repeated sequences, each (n_layers, max_heads).");

    mod.def(
        "patch",
        [](const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site, int64_t pos,
           int64_t target, int64_t other, const std::string& method, int64_t steps, int64_t verify) {
            if (method != "exact" && method != "attribution" && method != "mixed")
                throw std::invalid_argument(
                    std::format("method must be 'exact', 'attribution' or 'mixed', not '{}'", method));
            PatchResult r;
            {
                nb::gil_scoped_release release;
                const int64_t p = resolve_pos(clean, pos);
                const bool heads = site == "heads";
                if (method == "exact")
                    r = heads ? patch_heads(m, clean, corrupt, p, target, other)
                              : patch_residual(m, clean, corrupt, site, p, target, other);
                else if (method == "attribution")
                    r = heads ? attribution_patch_heads(m, clean, corrupt, p, target, other, steps)
                              : attribution_patch_residual(m, clean, corrupt, site, p, target, other, steps);
                else
                    r = heads ? mixed_patch_heads(m, clean, corrupt, p, target, other, verify, steps)
                              : mixed_patch_residual(m, clean, corrupt, site, p, target, other, verify, steps);
            }
            return patch_dict(std::move(r));
        },
        "model"_a, "clean"_a, "corrupt"_a, "site"_a, "pos"_a, "target"_a, "other"_a = -1, "method"_a = "exact",
        "steps"_a = 1, "verify"_a = 10,
        "Activation patching clean -> corrupted. site: resid_pre, resid_mid, resid_post, attn_out, mlp_out (rows "
        "layers, "
        "columns positions) or heads (rows layers, columns heads). method: exact, attribution or mixed.");

    mod.def(
        "metric_gradients",
        [](const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t other,
           std::optional<std::vector<std::string>> names) {
            NameFilter filter = names_filter(m, names);
            ActivationCache grads;
            float metric;
            {
                nb::gil_scoped_release release;
                metric = metric_gradients(m, tokens, resolve_pos(tokens, pos), target, other, grads, filter);
            }
            return nb::make_tuple(metric, cache_to_dict(std::move(grads)));
        },
        "model"_a, "tokens"_a, "pos"_a, "target"_a, "other"_a = -1, "names"_a = nb::none(),
        "(metric, {hook name: d metric / d activation}).");

    mod.def(
        "head_ablation",
        [](const Model& m, const Tokens& tokens, const std::string& mode, const std::vector<Tokens>& reference) {
            if (mode != "zero" && mode != "mean") throw std::invalid_argument("mode must be 'zero' or 'mean'");
            Tensor t;
            {
                nb::gil_scoped_release release;
                t = head_ablation(m, tokens, mode == "zero" ? Ablation::Zero : Ablation::Mean, reference);
            }
            return to_numpy(std::move(t));
        },
        "model"_a, "tokens"_a, "mode"_a = "zero", "reference"_a = std::vector<Tokens>{},
        "Increase in next-token loss when each head is ablated: (n_layers, max_heads).");

    mod.def(
        "composition_scores",
        [](const Model& m) {
            Composition c;
            {
                nb::gil_scoped_release release;
                c = composition_scores(m);
            }
            nb::dict d;
            d["labels"] = c.labels;
            d["q"] = to_numpy(std::move(c.q));
            d["k"] = to_numpy(std::move(c.k));
            d["v"] = to_numpy(std::move(c.v));
            return d;
        },
        "model"_a, "Q/K/V composition scores between heads (Elhage et al. 2021).");
    mod.def("composition_random_baseline", &composition_random_baseline, "model"_a, "n_pairs"_a = 64, "seed"_a = 0,
            nb::call_guard<nb::gil_scoped_release>());
    mod.def(
        "ov_copying_scores",
        [](const Model& m, int64_t max_tokens) {
            Tensor t;
            {
                nb::gil_scoped_release release;
                t = ov_copying_scores(m, max_tokens);
            }
            return to_numpy(std::move(t));
        },
        "model"_a, "max_tokens"_a = 1024,
        "How often each head's OV circuit copies a token to itself: (n_layers, max_heads).");
    mod.def(
        "head_weights",
        [](const Model& m, int64_t layer, int64_t head) {
            HeadWeights h;
            {
                nb::gil_scoped_release release;
                h = head_weights(m, layer, head);
            }
            nb::dict d;
            d["W_Q"] = to_numpy(std::move(h.W_Q));
            d["W_K"] = to_numpy(std::move(h.W_K));
            d["W_V"] = to_numpy(std::move(h.W_V));
            d["W_O"] = to_numpy(std::move(h.W_O));
            return d;
        },
        "model"_a, "layer"_a, "head"_a,
        "One head's W_Q, W_K, W_V (d_head, d_model) and W_O (d_model, d_head), norm folded in.");
}
