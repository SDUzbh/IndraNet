// The GPU lowering of the model's batched passes: Model::run_rows (several sequences' rows stacked through the
// blocks, the residual stream kept on the GPU where placement.h says so; otherwise block by block, linear layers on
// the GPU only where they are large) and Model::gradients_rows (forward and backward passes on the GPU, for
// attribution patching and run_with_gradients). GpuBlocks records each block's work, and its backward pass, as
// Gpu::Seq operations (gpu/gpu.h), from the same weights and config as the CPU's Model::block (model.cpp) and
// Backward::block (grad.cpp), which the tests hold it to.
#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <stdexcept>

#include "gpu/gpu.h"
#include "model/model_impl.h"
#include "model/ops.h"
#include "model/placement.h"
#include "support/half.h"
#include "support/vmath.h"
#include "tfi/device.h"
#include "tfi/model.h"

namespace tfi {

using detail::alibi_slopes;
using detail::hook;
using detail::layer_prefix;
using namespace detail::names;

#ifdef TFI_USE_ACCELERATE
namespace {

using detail::GpuAct;

GpuAct gpu_act(const std::string& act) {
    if (act == "relu") return GpuAct::Relu;
    if (act == "gelu") return GpuAct::Gelu;
    if (act == "gelu_tanh") return GpuAct::GeluTanh;
    if (act == "silu") return GpuAct::Silu;
    if (act == "tanh") return GpuAct::Tanh;
    if (act == "relu2") return GpuAct::Relu2;
    throw std::runtime_error("unsupported activation " + act);
}

// Gpu::Seq slots of a batched pass. Inference (run_rows) uses fixed slots: the residual stream
// alternates between kResid0 and kResid1 (each step reads one and writes the other), and a
// parallel-residual MLP's output between kPar0 and kPar1, so that a step never overwrites its own
// inputs (Gpu::Seq::run may run it twice). Gradient passes (gradients_rows) give each layer's
// activations and gradients slots of their own, from kLayerSlots on, for the backward pass.
enum Slot : int {
    kResid0,
    kResid1,
    kAttnIn,
    kQ,
    kK,
    kV,
    kZ,
    kAttnOut,
    kAttnNormed,
    kMlpIn,
    kGate,
    kUp,
    kMlpOut,
    kMlpRaw,
    kPar0,
    kPar1,
    kTmp,
    kDZ,
    kDQ,
    kDK,
    kDV,
    kDIn,
    kDPost,
    kDPre,
    kDUp,
    kDMlpIn,
    kTmpB,
    kTmpC,
    kTmpD,
    kTop,  // backward
    // attention on the GPU: q, k, v of odd layers; q, k after qk_norm / RoPE; per-row
    // positions; RoPE frequencies; ALiBi slopes; earlier positions' keys and values
    kQ2,
    kK2,
    kV2,
    kQn,
    kKn,
    kQr,
    kKr,
    kMeta,
    kInvFreq,
    kSlopes,
    kPK,
    kPV,
    kBlocks,
    // attention's backward on the GPU: dz of odd layers, delta, dq and dk between steps
    kDZ2,
    kDelta,
    kDQ1,
    kDQ2,
    kDK1,
    kDK2,
    // a whole sequence's attention pattern (hook_pattern) and the lse it is made from
    kPattern,
    kLse,
    kLayerSlots = 64
};
constexpr int kSlotsPerLayer = 21;
// Hooked activations of inference passes (GpuBlocks::at), in turn: the layer slots, which only
// gradient passes use (and they have no hooks). More than two blocks' hook points, so a slot is
// never reused while it may still be read.
constexpr int kHand0 = kLayerSlots, kHandSlots = 32;
// Activations that only capture hooks want (GpuBlocks::defer), copied on the GPU and handed over
// after each run: next to the hooked ones.
constexpr int kCapture0 = kHand0 + kHandSlots, kCaptureSlots = 64;

// Where one block's activations go.
struct LayerSlots {
    int x, out;            // the residual stream in and out
    int attn_raw;          // o_proj's output (before any sandwich norm)
    int mid;               // hook_resid_mid (may be `out`)
    int pre, up;           // the MLP's pre-activation (GLU: the gate's) and a GLU's up projection
    int mlp_raw, mlp_out;  // the MLP's output before and after a sandwich norm
};

// Inference: block `layer` reads slot `cur` and writes `next`, everything else in scratch slots.
LayerSlots inference_slots(const ModelConfig& c, int64_t layer, int cur, int next) {
    const int par = kPar0 + static_cast<int>(layer & 1);  // must survive from before to after attention
    const bool sandwich = c.norm_position == "sandwich";
    return {cur,
            next,
            kAttnOut,
            next,
            kGate,
            kUp,
            c.parallel_residual && !sandwich ? par : kMlpRaw,
            c.parallel_residual ? par : kMlpOut};
}

// Gradient passes: every activation the backward pass needs has a slot of its own.
LayerSlots saved_slots(int64_t layer) {
    const int b = kLayerSlots + kSlotsPerLayer * static_cast<int>(layer);
    return {b, b + kSlotsPerLayer, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6};
}
struct GradSlots {
    int resid_pre, mid, attn_out, mlp_out;  // gradients at these hook points (mid is scratch with parallel residual)
};
GradSlots grad_slots(int64_t layer) {
    const int b = kLayerSlots + kSlotsPerLayer * static_cast<int>(layer);
    return {b + 7, b + 8, b + 9, b + 10};
}
// Gradient passes with attention on the GPU: what attention's backward pass reads.
struct AttnLayerSlots {
    int q0, k0, v;       // the projections' outputs
    int qn, kn, qr, kr;  // q, k after qk_norm and after RoPE
    int z, lse;          // attention's output and each (row, head)'s log-sum-exp of its scores
};
AttnLayerSlots attn_slots(int64_t layer) {
    const int b = kLayerSlots + kSlotsPerLayer * static_cast<int>(layer) + 11;
    return {b, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6, b + 7, b + 8};
}

// The GPU half of each block (everything but attention) for a batched pass, built from the same
// weights and config as Model::block, and its backward pass (as grad.cpp's Backward::block).
struct GpuBlocks {
    const Model& m;
    detail::Gpu::Seq& s;
    int64_t R, d;
    int qs = kQ, ks = kK, vs = kV;               // where before_attention puts q, k and v
    int qn = kQn, kn = kKn, qr = kQr, kr = kKr;  // where qk() puts q, k after qk_norm and RoPE
    int zs = kZ, lse = -1;                       // where attention() puts z (and lse, if >= 0)
    int dzs = kDZ;                               // where backward_before puts dL/dz
    int64_t blocks = 0;                          // attention blocks in kBlocks (0: the per-row kernel)
    int64_t block_rows = 32;                     // ... of up to this many rows
    // Where attention reads earlier positions' keys and values, RoPE frequencies and ALiBi slopes
    // (a KV cache kept on the GPU gives each layer slots of its own: detail::GpuKV).
    int pk = kPK, pv = kPV, inv_freq = kInvFreq, slopes = kSlopes;
    int kbase = -1;                // a slot of each row's region start in pk / pv (several sequences' caches), or none
    bool half_prefix = false;      // pk / pv hold fp16 values (a KV cache kept in fp16)
    const Hooks* hooks = nullptr;  // the pass's hooks (GpuBlocks::at)
    // Gradient passes: hooked and captured activations go past every layer's slots, and a hook's edit goes back
    // into the activation's own slot (at), where the backward pass reads it as the activation, as
    // run_with_gradients' cache holds it. (Their runs are float32, never repeated.)
    int hand0 = kHand0, capture0 = kCapture0;
    bool in_place = false;
    // Gradient passes: called with each hook point's gradient where the backward pass leaves it (rows x width, a
    // View as at()'s), for the caller to copy before a later step overwrites it.
    std::function<void(const std::string& name, int slot, int64_t width, int view, int64_t D)> grad_at = nullptr;
    bool fp16 = false;             // ... and how its runs multiply
    int par_out = -1;              // a parallel-residual MLP's output (set by mlp())
    int next_hand = 0;

    // A hook point on the activation in `slot` (rows x width). If a hook wants it, everything
    // recorded so far runs, the hook gets the activation as Model::block passes it ((rows, width);
    // Heads: (width / D, rows, D), as q, k, v; RowsHeads: (rows, width / D, D), as z), and what the
    // hook leaves goes to a slot of its own, returned, that the rest of the pass reads instead (so no
    // later run overwrites a value it read, which Gpu::Seq::run's repeat of a run would need).
    // Otherwise returns `slot`. With in_place (gradient passes), what the hook leaves goes back into `slot`
    // instead, unless `keep`: the slot holds another activation too, or a value the backward pass reads as
    // it was before the hook.
    // When every hook that wants it only captures (Hooks::add_capture), the pass does not stop: the
    // activation is copied on the GPU and handed over after the run (defer).
    enum class View { Rows, Heads, RowsHeads, Whole };
    void report(const std::string& name, int slot, int64_t width, View view = View::Rows, int64_t D = 0) {
        if (grad_at) grad_at(name, slot, width, static_cast<int>(view), D);
    }
    Tensor view_of(const float* src, int64_t width, View view, int64_t D) const {
        const int64_t n = R * width;
        return view == View::Heads ? detail::split_heads(src, R, width / D, D)
                                   : Tensor(view == View::Rows ? Shape{R, width} : Shape{R, width / D, D},
                                            std::vector<float>(src, src + n));
    }
    int at(const std::string& name, int slot, int64_t width, View view = View::Rows, int64_t D = 0, bool keep = false) {
        if (!hooks || !hooks->matches(name)) return slot;
        const int64_t n = R * width;
        if (!hooks->changes(name)) {
            const int dst = capture_slot(n);
            s.copy(slot, dst, n);
            defer(name, dst, Shape{}, view, width, D);
            return slot;
        }
        run();
        const float* src = s.data(slot);
        Tensor t = view_of(src, width, view, D);
        hook(hooks, name, t);
        if (t.numel() != n) throw std::runtime_error(name + ": a hook changed the activation's size");
        const Tensor back = view == View::Heads ? detail::merge_heads(t) : std::move(t);
        if (in_place && !keep) {
            std::copy_n(back.ptr(), n, s.data(slot));
            return slot;
        }
        const int dst = hand0 + next_hand++ % kHandSlots;
        std::copy_n(back.ptr(), n, s.reserve(dst, n));
        return dst;
    }
    // Captures waiting for the end of the run: the slot the GPU copied the activation into, and how
    // to hand it over (View::Whole: as `shape`, e.g. a pattern).
    struct Capture {
        std::string name;
        int slot;
        Shape shape;
        View view;
        int64_t width, D;
    };
    std::vector<Capture> captures = {};
    // A free capture slot of n floats (running what is recorded first if they are all taken).
    int capture_slot(int64_t n) {
        if (captures.size() == static_cast<size_t>(kCaptureSlots)) run();
        const int dst = capture0 + static_cast<int>(captures.size());
        s.reserve(dst, n);
        return dst;
    }
    void defer(const std::string& name, int slot, Shape shape, View view, int64_t width, int64_t D) {
        captures.push_back({name, slot, std::move(shape), view, width, D});
    }
    // Runs what is recorded, then hands the captured activations to their hooks.
    void run() {
        s.run(fp16);
        std::vector<Capture> done;
        done.swap(captures);
        for (const Capture& c : done) {
            const float* src = s.data(c.slot);
            Tensor t = c.view == View::Whole ? Tensor(c.shape, std::vector<float>(src, src + numel(c.shape)))
                                             : view_of(src, c.width, c.view, c.D);
            hook(hooks, c.name, t);
        }
    }
    static int64_t numel(const Shape& shape) {
        int64_t n = 1;
        for (int64_t x : shape) n *= x;
        return n;
    }

    int64_t n_positions = 0;  // the longest sequence write_rows wrote: positions 0 .. n_positions - 1
    // For attention on the GPU: each row's position, segment start, segment's first row and
    // segment's rows (kMeta), and blocks of rows for the tiled kernel (kBlocks, unless `tiled` is false).
    void write_rows(const detail::RowBatch& batch, bool tiled) {
        const int64_t S = std::ssize(batch.start);
        uint32_t* meta = reinterpret_cast<uint32_t*>(s.reserve(kMeta, 4 * R));
        std::vector<uint32_t> blk;
        n_positions = 0;
        for (int64_t sg = 0; sg < S; ++sg) {
            const auto off = static_cast<uint32_t>(batch.offset[sg]), start = static_cast<uint32_t>(batch.start[sg]);
            const auto len = static_cast<uint32_t>(batch.len[sg]);
            n_positions = std::max<int64_t>(n_positions, int64_t(start) + len);
            for (uint32_t r = 0; r < len; ++r) {
                uint32_t* row = meta + 4 * (off + r);
                row[0] = start + r;
                row[1] = start;
                row[2] = off;
                row[3] = len;
            }
        }
        int64_t longest = 0;
        for (int64_t n : batch.len) longest = std::max(longest, n);
        block_rows = detail::Gpu::Seq::attn_block_rows(longest);
        const auto br = static_cast<uint32_t>(block_rows);
        for (int64_t sg = 0; sg < S; ++sg) {
            const auto off = static_cast<uint32_t>(batch.offset[sg]), start = static_cast<uint32_t>(batch.start[sg]);
            const auto len = static_cast<uint32_t>(batch.len[sg]);
            for (uint32_t r = 0; r < len; r += br) blk.insert(blk.end(), {off + r, std::min(br, len - r), start, off});
        }
        blocks = tiled ? static_cast<int64_t>(blk.size() / 4) : 0;
        if (blocks) std::ranges::copy(blk, reinterpret_cast<uint32_t*>(s.reserve(kBlocks, blk.size())));
    }
    // What block `layer`'s attention (and its backward pass) reads besides q, k, v and kMeta: RoPE
    // frequencies (kInvFreq) and ALiBi slopes (kSlopes). Written between runs.
    void write_layer_inputs(int64_t layer) {
        const ModelConfig& c = m.layer_cfg(layer);
        if (c.positional == "rope") {  // as apply_rope_
            const detail::RopeTable t = detail::rope_table(c);
            detail::check_rope_length(c, t, n_positions);
            std::ranges::copy(t.inv_freq, s.reserve(kInvFreq, t.rot / 2));
        }
        if (c.positional == "alibi") {
            const std::vector<float> slopes = alibi_slopes(c.n_heads);
            std::ranges::copy(slopes, s.reserve(kSlopes, c.n_heads));
        }
    }

    static std::string prefix(int64_t layer) { return std::format("blocks.{}.", layer); }
    static bool sandwich_norm(const ModelConfig& c) { return c.norm_position == "sandwich" && c.norm != "none"; }

    // norm(prefix, c, x) of slot `src` into `dst`; with norm "none", returns `src` unchanged.
    int norm(const std::string& name, const ModelConfig& c, int src, int dst) {
        if (c.norm == "none") return src;
        const Tensor& w = m.weight(name + ".weight").f32();
        if (w.numel() != d) throw std::runtime_error(name + ".weight does not match the normalized size");
        const bool layernorm = c.norm == "layernorm";
        const Weight* b = layernorm ? m.weight_or_null(name + ".bias") : nullptr;
        s.reserve(dst, R * d);
        s.norm(src, dst, R, w, b ? &b->f32() : nullptr, layernorm, static_cast<float>(c.norm_eps));
        return dst;
    }
    // out = norm(name, a + b)
    void add_norm(const std::string& name, const ModelConfig& c, int a, int b, int out) {
        s.reserve(out, R * d);
        if (c.norm == "none") {
            s.add(a, b, out, R * d);
            return;
        }
        s.reserve(kTmp, R * d);
        s.add(a, b, kTmp, R * d);
        norm(name, c, kTmp, out);
    }
    void linear(int x, int y, const std::string& name) {
        const Weight& W = m.weight(name + ".weight");
        if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
        s.reserve(y, R * W.shape[0]);
        s.linear(x, y, R, W, m.weight_or_null(name + ".bias"));
    }
    // Model::mlp (with sandwich's ln2_post) of slot `in`; returns the slot holding the result.
    int mlp(int64_t layer, int in, const LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer) + "mlp.";
        const GpuAct act = gpu_act(c.mlp_activation());
        const bool glu = c.mlp != "mlp";
        in = at(prefix(layer) + "hook_mlp_in", in, d, View::Rows, 0, /*keep=*/in != kMlpIn);  // else the residual
        linear(in, ls.pre, p + (glu ? "gate" : "up"));
        const int64_t width = m.weight(p + (glu ? "gate" : "up") + ".weight").shape[0], n = R * width;
        const int pre = at(p + "hook_pre", ls.pre, width);
        s.reserve(kGate, n);
        if (pre != kGate) s.copy(pre, kGate, n);  // keep the pre-activation
        if (glu) {
            linear(in, ls.up, p + "up");
            s.glu(kGate, ls.up, n, act);
        } else {
            s.activate(kGate, n, act);
        }
        linear(at(p + "hook_post", kGate, width), ls.mlp_raw, p + "down");
        const int out = sandwich_norm(c) ? norm(prefix(layer) + "ln2_post", c, ls.mlp_raw, ls.mlp_out) : ls.mlp_raw;
        return par_out = at(prefix(layer) + "hook_mlp_out", out, d);
    }
    // Block `layer` up to the attention projections (into qs, ks, vs); a parallel-residual MLP also
    // runs here, since it reads the block's input. A hook on hook_resid_pre moves ls.x.
    void before_attention(int64_t layer, LayerSlots&& ls) { before_attention(layer, ls); }  // no hooks to move ls.x
    void before_attention(int64_t layer, LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        ls.x = at(p + "hook_resid_pre", ls.x, d);
        const int src = c.norm_position == "post" ? ls.x : norm(p + "ln1", c, ls.x, kAttnIn);
        const int in = at(p + "hook_attn_in", src, d, View::Rows, 0, /*keep=*/src == ls.x);
        linear(in, qs, p + "attn.q_proj");
        linear(in, ks, p + "attn.k_proj");
        linear(in, vs, p + "attn.v_proj");
        if (c.parallel_residual && c.has_mlp()) mlp(layer, norm(p + "ln2", c, ls.x, kMlpIn), ls);
    }
    // Block `layer`'s qk_norm and RoPE of q and k (in slots q, k) on the GPU; returns the slots that
    // hold the results. RoPE needs kMeta and kInvFreq written.
    std::pair<int, int> qk(int64_t layer, int q, int k) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer) + "attn.";
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        if (c.qk_norm) {
            ModelConfig qc = c;
            if (qc.norm == "none") qc.norm = "rmsnorm";
            const bool layernorm = qc.norm == "layernorm";
            auto norm_heads = [&](const std::string& name, int src, int dst, int64_t heads) {
                const Tensor& w = m.weight(p + name + ".weight").f32();
                if (w.numel() != D) throw std::runtime_error(p + name + ".weight does not match the head size");
                const Weight* b = layernorm ? m.weight_or_null(p + name + ".bias") : nullptr;
                s.reserve(dst, R * heads * D);
                s.norm(src, dst, R * heads, w, b ? &b->f32() : nullptr, layernorm, static_cast<float>(qc.norm_eps));
            };
            norm_heads("q_norm", q, qn, H);
            norm_heads("k_norm", k, kn, Hkv);
            q = qn;
            k = kn;
        }
        if (c.positional == "rope") {
            const detail::RopeTable t = detail::rope_table(c);
            s.reserve(qr, R * H * D);
            s.reserve(kr, R * Hkv * D);
            s.rope(q, qr, R, H, D, t.rot, inv_freq, kMeta, false, t.factor);
            s.rope(k, kr, R, Hkv, D, t.rot, inv_freq, kMeta, false, t.factor);
            q = qr;
            k = kr;
        }
        return {q, k};
    }
    detail::Gpu::Seq::AttnShape attn_shape(int64_t layer, int64_t Tp) const {
        const ModelConfig& c = m.layer_cfg(layer);
        const int64_t D = c.head_dim();
        return {R,
                c.n_heads,
                c.kv_heads(),
                D,
                Tp,
                c.sliding_window ? *c.sliding_window : 0,
                static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(D))),
                c.attn_logit_softcap ? static_cast<float>(*c.attn_logit_softcap) : 0.f,
                c.positional == "alibi",
                blocks,
                half_prefix && Tp > 0,
                block_rows};
    }
    // Block `layer`'s attention on the GPU, into zs; with Tp > 0, positions before each segment's
    // start read kPK / kPV (Hkv, Tp, D). Needs write_rows and write_layer_inputs.
    void attention(int64_t layer, int q, int k, int v, int64_t Tp) {
        const detail::Gpu::Seq::AttnShape shape = attn_shape(layer, Tp);
        s.reserve(zs, R * shape.H * shape.D);
        if (lse >= 0) s.reserve(lse, R * shape.H);
        s.attention(q, k, v, zs, Tp ? pk : k, Tp ? pv : v, kMeta, shape.has_slopes ? slopes : kMeta, shape, kBlocks,
                    lse, Tp ? kbase : -1);
    }
    // Block `layer`'s attention backward pass on the GPU, from dL/dz in slot `dz` to the q, k, v
    // projections' gradients (kDQ, kDK, kDV): attention, then RoPE and qk_norm backward. `a` holds
    // the forward pass's slots (attention() with lse, after qk() into a's slots); q, k are the
    // slots qk() returned. Needs write_rows and write_layer_inputs. With pattern >= 0, the slot of the
    // pattern the forward pass applied (a hook edited it): the backward pass then takes it as it is, with dP
    // (dL/dpattern) into slot dp and scratch of the same size in slot ds.
    void attention_backward(int64_t layer, const AttnLayerSlots& a, int q, int k, int dz, int pattern = -1,
                            int dp = -1, int ds = -1) {
        const ModelConfig& c = m.layer_cfg(layer);
        const detail::Gpu::Seq::AttnShape shape = attn_shape(layer, 0);
        const int64_t H = shape.H, Hkv = shape.Hkv, D = shape.D;
        const bool rope = c.positional == "rope";
        // Each step writes the next slot; the last writes kDQ / kDK.
        int dq = c.qk_norm && rope ? kDQ2 : c.qk_norm || rope ? kDQ1 : kDQ;
        int dk = c.qk_norm && rope ? kDK2 : c.qk_norm || rope ? kDK1 : kDK;
        s.reserve(kDelta, R * H);
        s.reserve(dq, R * H * D);
        s.reserve(dk, R * Hkv * D);
        s.reserve(kDV, R * Hkv * D);
        const int slopes = shape.has_slopes ? static_cast<int>(kSlopes) : static_cast<int>(kMeta);
        const detail::Gpu::Seq::AttnSlots in{q, k, a.v, a.z, dz, a.lse, kMeta, slopes, kBlocks};
        if (pattern >= 0) {
            s.reserve(dp, H * R * R);
            s.reserve(ds, H * R * R);
            s.attention_backward_pattern(in, pattern, dp, ds, kDelta, dq, dk, kDV, shape);
        } else {
            s.attention_backward(in, kDelta, dq, dk, kDV, shape);
        }
        const std::string p = prefix(layer) + "attn.";
        report(p + "hook_q", dq, H * D, View::Heads, D);
        report(p + "hook_k", dk, Hkv * D, View::Heads, D);
        report(p + "hook_v", kDV, Hkv * D, View::Heads, D);
        if (rope) {
            const detail::RopeTable t = detail::rope_table(c);
            const int nq = c.qk_norm ? kDQ1 : kDQ, nk = c.qk_norm ? kDK1 : kDK;
            s.reserve(nq, R * H * D);
            s.reserve(nk, R * Hkv * D);
            s.rope(dq, nq, R, H, D, t.rot, kInvFreq, kMeta, /*inverse=*/true, t.factor);
            s.rope(dk, nk, R, Hkv, D, t.rot, kInvFreq, kMeta, /*inverse=*/true, t.factor);
            dq = nq;
            dk = nk;
        }
        if (c.qk_norm) {
            const bool layernorm = c.norm == "layernorm";
            const float eps = static_cast<float>(c.norm_eps);
            s.reserve(kDQ, R * H * D);
            s.reserve(kDK, R * Hkv * D);
            s.norm_backward(a.q0, dq, kDQ, R * H, m.weight(p + "q_norm.weight").f32(), layernorm, eps);
            s.norm_backward(a.k0, dk, kDK, R * Hkv, m.weight(p + "k_norm.weight").f32(), layernorm, eps);
        }
    }
    // The rest of block `layer` from z (slot z, or zs if z < 0): o_proj, residual adds, norms and the
    // MLP. Returns the slot of the block's output (ls.out, unless hook_resid_post moved it).
    int after_attention(int64_t layer, const LayerSlots& ls, int z = -1) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        const bool post = c.norm_position == "post";
        linear(z < 0 ? zs : z, ls.attn_raw, p + "attn.o_proj");
        const int attn =
            at(p + "hook_attn_out",
               c.norm_position == "sandwich" ? norm(p + "ln1_post", c, ls.attn_raw, kAttnNormed) : ls.attn_raw, d);
        s.reserve(ls.out, R * d);
        if (c.parallel_residual) {
            s.add(ls.x, attn, ls.out, R * d);
            const int mid = at(p + "hook_resid_mid", ls.out, d);
            if (c.has_mlp()) s.add(mid, par_out, ls.out, R * d);
            else if (mid != ls.out) s.copy(mid, ls.out, R * d);
            return at(p + "hook_resid_post", ls.out, d);
        }
        s.reserve(ls.mid, R * d);
        if (post) add_norm(p + "ln1", c, ls.x, attn, ls.mid);
        else s.add(ls.x, attn, ls.mid, R * d);
        const int mid = at(p + "hook_resid_mid", ls.mid, d);
        if (!c.has_mlp()) {
            if (mid != ls.out) s.copy(mid, ls.out, R * d);
            return at(p + "hook_resid_post", ls.out, d);
        }
        const int o = mlp(layer, post ? mid : norm(p + "ln2", c, mid, kMlpIn), ls);
        if (post) add_norm(p + "ln2", c, mid, o, ls.out);
        else s.add(mid, o, ls.out, R * d);
        return at(p + "hook_resid_post", ls.out, d);
    }

    // --- backward ---
    // norm_backward(name, c, x, dy) into `dst`; with norm "none", returns `dy`.
    int norm_bwd(const std::string& name, const ModelConfig& c, int x, int dy, int dst) {
        if (c.norm == "none") return dy;
        const Tensor& w = m.weight(name + ".weight").f32();
        if (w.numel() != d) throw std::runtime_error(name + ".weight does not match the normalized size");
        s.reserve(dst, R * d);
        s.norm_backward(x, dy, dst, R, w, c.norm == "layernorm", static_cast<float>(c.norm_eps));
        return dst;
    }
    void linear_t(int dy, int dx, const std::string& name) {
        const Weight& W = m.weight(name + ".weight");
        s.reserve(dx, R * W.shape[1]);
        s.linear_t(dy, dx, R, W);
    }
    // dL/d(MLP input) into kDMlpIn from dL/d(MLP output, after any sandwich norm) in g_out.
    int mlp_bwd(int64_t layer, int g_out, const LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer) + "mlp.";
        const GpuAct act = gpu_act(c.mlp_activation());
        const int g =
            c.norm_position == "sandwich" ? norm_bwd(prefix(layer) + "ln2_post", c, ls.mlp_raw, g_out, kTmpD) : g_out;
        linear_t(g, kDPost, p + "down");
        const int64_t width = m.weight(p + "down.weight").shape[1], n = R * width;
        report(p + "hook_post", kDPost, width);
        s.reserve(kDPre, n);
        if (c.mlp == "mlp") {
            s.act_backward(kDPost, ls.pre, kDPre, n, act);
            report(p + "hook_pre", kDPre, width);
            linear_t(kDPre, kDMlpIn, p + "up");
        } else {
            s.reserve(kDUp, n);
            s.glu_backward(kDPost, ls.pre, ls.up, kDPre, kDUp, n, act);
            report(p + "hook_pre", kDPre, width);
            linear_t(kDPre, kDMlpIn, p + "gate");
            linear_t(kDUp, kTmpC, p + "up");
            s.add(kDMlpIn, kTmpC, kDMlpIn, R * d);
        }
        report(prefix(layer) + "hook_mlp_in", kDMlpIn, d);
        return kDMlpIn;
    }
    // Gradient slots of the hook points between the attention and the block's output.
    struct Reported {
        int mid = -1, attn_out = -1, mlp_out = -1;
    };
    // Block `layer`'s backward pass from dL/d(hook_resid_post) in `g_post` down to dL/dz (into kDZ).
    // Returns the slot that dL/d(hook_resid_pre) adds the attention path to (backward_after).
    int backward_before(int64_t layer, const LayerSlots& ls, const GradSlots& gs, int g_post, Reported& rep) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        const bool post = c.norm_position == "post";
        int base, gx;
        if (c.parallel_residual) {  // mid = x + attn(ln1 x); out = mid + mlp(ln2 x)
            rep.mid = rep.attn_out = g_post;
            base = gx = g_post;
            if (c.has_mlp()) {
                rep.mlp_out = g_post;
                const int t = norm_bwd(p + "ln2", c, ls.x, mlp_bwd(layer, g_post, ls), kTmpB);
                s.reserve(gs.mid, R * d);
                s.add(g_post, t, gs.mid, R * d);
                base = gs.mid;
            }
        } else {
            int g_mid = g_post;
            if (c.has_mlp()) {
                int g_raw = g_post;  // gradient of mid + mlp_out
                if (post) {
                    s.reserve(kTmp, R * d);
                    s.add(ls.mid, ls.mlp_raw, kTmp, R * d);
                    g_raw = norm_bwd(p + "ln2", c, kTmp, g_post, gs.mlp_out);
                }
                rep.mlp_out = g_raw;
                const int d_in = mlp_bwd(layer, g_raw, ls);
                const int t = post ? d_in : norm_bwd(p + "ln2", c, ls.mid, d_in, kTmpB);
                s.reserve(gs.mid, R * d);
                s.add(g_raw, t, gs.mid, R * d);
                g_mid = gs.mid;
            }
            rep.mid = g_mid;
            gx = g_mid;  // gradient of x + attn_out
            if (post) {
                s.reserve(kTmp, R * d);
                s.add(ls.x, ls.attn_raw, kTmp, R * d);
                gx = norm_bwd(p + "ln1", c, kTmp, g_mid, gs.attn_out);
            }
            rep.attn_out = gx;
            base = gx;
        }
        const int g = c.norm_position == "sandwich" ? norm_bwd(p + "ln1_post", c, ls.attn_raw, gx, kTmpD) : gx;
        linear_t(g, dzs, p + "attn.o_proj");
        report(p + "attn.hook_z", dzs, c.n_heads * c.head_dim(), View::RowsHeads, c.head_dim());
        return base;
    }
    // The rest of block `layer`'s backward pass from the q, k, v projections' gradients (kDQ, kDK,
    // kDV): dL/d(hook_resid_pre) = base + the attention path, into gs.resid_pre.
    void backward_after(int64_t layer, const LayerSlots& ls, const GradSlots& gs, int base) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        linear_t(kDQ, kDIn, p + "attn.q_proj");
        linear_t(kDK, kTmpB, p + "attn.k_proj");
        s.add(kDIn, kTmpB, kDIn, R * d);
        linear_t(kDV, kTmpB, p + "attn.v_proj");
        s.add(kDIn, kTmpB, kDIn, R * d);
        report(p + "hook_attn_in", kDIn, d);
        const int t = c.norm_position == "post" ? kDIn : norm_bwd(p + "ln1", c, ls.x, kDIn, kTmpC);
        s.reserve(gs.resid_pre, R * d);
        s.add(base, t, gs.resid_pre, R * d);
    }
};
}  // namespace
#endif

// On the GPU, the residual stream stays on the GPU from `layer` to the end: one submission per block
// (Gpu::Seq), the CPU doing only attention in between (or nothing, with attention on the GPU), and
// hooks getting their activations through GpuBlocks::at. Otherwise (the CPU, or set_gpu_resident
// false) Model::block, its linear layers still on the GPU if batch.gpu is set.
Tensor Model::run_rows(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks, bool last_rows,
                       Tensor* logits, bool logits_last_only) const {
    const int64_t L = cfg_.n_layers, d = cfg_.d_model, S = std::ssize(batch.start);
    (void)logits;  // only with a KV cache on the GPU (macOS)
    (void)logits_last_only;
    // Each segment's last row of a (rows, d) residual stream at `src`.
    auto lasts = [&](const float* src) {
        Tensor out({S, d});
        for (int64_t s = 0; s < S; ++s) {
            const int64_t r = batch.offset[s] + batch.len[s] - 1;
            std::copy(src + r * d, src + (r + 1) * d, out.ptr() + s * d);
        }
        return out;
    };
#ifdef TFI_USE_ACCELERATE
    const int64_t R = batch.rows();
    if (detail::blocks_on_gpu(batch, R)) {
        // A KV cache on the GPU brings its own Gpu::Seq, whose slots hold it; otherwise one for this pass.
        std::optional<detail::Gpu::Seq> own;
        if (!batch.kv) own.emplace(*batch.gpu);
        detail::Gpu::Seq& s = batch.kv ? *batch.kv->seq : *own;
        GpuBlocks gb{*this, s, R, d};
        const bool gpu_attention = detail::batch_attention_on_gpu(*this, batch, layer);
        // A KV cache pass is one submission for every layer: each layer writes its own residual slot
        // (detail::GpuKV::resid), so no run overwrites a slot it read before writing, and fp16's repeat
        // of a run in fp32 finds its inputs intact.
        const bool single_run = batch.kv != nullptr;
        // q, k, v alternate between two sets of slots by layer, so that a run never overwrites the
        // slots its attention reads (Gpu::Seq::run may repeat a run).
        const int qkv[2][3] = {{kQ, kK, kV}, {kQ2, kK2, kV2}};
        auto use_qkv = [&](int64_t l) {
            gb.qs = qkv[l & 1][0];
            gb.ks = qkv[l & 1][1];
            gb.vs = qkv[l & 1][2];
        };
        const bool prefixed = std::ranges::any_of(batch.start, [](int64_t st) { return st > 0; });
        if (gpu_attention) gb.write_rows(batch, detail::gpu_attention_tiled());
        // What layer l's attention reads besides q, k, v: RoPE frequencies, ALiBi slopes and the earlier
        // positions' keys and values (Hkv, Tp, D). Written between runs. Returns Tp (0: none needed).
        auto attention_inputs = [&](int64_t l) {
            const ModelConfig& c = layer_cfgs_[l];
            const int64_t Hkv = c.kv_heads(), D = c.head_dim();
            if (batch.kv) {  // the layer's own slots, written before the pass (below): nothing to copy
                gb.inv_freq = detail::GpuKV::inv_freq(l);
                gb.slopes = detail::GpuKV::slopes(l);
                gb.pk = detail::GpuKV::keys(l);
                gb.pv = detail::GpuKV::values(l);
                return prefixed ? batch.kv->cap * batch.kv->regions : int64_t(0);
            }
            gb.write_layer_inputs(l);
            if (!prefixed) return int64_t(0);
            const detail::KVRef& pk = batch.prefix_k.at(l);
            const detail::KVRef& pv = batch.prefix_v.at(l);
            if (!pk.f || !pv.f || pk.H != Hkv || pk.D != D || pv.H != pk.H || pv.T != pk.T ||
                pv.D != pk.D)  // (a cached run's: float32)
                throw std::invalid_argument(
                    "forward_rows: prefix keys / values must be float32 (Hkv, T, D) for every layer");
            const int64_t n = pk.H * pk.T * pk.D;
            std::copy_n(pk.f, n, s.reserve(kPK, n));
            std::copy_n(pv.f, n, s.reserve(kPV, n));
            return pk.T;
        };
        gb.hooks = hooks;
        gb.fp16 = batch.fp16;
        if (batch.kv) gb.half_prefix = batch.kv->half;
        if (batch.kv && !batch.kbase.empty()) {  // each row's sequence's region start, for attention
            uint32_t* kb = reinterpret_cast<uint32_t*>(s.reserve(detail::GpuKV::kKBase, R));
            for (int64_t sg = 0; sg < S; ++sg)
                for (int64_t r = 0; r < batch.len[sg]; ++r)
                    kb[batch.offset[sg] + r] = static_cast<uint32_t>(batch.kbase.at(sg));
            gb.kbase = detail::GpuKV::kKBase;
        }
        if (batch.kv) {  // every layer's RoPE frequencies and ALiBi slopes, into its own slots
            for (int64_t l = layer; l < L; ++l) {
                const ModelConfig& c = layer_cfgs_[l];
                if (c.positional == "rope") {
                    const detail::RopeTable t = detail::rope_table(c);
                    detail::check_rope_length(c, t, gb.n_positions);
                    std::ranges::copy(t.inv_freq, s.reserve(detail::GpuKV::inv_freq(l), t.rot / 2));
                }
                if (c.positional == "alibi") {
                    const std::vector<float> sl = alibi_slopes(c.n_heads);
                    std::ranges::copy(sl, s.reserve(detail::GpuKV::slopes(l), c.n_heads));
                }
            }
        }
        auto other = [](int slot) { return slot == kResid0 ? static_cast<int>(kResid1) : static_cast<int>(kResid0); };
        // Where layer l writes the residual stream: alternating slots, or (KV cache) a slot of its own.
        auto out_slot = [&](int64_t l, int cur_slot) { return batch.kv ? detail::GpuKV::resid(l) : other(cur_slot); };
        s.reserve(kResid1, R * d);
        if (batch.kv)
            for (int64_t l = layer; l < L; ++l) s.reserve(detail::GpuKV::resid(l), R * d);
        std::copy_n(x.ptr(), R * d, s.reserve(kResid0, R * d));
        int cur = kResid0;
        LayerSlots ls = inference_slots(layer_cfgs_[layer], layer, cur, out_slot(layer, cur));
        use_qkv(layer);
        gb.before_attention(layer, ls);
        if (!gpu_attention) gb.run();
        for (int64_t l = layer; l < L; ++l) {
            const ModelConfig& c = layer_cfgs_[l];
            const std::string p = layer_prefix(l) + "attn.";
            const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
            const int q0 = qkv[l & 1][0], k0 = qkv[l & 1][1], v0 = qkv[l & 1][2];
            int z = kZ;
            if (gpu_attention) {
                // One run per block: q, k, v never leave the GPU unless a hook asks for them.
                const int64_t Tp = attention_inputs(l);  // the previous run has finished with these slots
                auto [q, k] = gb.qk(l, q0, k0);
                q = gb.at(p + "hook_q", q, H * D, GpuBlocks::View::Heads, D);
                k = gb.at(p + "hook_k", k, Hkv * D, GpuBlocks::View::Heads, D);
                const int v = gb.at(p + "hook_v", v0, Hkv * D, GpuBlocks::View::Heads, D);
                if (batch.kv &&
                    batch.kv->half) {  // an fp16 cache: the new rows rounded as stored, before attention reads them
                    s.round_half(k, R * Hkv * D);
                    s.round_half(v, R * Hkv * D);
                }
                const bool pattern = batch.whole && hooks && hooks->matches(p + "hook_pattern");
                gb.lse = pattern ? static_cast<int>(kLse) : -1;
                gb.attention(l, q, k, v, Tp);
                gb.lse = -1;
                if (batch.kv) {  // the new positions' keys and values (as the hooks left them) join the cache
                    const int64_t Tp_all = batch.kv->cap * batch.kv->regions;
                    for (int64_t sg = 0; sg < S; ++sg) {
                        const int64_t at = (batch.kbase.empty() ? 0 : batch.kbase[sg]) + batch.start[sg];
                        s.kv_append(k, gb.pk, batch.len[sg], Hkv, D, Tp_all, at, batch.offset[sg], batch.kv->half);
                        s.kv_append(v, gb.pv, batch.len[sg], Hkv, D, Tp_all, at, batch.offset[sg], batch.kv->half);
                    }
                }
                const detail::Gpu::Seq::AttnShape shape = gb.attn_shape(l, 0);
                if (pattern &&
                    !hooks->changes(p + "hook_pattern")) {  // only captured: computed and handed over after the run
                    const int dst = gb.capture_slot(H * R * R);
                    s.attention_pattern(q, k, kLse, shape.has_slopes ? kSlopes : kLse, dst, shape);
                    gb.defer(p + "hook_pattern", dst, Shape{H, R, R}, GpuBlocks::View::Whole, 0, 0);
                } else if (pattern) {  // the pattern itself, handed to the hook, then z from what it leaves
                    s.reserve(kPattern, H * R * R);
                    s.attention_pattern(q, k, kLse, shape.has_slopes ? kSlopes : kLse, kPattern, shape);
                    gb.run();
                    Tensor P({H, R, R}, std::vector<float>(s.data(kPattern), s.data(kPattern) + H * R * R));
                    hook(hooks, hook_name(l, kHookPattern), P);
                    if (P.numel() != H * R * R)
                        throw std::runtime_error(p + "hook_pattern: a hook changed the activation's size");
                    std::copy_n(P.ptr(), P.numel(), s.data(kPattern));
                    s.attention_apply(kPattern, v, kZ, shape);
                }
                z = gb.at(p + "hook_z", kZ, H * D, GpuBlocks::View::RowsHeads, D);
            } else {  // attention on the CPU, reading q, k, v and writing z in the GPU's slots
                float* zp = s.reserve(kZ, R * H * D);
                attend_into(l, s.data(q0), s.data(k0), s.data(v0), R, hooks, &batch, zp);
            }
            cur = gb.after_attention(l, ls, z);
            if (l + 1 < L) {
                ls = inference_slots(layer_cfgs_[l + 1], l + 1, cur, out_slot(l + 1, cur));
                use_qkv(l + 1);
                gb.before_attention(l + 1, ls);
            }
            if (!single_run) gb.run();
        }
        int64_t logit_rows = 0;
        if (batch.kv && logits) {  // the final norm (every row, for hook_final) and the unembedding, in the same run
            const int fin =
                gb.at("hook_final",
                      cfg_.norm_position == "post" ? cur : gb.norm("ln_final", cfg_, cur, detail::GpuKV::kFinal), d);
            int src = fin;
            logit_rows = logits_last_only ? S : R;  // each segment's last row, or every row
            if (logits_last_only && R > S) {
                s.reserve(detail::GpuKV::kFinalLast, S * d);
                for (int64_t sg = 0; sg < S; ++sg)
                    s.copy(fin, detail::GpuKV::kFinalLast, d, (batch.offset[sg] + batch.len[sg] - 1) * d, sg * d);
                src = detail::GpuKV::kFinalLast;
            }
            s.reserve(detail::GpuKV::kLogits, logit_rows * cfg_.vocab_size);
            s.linear(src, detail::GpuKV::kLogits, logit_rows, unembed(), nullptr);
        }
        if (single_run) gb.run();
        if (logit_rows) {
            const float* lp = s.data(detail::GpuKV::kLogits);
            *logits = Tensor({logit_rows, cfg_.vocab_size}, std::vector<float>(lp, lp + logit_rows * cfg_.vocab_size));
        }
        if (last_rows) return lasts(s.data(cur));  // only what the caller reads comes back
        x = Tensor({R, d});
        std::copy_n(s.data(cur), R * d, x.ptr());
        return last_rows ? lasts(x.ptr()) : x;
    }
#endif
    for (int64_t i = layer; i < L; ++i) x = block(i, std::move(x), hooks, &batch);
    return last_rows ? lasts(x.ptr()) : x;
}

// Forward: one submission per block as in run_rows, keeping each block's activations on the GPU
// (saved_slots) and each segment's attention tensors on the CPU. Backward: from the last layer
// down, one submission per block (GpuBlocks::backward_after of one block, backward_before of the
// next), with attention's backward on the CPU in between, per segment (detail::attention_backward).
// With attention on the GPU (GPU-only mode), attention and its backward pass run inside those
// submissions instead, from what the forward pass kept on the GPU (attn_slots).
std::vector<ActivationCache> Model::gradients_rows(const Tensor& x0, const detail::RowBatch& batch,
                                                   const std::function<Tensor(int64_t, const Tensor&)>& dmetric,
                                                   const NameFilter& filter) const {
    return gradients_rows(x0, batch, dmetric, filter, detail::GradientOptions{});
}

std::vector<ActivationCache> Model::gradients_rows(const Tensor& x0, const detail::RowBatch& batch,
                                                   const std::function<Tensor(int64_t, const Tensor&)>& dmetric,
                                                   const NameFilter& filter, const detail::GradientOptions& opt) const {
#ifdef TFI_USE_ACCELERATE
    const int64_t L = cfg_.n_layers, d = cfg_.d_model, R = batch.rows(), S = std::ssize(batch.start);
    const int64_t V = cfg_.vocab_size;
    if (!batch.gpu) throw std::logic_error("gradients_rows needs a GPU");
    if (x0.dim() != 2 || x0.shape[0] != R || x0.shape[1] != d)
        throw std::invalid_argument("gradients_rows: input must be (rows, d_model), got " + shape_str(x0.shape));
    for (int64_t start : batch.start)
        if (start != 0) throw std::invalid_argument("gradients_rows: every segment must start at position 0");
    const Hooks* hooks = opt.hooks;
    if (hooks && S != 1) throw std::invalid_argument("gradients_rows: hooks need a single sequence");
    auto wanted = [&](const std::string& n) { return !filter || filter(n); };

    detail::Gpu::Seq s(*batch.gpu);
    GpuBlocks gb{*this, s, R, d};
    // Hooks edit activations in their own slots, where the backward pass reads them; hooked and captured
    // activations, the gradients handed over and edited patterns go past every layer's slots.
    gb.hooks = hooks;
    gb.in_place = true;
    gb.hand0 = saved_slots(L - 1).out + 1;
    gb.capture0 = gb.hand0 + kHandSlots;
    int next_slot = gb.capture0 + kCaptureSlots;
    // Slots of their own per layer only for what the backward pass reads, or (gradients) what is handed over;
    // the rest share scratch slots. (Each pass holds every layer's: 5.9 GB for a 0.9B model's 2048 rows.)
    // - o_proj's and the MLP's outputs before the residual add: read back only with post-norm or sandwich norms.
    // - q, k before RoPE: read back only by qk_norm's backward pass, and qk_norm's outputs only without RoPE.
    // - The residual gradients of a layer nobody asked about: two sets, by layer parity (a layer's are read
    //   while the next one down writes its own).
    const int raw_scratch = next_slot++, mlp_scratch = next_slot++, q_scratch = next_slot++, k_scratch = next_slot++;
    const int qn_scratch = next_slot++, kn_scratch = next_slot++, grad_scratch = next_slot;
    next_slot += 8;
    auto layer_slots = [&](int64_t l) {
        const ModelConfig& c = layer_cfgs_[l];
        LayerSlots ls = saved_slots(l);
        if (c.norm_position == "pre") {
            ls.attn_raw = raw_scratch;
            ls.mlp_raw = mlp_scratch;
        }
        return ls;
    };
    auto attn_of = [&](int64_t l) {
        const ModelConfig& c = layer_cfgs_[l];
        AttnLayerSlots a = attn_slots(l);
        if (c.positional == "rope") {
            if (!c.qk_norm) a.q0 = q_scratch, a.k0 = k_scratch;
            a.qn = qn_scratch;
            a.kn = kn_scratch;
        }
        return a;
    };
    auto handed_over = [&](int64_t l) {
        if (l < 0) return false;
        const std::string p = layer_prefix(l);
        for (const char* n : {"hook_resid_pre", "hook_resid_mid", "hook_attn_out", "hook_mlp_out", "hook_resid_post"})
            if (wanted(p + n)) return true;
        return l == 0 && (wanted("hook_embed") || wanted("hook_pos_embed"));
    };
    auto gslots = [&](int64_t l) {
        if (handed_over(l) || handed_over(l - 1)) return grad_slots(l);  // (l - 1's resid_post is l's resid_pre)
        const int b = grad_scratch + 4 * static_cast<int>(l & 1);
        return GradSlots{b, b + 1, b + 2, b + 3};
    };
    const bool gpu_attention = detail::gradient_attention_on_gpu(*this, *batch.gpu);
    if (gpu_attention) gb.write_rows(batch, detail::gpu_attention_tiled());
    std::vector<std::pair<int, int>> qk_slots(L);  // with gpu_attention: each layer's q, k after qk()
    std::vector<int> pattern_slot(L, -1);          // ... and the pattern a hook edited, if one did
    auto use_attn_slots = [&](int64_t l) {
        const AttnLayerSlots a = attn_of(l);
        gb.qs = a.q0;
        gb.ks = a.k0;
        gb.vs = a.v;
        gb.qn = a.qn;
        gb.kn = a.kn;
        gb.qr = a.qr;
        gb.kr = a.kr;
        gb.zs = a.z;
        gb.lse = a.lse;
    };
    // Each segment's attention inputs (after qk_norm and RoPE), pattern, and projections before qk_norm.
    struct SegAttn {
        Tensor q, k, v, pattern, q0, k0;
    };
    std::vector<std::vector<SegAttn>> saved(L, std::vector<SegAttn>(S));
    auto rows = [&](int slot, int64_t seg, int64_t width) {
        Tensor t({batch.len[seg], width});
        const float* src = s.data(slot) + batch.offset[seg] * width;
        std::copy_n(src, t.numel(), t.ptr());
        return t;
    };
    // Gradients the backward pass leaves in slots that later steps overwrite: copied on the GPU when wanted,
    // handed over at the end as run_with_gradients gives them.
    struct Kept {
        std::string name;
        int slot;
        int64_t width;
        GpuBlocks::View view;
        int64_t D;
        Shape whole;  // View::Whole
    };
    std::vector<Kept> kept;
    gb.grad_at = [&](const std::string& name, int slot, int64_t width, int view, int64_t D) {
        if (!wanted(name)) return;
        const int dst = next_slot++;
        s.reserve(dst, R * width);
        s.copy(slot, dst, R * width);
        kept.push_back({name, dst, width, static_cast<GpuBlocks::View>(view), D, {}});
    };

    std::copy_n(x0.ptr(), R * d, s.reserve(layer_slots(0).x, R * d));
    if (gpu_attention) use_attn_slots(0);
    gb.before_attention(0, layer_slots(0));
    if (!gpu_attention) gb.run();
    for (int64_t l = 0; l < L; ++l) {
        const ModelConfig& c = layer_cfgs_[l];
        const std::string p = layer_prefix(l) + "attn.";
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        if (gpu_attention) {  // one run per block, attention's inputs and outputs kept per layer
            const AttnLayerSlots a = attn_of(l);
            gb.write_layer_inputs(l);
            auto [q, k] = gb.qk(l, a.q0, a.k0);
            q = gb.at(p + "hook_q", q, H * D, GpuBlocks::View::Heads, D);
            k = gb.at(p + "hook_k", k, Hkv * D, GpuBlocks::View::Heads, D);
            gb.at(p + "hook_v", a.v, Hkv * D, GpuBlocks::View::Heads, D);
            qk_slots[l] = {q, k};
            gb.attention(l, q, k, a.v, 0);
            if (hooks && hooks->matches(p + "hook_pattern")) {  // from q, k and the lse the attention just wrote
                const detail::Gpu::Seq::AttnShape shape = gb.attn_shape(l, 0);
                const int slopes = shape.has_slopes ? static_cast<int>(kSlopes) : a.lse;
                if (!hooks->changes(p + "hook_pattern")) {  // only captured: handed over after the run
                    const int dst = gb.capture_slot(H * R * R);
                    s.attention_pattern(q, k, a.lse, slopes, dst, shape);
                    gb.defer(p + "hook_pattern", dst, Shape{H, R, R}, GpuBlocks::View::Whole, 0, 0);
                } else {  // the hook's pattern, kept for the backward pass, makes z
                    const int P = pattern_slot[l] = next_slot++;
                    s.reserve(P, H * R * R);
                    s.attention_pattern(q, k, a.lse, slopes, P, shape);
                    gb.run();
                    Tensor pt({H, R, R}, std::vector<float>(s.data(P), s.data(P) + H * R * R));
                    hook(hooks, p + "hook_pattern", pt);
                    if (pt.numel() != H * R * R)
                        throw std::runtime_error(p + "hook_pattern: a hook changed the activation's size");
                    std::copy_n(pt.ptr(), pt.numel(), s.data(P));
                    s.attention_apply(P, a.v, a.z, shape);
                }
            }
            // An edited z goes to a slot of its own: the backward pass reads z = P v.
            const int z = gb.at(p + "hook_z", a.z, H * D, GpuBlocks::View::RowsHeads, D, /*keep=*/true);
            gb.after_attention(l, layer_slots(l), z);
            if (l + 1 < L) {
                use_attn_slots(l + 1);
                gb.before_attention(l + 1, layer_slots(l + 1));
            }
            gb.run();
            continue;
        }
        float* z = s.reserve(kZ, R * H * D);
        for (int64_t g = 0; g < S; ++g) {
            SegAttn& a = saved[l][g];
            const int64_t off = batch.offset[g], n = batch.len[g];
            // This segment's rows of the GPU buffers, read in place.
            const float* q2 = s.data(kQ) + off * H * D;
            const float* k2 = s.data(kK) + off * Hkv * D;
            const float* v2 = s.data(kV) + off * Hkv * D;
            if (c.qk_norm) {
                a.q0 = detail::split_heads(q2, n, H, D);
                a.k0 = detail::split_heads(k2, n, Hkv, D);
            }
            Hooks keep;  // the caller's hooks, then captures of what they leave, for the backward pass
            if (hooks) keep.extend(*hooks);
            auto only = [](std::string name) { return [name](const std::string& n) { return n == name; }; };
            keep.add_capture(only(p + "hook_q"), [&](const std::string&, Tensor& t) { a.q = t; });
            keep.add_capture(only(p + "hook_k"), [&](const std::string&, Tensor& t) { a.k = t; });
            keep.add_capture(only(p + "hook_v"), [&](const std::string&, Tensor& t) { a.v = t; });
            keep.add_capture(only(p + "hook_pattern"), [&](const std::string&, Tensor& t) { a.pattern = t; });
            attend_into(l, q2, k2, v2, n, &keep, nullptr, z + off * H * D);
        }
        gb.after_attention(l, layer_slots(l));
        if (l + 1 < L) gb.before_attention(l + 1, layer_slots(l + 1));
        gb.run();
    }

    // dF/d(hook_resid_post) of the last block, from dmetric's gradient on the logits.
    const int final_resid = layer_slots(L - 1).out;
    float* top = s.reserve(kTop, R * d);
    std::fill_n(top, R * d, 0.f);
    auto softcap_grad = [&](const Tensor& logits, Tensor& dl) {
        if (dl.shape != logits.shape)
            throw std::invalid_argument(std::format("dlogits must have the logits' shape {}, got {}",
                                                    shape_str(logits.shape), shape_str(dl.shape)));
        if (!cfg_.final_logit_softcap) return;
        const float cap = static_cast<float>(*cfg_.final_logit_softcap);
        for (int64_t i = 0; i < dl.numel(); ++i) {
            const float th = logits.data[i] / cap;
            dl.data[i] *= 1.f - th * th;
        }
    };
    if (opt.logits) opt.logits->assign(static_cast<size_t>(S), Tensor());
    if (opt.all_rows) {
        // Every row: the final norm (hook_final among the hook points) and the unembedding on the GPU, the
        // gradient on the logits back through them there.
        const bool post = cfg_.norm_position == "post";
        const int fin_slot = next_slot++, fgrad = next_slot++;
        int fin = post ? final_resid : gb.norm("ln_final", cfg_, final_resid, fin_slot);
        fin = gb.at("hook_final", fin, d, GpuBlocks::View::Rows, 0, /*keep=*/fin == final_resid);
        const int lg = next_slot++;
        s.reserve(lg, R * V);
        s.linear(fin, lg, R, unembed(), nullptr);
        gb.run();
        float* lp = s.data(lg);
        if (cfg_.final_logit_softcap) {
            const float cap = static_cast<float>(*cfg_.final_logit_softcap);
            for (int64_t i = 0; i < R * V; ++i) lp[i] = cap * vm::tanh(lp[i] / cap);
        }
        for (int64_t g = 0; g < S; ++g) {
            const int64_t off = batch.offset[g], n = batch.len[g];
            Tensor logits({n, V}, std::vector<float>(lp + off * V, lp + (off + n) * V));
            Tensor dl = dmetric(g, logits);
            softcap_grad(logits, dl);
            std::copy_n(dl.ptr(), n * V, lp + off * V);  // the gradient takes the logits' place
            if (opt.logits) (*opt.logits)[static_cast<size_t>(g)] = std::move(logits);
        }
        s.reserve(fgrad, R * d);
        s.linear_t(lg, fgrad, R, unembed());
        gb.report("hook_final", fgrad, d);
        const int t = post ? fgrad : gb.norm_bwd("ln_final", cfg_, final_resid, fgrad, kTop);
        if (t != kTop) s.copy(t, kTop, R * d);
    } else {  // only each segment's last row, from the CPU
        for (int64_t g = 0; g < S; ++g) {
            const int64_t r = batch.offset[g] + batch.len[g] - 1;
            Tensor x({1, d});
            std::copy(s.data(final_resid) + r * d, s.data(final_resid) + (r + 1) * d, x.ptr());
            Tensor logits = logits_from_final(final_norm(x));
            Tensor dl = dmetric(g, logits);
            softcap_grad(logits, dl);
            Tensor gr = matmul(dl, unembed());
            if (cfg_.norm_position != "post") gr = detail::norm_backward(*this, "ln_final", cfg_, x, gr);
            std::copy_n(gr.ptr(), d, top + r * d);
            if (opt.logits) (*opt.logits)[static_cast<size_t>(g)] = std::move(logits);
        }
    }

    auto g_post = [&](int64_t l) { return l == L - 1 ? static_cast<int>(kTop) : gslots(l + 1).resid_pre; };
    std::vector<GpuBlocks::Reported> rep(L);
    std::vector<int> base(L);
    std::vector<ActivationCache> out(S);
    const int dscore = next_slot++;  // scratch of the backward passes from an edited pattern
    // With gpu_attention, dL/dz alternates between two slots by layer: a run reads one layer's and
    // writes the next one's (Gpu::Seq::run may repeat a run).
    auto dz_slot = [&](int64_t l) { return gpu_attention && (l & 1) ? static_cast<int>(kDZ2) : static_cast<int>(kDZ); };
    gb.dzs = dz_slot(L - 1);
    base[L - 1] = gb.backward_before(L - 1, layer_slots(L - 1), gslots(L - 1), g_post(L - 1), rep[L - 1]);
    if (!gpu_attention) gb.run();
    for (int64_t l = L - 1; l >= 0; --l) {
        const ModelConfig& c = layer_cfgs_[l];
        const std::string p = layer_prefix(l) + "attn.";
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        if (gpu_attention) {
            const AttnLayerSlots a = attn_of(l);
            const bool want_pattern = S == 1 && wanted(p + "hook_pattern");  // (a whole sequence's pattern)
            gb.write_layer_inputs(l);
            int dp = -1;  // dL/d(hook_pattern), handed over if wanted
            if (want_pattern || pattern_slot[l] >= 0) {
                dp = next_slot++;
                if (want_pattern)
                    kept.push_back({p + "hook_pattern", dp, 0, GpuBlocks::View::Whole, 0, Shape{H, R, R}});
            }
            if (pattern_slot[l] < 0 && want_pattern) {  // from dz and v; the backward pass needs no pattern
                s.reserve(dp, H * R * R);
                s.attention_dpattern(dz_slot(l), a.v, dp, gb.attn_shape(l, 0));
            }
            gb.attention_backward(l, a, qk_slots[l].first, qk_slots[l].second, dz_slot(l), pattern_slot[l], dp,
                                  dscore);
            gb.backward_after(l, layer_slots(l), gslots(l), base[l]);
            if (l > 0) {
                gb.dzs = dz_slot(l - 1);
                base[l - 1] =
                    gb.backward_before(l - 1, layer_slots(l - 1), gslots(l - 1), g_post(l - 1), rep[l - 1]);
            }
            gb.run();
            continue;
        }
        float* dq = s.reserve(kDQ, R * H * D);
        float* dk = s.reserve(kDK, R * Hkv * D);
        float* dv = s.reserve(kDV, R * Hkv * D);
        for (int64_t g = 0; g < S; ++g) {
            const SegAttn& a = saved[l][g];
            const int64_t off = batch.offset[g], n = batch.len[g];
            // dF/dz of this segment, straight from its GPU buffer.
            Tensor dzg({n, H * D}, std::vector<float>(s.data(kDZ) + off * H * D, s.data(kDZ) + (off + n) * H * D));
            detail::AttentionGrads ag = detail::attention_backward(
                *this, l, dzg, a.q, a.k, a.v, a.pattern, c.qk_norm ? &a.q0 : nullptr, c.qk_norm ? &a.k0 : nullptr);
            for (auto [name, t] : {std::pair{"hook_pattern", &ag.pattern}, std::pair{"hook_q", &ag.q},
                                   std::pair{"hook_k", &ag.k}, std::pair{"hook_v", &ag.v}})
                if (wanted(p + name)) out[g][p + name] = std::move(*t);
            std::copy_n(ag.q_proj.ptr(), n * H * D, dq + off * H * D);
            std::copy_n(ag.k_proj.ptr(), n * Hkv * D, dk + off * Hkv * D);
            std::copy_n(ag.v_proj.ptr(), n * Hkv * D, dv + off * Hkv * D);
        }
        gb.backward_after(l, layer_slots(l), gslots(l), base[l]);
        if (l > 0)
            base[l - 1] = gb.backward_before(l - 1, layer_slots(l - 1), gslots(l - 1), g_post(l - 1), rep[l - 1]);
        gb.run();
    }

    auto put = [&](const std::string& name, int slot) {
        if (slot < 0 || !wanted(name)) return;
        for (int64_t g = 0; g < S; ++g) out[g][name] = rows(slot, g, d);
    };
    for (int64_t l = 0; l < L; ++l) {
        const std::string p = layer_prefix(l);
        put(p + "hook_resid_pre", gslots(l).resid_pre);
        put(p + "hook_resid_post", g_post(l));
        put(p + "hook_resid_mid", rep[l].mid);
        put(p + "hook_attn_out", rep[l].attn_out);
        put(p + "hook_mlp_out", rep[l].mlp_out);
    }
    put("hook_embed", gslots(0).resid_pre);
    if (cfg_.embedding_positional()) put("hook_pos_embed", gslots(0).resid_pre);
    for (const Kept& k : kept) {
        const float* src = s.data(k.slot);
        for (int64_t g = 0; g < S; ++g) {
            const int64_t off = batch.offset[g], n = batch.len[g];
            const float* a = src + off * k.width;
            Tensor t;
            if (k.view == GpuBlocks::View::Whole)
                t = Tensor(k.whole, std::vector<float>(src, src + shape_numel(k.whole)));
            else if (k.view == GpuBlocks::View::Heads) t = detail::split_heads(a, n, k.width / k.D, k.D);
            else if (k.view == GpuBlocks::View::RowsHeads)
                t = Tensor({n, k.width / k.D, k.D}, std::vector<float>(a, a + n * k.width));
            else t = Tensor({n, k.width}, std::vector<float>(a, a + n * k.width));
            out[g][k.name] = std::move(t);
        }
    }
    return out;
#else
    (void)x0;
    (void)batch;
    (void)dmetric;
    (void)filter;
    (void)opt;
    throw std::logic_error("gradients_rows needs a GPU (macOS)");
#endif
}

// run_with_gradients on the GPU (grad.cpp asks first), where a single pass of the tokens would run there: one
// sequence through gradients_rows with the caller's hooks and every row's logits. Embeddings are made on the CPU.
bool detail::gradients_on_gpu(const Model& m, const Tokens& tokens, const std::function<Tensor(const Tensor&)>& dlogits,
                              ActivationCache& grads, const NameFilter& filter, ActivationCache* acts,
                              const Hooks* hooks, Tensor& logits) {
#ifdef TFI_USE_ACCELERATE
    const int64_t T = std::ssize(tokens);
    detail::Gpu* g = T ? detail::single_pass_gpu(m, 0, T) : nullptr;
    if (!g) return false;
    Hooks all;  // the caller's hooks, then (acts) a capture of every activation they leave
    if (hooks) all.extend(*hooks);
    ActivationCache cache;
    if (acts) all.add_capture(nullptr, [&cache](const std::string& name, Tensor& t) { cache[name] = t; });
    const Hooks* h = hooks || acts ? &all : nullptr;
    const Tensor x0 = m.embed(tokens, h);
    detail::RowBatch rb;
    rb.gpu = g;
    rb.add(0, T);
    detail::GradientOptions opt;
    opt.hooks = h;
    opt.all_rows = true;
    std::vector<Tensor> lg;
    opt.logits = &lg;
    std::vector<ActivationCache> out =
        m.gradients_rows(x0, rb, [&](int64_t, const Tensor& l) { return dlogits(l); }, filter, opt);
    grads = std::move(out.at(0));
    if (acts) *acts = std::move(cache);
    logits = std::move(lg.at(0));
    return true;
#else
    (void)m, (void)tokens, (void)dlogits, (void)grads, (void)filter, (void)acts, (void)hooks, (void)logits;
    return false;
#endif
}

}  // namespace tfi
