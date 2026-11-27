// Unit tests, group "fp16": half-precision weights and caches.
#include "tests.h"

// The vectorizable exp, erf and tanh (src/support/vmath.h) against double precision, at every 16th float
// of their working range, plus special values.
// --- fp16 weights ----------------------------------------------------------------------------

// fp16 <-> float32: every fp16 value survives the round trip; rounding is to nearest, ties to even;
// the bulk (vectorized) conversions match the scalar ones, and on arm64 the hardware's.
void test_half_conversion() {
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
void test_fp16_weight() {
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

// A model with fp16 weights computes what the float32 model with the rounded weights computes.
void test_fp16_model(const Model& m16) {
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

void test_fp16_load() {
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

// Test groups, for running part of the suite:
//   core      json, tensors, matmul, vector math, weight shapes
//   analysis  per model type: cache and hooks, lens, DLA, patching, ablation, circuits
//   grad      gradients against finite differences, attribution and mixed patching
//   fp16      fp16 conversion and weights, and the analysis invariants with fp16 weights
//   gpu       GPU work: chunking and thresholds everywhere; kernels, resident passes, attention on macOS
// Linear layers on 1-4 rows multiply straight from the weights (src/core/weight.cpp): = a double-precision
// reference, for float32 and fp16 weights and sizes with a tail past the 8-wide steps; fp16 weights =
// float32 weights holding the same rounded values, bit for bit (on 1-4 rows); the same with one thread
// or many.
void test_linear_few_rows() {
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

// With fp16 weights the KV cache holds fp16 values: cached decoding (in chunks, and a batch) = the whole pass
// with keys and values rounded to fp16, up to an fp16 step where the two round differently; keys() are
// fp16 values; overflow is clamped, not infinite; a float32 model can't continue the cache, nor the reverse.
void test_fp16_kv(const Model& m16, const ModelConfig& cfg) {
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