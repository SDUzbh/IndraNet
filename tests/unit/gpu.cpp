// Unit tests, group "gpu": the GPU paths (macOS) and the placement rules.
#include "tests.h"

// The portable half of chunked GPU work: packing groups under a budget, and running chunks through a
// fake backend that logs every call and fails on command. Every chunk is submitted before the first
// wait; only failed chunks run again, each up to twice; a third failure throws GpuFailure.
void test_gpu_chunks() {
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

#ifdef TFI_USE_ACCELERATE
// The GPU linear layer (macOS) against the CPU one, on awkward sizes, with and without a bias.
void test_gpu_linear() {
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
void test_gpu_seq() {
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

void test_gpu_attention_long() {
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
        for (int kernel = 0; kernel < 3; ++kernel) {  // neural accelerators (where available), tiled, per row
            detail::set_gpu_attention_tensor(kernel == 0);
            detail::set_gpu_attention_tiled(kernel < 2);
            check_close(m.forward_rows_final(0, x, rb, nullptr), cpu);
        }
        detail::set_gpu_attention_tensor(true);
        detail::set_gpu_attention_tiled(true);
        detail::set_gpu_attention(false);
        detail::set_gpu_min_rows(0);
        detail::set_gpu_min_work(0);
    }
}

// The tiled attention kernels straight through Gpu::Seq, against attention in double: one segment after Tp earlier
// positions (float32 or fp16, as a KV cache keeps them), head sizes 24 to 128, a window, ALiBi and a softcap,
// blocks of 32, 64 and 128 rows; with the neural accelerators' kernel (where the GPU has them) and without. fp32
// runs within float rounding; fp16 runs (half-precision products) within fp16's; and an fp16 run with a query
// outside fp16's range is repeated in fp32, giving the fp32 run's result.
void test_gpu_attention_tiled() {
    const auto gpu = detail::Gpu::create();
    if (!gpu) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    struct Case {
        int64_t D, H, Hkv, Tp, window;
        float cap;
        bool alibi, half_prefix;
    };
    for (const Case& c : {Case{24, 4, 2, 20, 0, 0.f, false, false}, Case{128, 2, 1, 37, 50, 4.f, true, true},
                          Case{128, 4, 4, 0, 0, 0.f, false, false}, Case{24, 2, 2, 70, 40, 0.f, true, true},
                          Case{64, 4, 2, 45, 0, 0.f, false, false}, Case{32, 2, 1, 33, 20, 3.f, true, true},
                          Case{96, 2, 2, 51, 70, 0.f, false, false}}) {
        const int64_t T = 150, H = c.H, Hkv = c.Hkv, D = c.D, Tp = c.Tp, rep = H / Hkv;
        std::mt19937 rng(static_cast<unsigned>(D + Tp));
        std::normal_distribution<float> nd;
        std::vector<float> q(T * H * D), k(T * Hkv * D), v(T * Hkv * D), pk(Hkv * Tp * D), pv(Hkv * Tp * D);
        for (auto* x : {&q, &k, &v, &pk, &pv})
            for (float& f : *x) f = nd(rng);
        if (c.half_prefix)  // what an fp16 cache holds
            for (auto* x : {&pk, &pv})
                for (float& f : *x) f = detail::half_to_float(detail::float_to_half(f));
        std::vector<float> slopes = detail::alibi_slopes(H);
        const float scale = 1.f / std::sqrt(static_cast<float>(D));
        // Reference: row r at position Tp + r attends to keys lo..Tp + r (prefix first).
        std::vector<double> want(T * H * D);
        for (int64_t r = 0; r < T; ++r)
            for (int64_t h = 0; h < H; ++h) {
                const int64_t pos = Tp + r, g = h / rep, lo = c.window ? std::max<int64_t>(0, pos + 1 - c.window) : 0;
                auto key = [&](int64_t j) { return j < Tp ? &pk[(g * Tp + j) * D] : &k[((j - Tp) * Hkv + g) * D]; };
                auto val = [&](int64_t j) { return j < Tp ? &pv[(g * Tp + j) * D] : &v[((j - Tp) * Hkv + g) * D]; };
                std::vector<double> p(pos + 1 - lo);
                double mx = -1e300, sum = 0;
                for (int64_t j = lo; j <= pos; ++j) {
                    double s = 0;
                    for (int64_t i = 0; i < D; ++i) s += double(q[(r * H + h) * D + i]) * key(j)[i];
                    s *= scale;
                    if (c.cap) s = c.cap * std::tanh(s / c.cap);
                    if (c.alibi) s -= slopes[h] * double(pos - j);
                    mx = std::max(mx, p[j - lo] = s);
                }
                for (double& x : p) sum += (x = std::exp(x - mx));
                for (int64_t i = 0; i < D; ++i) {
                    double o = 0;
                    for (int64_t j = lo; j <= pos; ++j) o += p[j - lo] * val(j)[i];
                    want[(r * H + h) * D + i] = o / sum;
                }
            }
        detail::Gpu::Seq s(*gpu);
        std::ranges::copy(k, s.reserve(1, std::ssize(k)));
        std::ranges::copy(v, s.reserve(2, std::ssize(v)));
        s.reserve(3, T * H * D);
        const int64_t prefix = std::max<int64_t>(1, Hkv * Tp * D);
        for (int slot : {4, 5}) {
            const std::vector<float>& src = slot == 4 ? pk : pv;
            if (c.half_prefix) {
                auto* dst = reinterpret_cast<uint16_t*>(s.reserve(slot, (prefix + 1) / 2));
                for (size_t i = 0; i < src.size(); ++i) dst[i] = detail::float_to_half(src[i]);
            } else {
                std::ranges::copy(src, s.reserve(slot, prefix));
            }
        }
        auto* meta = reinterpret_cast<uint32_t*>(s.reserve(6, 4 * T));
        for (int64_t r = 0; r < T; ++r) {
            const uint32_t row[4] = {uint32_t(Tp + r), uint32_t(Tp), 0, uint32_t(T)};
            std::copy_n(row, 4, meta + 4 * r);
        }
        std::ranges::copy(slopes, s.reserve(7, H));
        auto run = [&](int64_t block_rows, bool fp16, float big = 0.f) {
            std::vector<float> qq = q;
            if (big) qq[(77 * H + 1) * D + 5] = big;
            std::ranges::copy(qq, s.reserve(0, std::ssize(qq)));
            std::vector<uint32_t> blk;
            for (int64_t r = 0; r < T; r += block_rows)
                blk.insert(blk.end(), {uint32_t(r), uint32_t(std::min(block_rows, T - r)), uint32_t(Tp), 0u});
            std::ranges::copy(blk, reinterpret_cast<uint32_t*>(s.reserve(8, std::ssize(blk))));
            const detail::Gpu::Seq::AttnShape shape{
                T, H, Hkv, D, Tp, c.window, scale, c.cap, c.alibi, std::ssize(blk) / 4, c.half_prefix, block_rows};
            s.attention(0, 1, 2, 3, Tp ? 4 : 1, Tp ? 5 : 2, 6, 7, shape, 8);
            s.run(fp16);
            return std::vector<float>(s.data(3), s.data(3) + T * H * D);
        };
        for (bool tensor : {true, false}) {  // the neural accelerators' kernel where available, and the tiled one
            detail::set_gpu_attention_tensor(tensor);
            for (int64_t block_rows : {32, 64, 128})
                for (bool fp16 : {false, true}) {
                    const std::vector<float> z = run(block_rows, fp16);
                    double err = 0;
                    for (size_t i = 0; i < z.size(); ++i) err = std::max(err, std::fabs(z[i] - want[i]));
                    CHECK(err < (fp16 ? 4e-3 : 2e-6));
                }
            // fp16 with a query of 1e5: repeated in fp32, so exactly the fp32 run (whose own query is that 1e5).
            CHECK(run(64, true, 1e5f) == run(64, false, 1e5f));
        }
        detail::set_gpu_attention_tensor(true);
    }
}

// Single passes in GPU-only mode run on the GPU (Model::forward_from through run_rows), hooks
// included: every activation run_with_cache keeps (hook_pattern too) and the logits = the CPU's,
// with and without the resident path; hooks that edit every activation (the pattern too, so z is
// recomputed from it) have the same effect as on the CPU and are called as often. In both mode,
// short sequences stay on the CPU, bit for bit. Activations are compared relative to each one's
// largest value: nearly one-hot patterns (the large random model) magnify rounding differences.
void test_gpu_single_pass(const Model& m, int64_t T) {
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
void test_gpu_captures(const Model& m, int64_t T) {
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
void test_gpu_single_pass_defaults() {
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
void test_gpu_resident(const Model& m) {
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
void test_gpu_gradients(const Model& m, const std::vector<int64_t>& lens) {
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

#ifdef TFI_USE_ACCELERATE
// run_with_gradients on the GPU = on the CPU, at every hook point, for a gradient on every row's logits: without
// hooks, with hooks that edit every activation (q, k, v, the pattern, z and hook_final included), with hooks that
// edit only the patterns, and with the activations wanted (captures). In GPU-only mode, and in both mode where a
// single pass goes to the GPU (its attention there or on the CPU, by head size).
void test_gpu_run_with_gradients(const Model& m, int64_t T) {
    const HalfTolerance half_tolerance(m);
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens t = seq(T, 37);
    auto dlogits = [](const Tensor& logits) {
        Tensor dl(logits.shape);
        for (int64_t i = 0; i < dl.numel(); ++i)
            dl.data[i] = 0.05f * std::sin(0.37f * static_cast<float>(i)) + 0.01f * logits.data[i];
        return dl;
    };
    Hooks edit;  // every activation but the embeddings (made on the CPU either way), each element differently
    edit.add_filter([](const std::string& n) { return n != "hook_embed" && n != "hook_pos_embed"; },
                    [](const std::string&, Tensor& a) {
                        for (int64_t i = 0; i < a.numel(); ++i) a.data[i] *= 1.f + 0.01f * static_cast<float>(i % 5);
                    });
    Hooks patterns;  // only the patterns: rows no longer summing to 1
    patterns.add_filter([](const std::string& n) { return n.ends_with("hook_pattern"); },
                        [](const std::string&, Tensor& a) {
                            for (int64_t i = 0; i < a.numel(); ++i)
                                a.data[i] *= 0.9f + 0.05f * static_cast<float>(i % 3);
                        });
    const double tol = tol_for(m, 1e-3);
    auto near = [&](const Tensor& got, const Tensor& want, const std::string& what) {
        CHECK(got.shape == want.shape);
        if (got.shape != want.shape) return;
        const bool ok = max_abs_diff(got, want) <= tol * std::max(1.f, max_abs(want));
        CHECK(ok);
        if (!ok) std::printf("     %s: %g off (max %g)\n", what.c_str(), max_abs_diff(got, want), max_abs(want));
    };
    auto same = [&](const ActivationCache& got, const ActivationCache& want) {
        CHECK(got.size() == want.size());
        for (const auto& [name, w] : want) {
            CHECK(got.contains(name));
            if (got.contains(name)) near(got.at(name), w, name);
        }
    };
    const auto only_some = [](const std::string& n) { return n.ends_with("hook_resid_pre") || n.ends_with("hook_q"); };
    for (const Hooks* h : std::initializer_list<const Hooks*>{nullptr, &edit, &patterns})
        for (const bool acts : {false, true})
            for (const bool filtered : {false, true}) {
                const NameFilter filter = filtered ? NameFilter(only_some) : NameFilter();
                ActivationCache cg, ca;
                set_device(Device::Cpu);
                const Tensor cl = m.run_with_gradients(t, dlogits, cg, filter, acts ? &ca : nullptr, h);
                CHECK(cg.contains("blocks.0.attn.hook_q") && (filtered || cg.contains("blocks.0.attn.hook_pattern")));
                for (const Device dev : {Device::Gpu, Device::Both}) {
                    set_device(dev);
                    detail::set_gpu_min_rows(1);  // both mode: the GPU, for this short pass
                    detail::set_gpu_min_work(1);
                    ActivationCache gg, ga;
                    const Tensor gl = m.run_with_gradients(t, dlogits, gg, filter, acts ? &ga : nullptr, h);
                    detail::set_gpu_min_rows(0);
                    detail::set_gpu_min_work(0);
                    near(gl, cl, "logits");
                    CHECK(max_abs_diff(gl, cl) > 0.f);  // the GPU really ran
                    same(gg, cg);
                    if (acts) same(ga, ca);
                }
                set_device(Device::Both);
            }
}
#endif

#ifdef TFI_USE_ACCELERATE
// Attribution patching with step 0's gradients on the small-batch path while run_with_gradients goes to the GPU
// (layers big enough for a short single pass there): the corrupted activations still come from a forward pass, as
// the clean ones do, so its metric is exact patching's and the positions before the corruption have no effect, bit
// for bit, with fp16 weights too (where the gradient pass's float32 multiplies differ from a forward pass's).
void test_attribution_corrupted_run_matches_clean() {
    const ModelConfig cfg = tiny([](ModelConfig& c) {
        c.d_model = 512;
        c.n_heads = 8;
        c.d_mlp = 2048;
        c.n_layers = 2;
    });
    for (const WeightFormat f : {WeightFormat::fp32, WeightFormat::fp16}) {
        const Model m = random_model(cfg, 71, f);
        if (!m.gpu()) {
            std::printf("     (no Metal device: GPU path not tested)\n");
            return;
        }
        set_device(Device::Gpu);
        m.forward(seq(8, 3));  // the GPU has started
        set_device(Device::Both);
        const Tokens clean = seq(8, 6);
        Tokens corrupt = clean;
        corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
        CHECK(detail::gradients_go_to_gpu(m, 7));  // the case under test
        const PatchResult exact = patch_residual(m, clean, corrupt, "resid_pre", 6, 1, 2);
        const PatchResult atp = attribution_patch_residual(m, clean, corrupt, "resid_pre", 6, 1, 2);
        CHECK(atp.clean_metric == exact.clean_metric);
        CHECK(atp.corrupt_metric == exact.corrupt_metric);
        for (int64_t i = 0; i < m.cfg().n_layers; ++i)
            for (int64_t p = 0; p < 3; ++p) CHECK(atp.effect.at({i, p}) == 0.f);
    }
}
#endif

#ifdef TFI_USE_ACCELERATE
// Attribution and mixed patching with their gradient runs batched on the GPU: close to the CPU's
// results for any batch size, the same for any thread count, and the CPU tests' invariants hold.
void test_gpu_attribution(const Model& m) {
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
void test_gpu_fp16_matches_rounded(const Model& m16) {
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
void test_gpu_work_threshold() {
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
void test_gpu_batch() {
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
void test_gpu_circuits(const Model& m) {
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
void test_single_pass_rule() {
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
void test_device_modes() {
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
void test_gpu_mapped_weights() {
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
void test_gpu_wide_heads() {
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

#ifdef TFI_USE_ACCELERATE
// forward_cached and generate on the GPU (macOS): GPU-only mode, where every cached step runs there,
// and both mode with chunks of 2 rows or more on the GPU.
void test_gpu_forward_cached(const Model& m) {
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
    test_generate_prompt_cache(m, random_model(m.cfg(), 43));  // the cache's GPU copy moves between calls
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
void test_gpu_chunked_commits(const Model& m) {
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
void test_kv_random(const Model& m, uint64_t seed) {
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
void test_gpu_fp16_cached_repeat() {
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
void test_gpu_forward_cached_long() {
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