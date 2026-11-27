// Unit tests, group "analysis": the analysis tools: lens, attribution, patching, ablation, circuits.
#include "tests.h"

void test_cache_and_hooks(const std::string& name, const Model& m) {
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

void test_logit_lens(const Model& m) {
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

void test_dla(const Model& m) {
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

void test_patching(const Model& m) {
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

void test_patching_edge_cases(const Model& m) {
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

// cpu_parts false leaves out what doesn't involve the GPU (forward_from, the CPU's batched passes), for
// callers that already ran it on the same model and only change a GPU setting.
void test_optimized_patching_matches_naive(const Model& m, bool cpu_parts) {
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

void test_patch_gap_threshold() {
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

void test_mean_ablation(const Model& m) {
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

void test_ablation_and_scores(const Model& m) {
    const Tensor abl = head_ablation(m, seq(12, 7), Ablation::Mean, {seq(12, 8), seq(12, 9)});
    CHECK(abl.shape == (Shape{2, 4}));
    for (float v : abl.data) CHECK(std::isfinite(v));
    const HeadScores s = head_scores(m, 6, 2, 0);
    CHECK(s.induction.shape == (Shape{2, 4}));
    for (float v : s.previous_token.data) CHECK(v >= 0.f && v <= 1.f + 1e-5f);
    CHECK(s.top_two_hop.size() == 5);
}

void test_composition_matches_direct(const Model& m) {
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

// circuits: factored OV paths = the direct product (up to rounding at near-ties); the all-pairs
// scores = one path at a time, bit for bit; nothing depends on the thread count.
void test_circuits_fast_paths(const Model& m) {
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

void test_copying_and_positional() {
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

void test_post_norm_dla_refused() {
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

void test_nan_never_matches() {
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
void test_heterogeneous_heads() {
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

// forward_cached, in chunks of every size, gives forward()'s rows; the cache holds hook_k / hook_v; a
// copy or a truncated cache continues on its own; hooks see the new rows and their edits carry over.
void test_forward_cached(const Model& m) {
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
void test_forward_cached_batch(const Model& m) {
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
void test_generate_batch(const Model& m) {
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

// KVCache::clamped(): an fp16 cache counts the keys and values it cut off at fp16's limit as it stored them, from CPU
// and GPU steps alike (on the current device); truncating drops the dropped positions' counts. A float32 cache, or
// one whose values all fit, counts none. generate() and generate_batch() pass the count on (kv_clamped).
void test_kv_clamped(const Model& m16, const Model& m32) {
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
void test_generate(const Model& m) {
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
void test_forward_cached_many_layers() {
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

// A prompt cache: each call reuses the longest prefix its prompt shares with the cache's text (at least the last
// token runs) and leaves the cache holding its prompt and new tokens; the output is a fresh call's (up to rounding:
// a differing token must be a near tie in the full pass's logits). A call stopped early leaves what it ran; another
// model starts the cache over; editing hooks are refused, capture-only hooks allowed; speculation keeps it.
void test_generate_prompt_cache(const Model& m, const Model& other) {
    GenerateOptions o;
    o.max_new_tokens = 6;
    o.stop_at_eos = false;
    GenerateOptions fresh = o;
    // got and want greedy from prompt: equal, or where they first differ, a near tie in the full pass.
    auto agree = [&](const Tokens& prompt, const Tokens& got, const Tokens& want) {
        CHECK(got.size() == want.size());
        Tokens text = prompt;
        for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
            if (got[i] != want[i]) {
                const Tensor lg = m.forward(text);
                const float* last = lg.ptr() + (lg.shape[0] - 1) * lg.shape[1];
                CHECK(std::fabs(last[got[i]] - last[want[i]]) <=
                      tol_for(m, 1e-4) * std::max(1.f, std::fabs(last[want[i]])));
                return;
            }
            text.push_back(got[i]);
        }
    };
    PromptCache pc;
    o.prompt_cache = &pc;
    const Tokens a = seq(7, 3);
    const Tokens out1 = generate(m, a, o);
    CHECK(pc.reused == 0 && pc.model == &m);
    agree(a, out1, generate(m, a, fresh));
    Tokens text = a;  // what the call ran: the prompt and its new tokens but the last
    text.insert(text.end(), out1.begin(), out1.end() - 1);
    CHECK(pc.tokens == text && pc.kv.size() == std::ssize(text));
    // The next turn: the earlier text, the last reply token and a new message.
    Tokens turn = a;
    turn.insert(turn.end(), out1.begin(), out1.end());
    turn.insert(turn.end(), {5, 9, 2});
    const Tokens out2 = generate(m, turn, o);
    CHECK(pc.reused == std::ssize(text));
    agree(turn, out2, generate(m, turn, fresh));
    // The same prompt again: all but its last token.
    agree(turn, generate(m, turn, o), out2);
    CHECK(pc.reused == std::ssize(turn) - 1);
    // A prompt that differs at position 4: positions 0..3; a shorter one: its prefix.
    Tokens branch = turn;
    branch[4] = (branch[4] + 1) % m.cfg().vocab_size;
    agree(branch, generate(m, branch, o), generate(m, branch, fresh));
    CHECK(pc.reused == 4);
    const Tokens head(branch.begin(), branch.begin() + 3);
    agree(head, generate(m, head, o), generate(m, head, fresh));
    CHECK(pc.reused == 2);
    // Stopped after two tokens: the prompt and the first (the second never ran).
    int calls = 0;
    o.on_token = [&](int64_t) { return ++calls < 2; };
    const Tokens two = generate(m, turn, o);
    o.on_token = nullptr;
    Tokens ran = turn;
    ran.push_back(two.at(0));
    CHECK(two.size() == 2 && pc.tokens == ran && pc.kv.size() == std::ssize(ran));
    // No new tokens: the cache is left as it was.
    o.max_new_tokens = 0;
    CHECK(generate(m, a, o).empty() && pc.tokens == ran);
    o.max_new_tokens = 6;
    // Another model starts it over.
    generate(other, turn, o);
    CHECK(pc.reused == 0 && pc.model == &other);
    // Editing hooks are refused (the cache untouched); capture-only hooks run on the positions computed.
    Hooks edit;
    edit.add("blocks.0.hook_attn_out", [](const std::string&, Tensor& t) { std::ranges::fill(t.data, 0.f); });
    CHECK_THROWS(generate(m, turn, o, &edit));
    CHECK(pc.model == &other);
    generate(m, turn, o);
    int64_t rows = 0;
    Hooks cap;
    cap.add_capture([](const std::string& n) { return n == "blocks.0.hook_resid_pre"; },
                    [&](const std::string&, Tensor& t) { rows += t.shape[0]; });
    agree(turn, generate(m, turn, o, &cap), generate(m, turn, fresh));
    CHECK(pc.reused == std::ssize(turn) - 1 && rows == 1 + 5);  // the prompt's last token, then 5 new ones
    // With speculation (prompt lookup): the same tokens as plain greedy.
    o.speculative.draft_tokens = 3;
    agree(branch, generate(m, branch, o), generate(m, branch, fresh));
    CHECK(pc.reused == 4);
    o.speculative.draft_tokens = 0;
}

void test_generate_trace(const Model& m) {
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
void test_generate_speculative(const Model& m, const Model& other) {
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
void test_speculative_sampling() {
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
void test_speculative_small_draft() {
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

// A hook that changes what it does once predictions trust it: the guesses in that pass hold the wrong edit, and
// the real hook's output must replace it. The model is its own draft, so proposals survive until the edit starts.
void test_runtime_mispredictions(const Model& m) {
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
void test_runtime_hooks(const Model& m, const Model& other) {
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

void test_runtime_interface() {
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