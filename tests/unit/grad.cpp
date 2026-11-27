// Unit tests, group "grad": gradients (run_with_gradients) and attribution patching.
#include "tests.h"

// The backward pass against central finite differences of the metric, at every hook point. The
// direction mixes the computed gradient with noise, so a wrong gradient can't agree by accident.
void test_gradients(const Model& m) {
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

void test_attribution_patching(const Model& m) {
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
void test_attribution_exact_when_linear() {
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

void test_mixed_patching(const Model& m) {
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