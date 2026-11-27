// The unit tests' runner: every test, in order, or the groups and names asked for (--group, --filter).
#include "tests.h"

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
                if (std::find_if(std::begin(kGroups), std::end(kGroups), [&](const char* k) { return g == k; }) ==
                    std::end(kGroups)) {
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
        if ((!only.empty() && std::ranges::find(only, group) == only.end()) || name.find(filter) == std::string::npos)
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
    for (const auto& [name, cfg] : configs()) {
        const Model m = random_model(cfg, 42);
        run("analysis", "cache/hooks/causality: " + name, [&] { test_cache_and_hooks(name, m); });
        run("analysis", "logit lens: " + name, [&] { test_logit_lens(m); });
        run("analysis", "direct logit attribution: " + name, [&] { test_dla(m); });
        run("analysis", "activation patching: " + name, [&] { test_patching(m); });
        run("analysis", "patching edge cases (zero gap, missing sites): " + name, [&] { test_patching_edge_cases(m); });
        run("analysis", "optimized patching (batched, GPU) and ablation = naive: " + name,
            [&] { test_optimized_patching_matches_naive(m); });
        run("analysis", "ablation + head scores: " + name, [&] { test_ablation_and_scores(m); });
        run("analysis", "mean ablation has no future leak: " + name, [&] { test_mean_ablation(m); });
        run("grad", "gradients = finite differences: " + name, [&] { test_gradients(m); });
        run("grad", "attribution patching: " + name, [&] { test_attribution_patching(m); });
        run("grad", "mixed patching: " + name, [&] { test_mixed_patching(m); });
        if (cfg.kv_heads() == cfg.n_heads)
            run("analysis", "composition = direct: " + name, [&] { test_composition_matches_direct(m); });
        run("analysis", "circuits: factored, parallel = direct: " + name, [&] { test_circuits_fast_paths(m); });
        run("analysis", "KV cache: forward_cached = forward, hooks, copies: " + name, [&] { test_forward_cached(m); });
        run("analysis", "KV batch: several sequences in one pass = each alone: " + name,
            [&] { test_forward_cached_batch(m); });
        run("analysis", "generate_batch = generate on each, per-sequence hooks, traces: " + name,
            [&] { test_generate_batch(m); });
        run("analysis", "generate: greedy = argmax, stops, hooks, sampling: " + name, [&] { test_generate(m); });
        run("analysis", "generate's trace = full-pass analyses at each step: " + name, [&] { test_generate_trace(m); });
        run("analysis", "prompt cache: reuses the shared prefix, = a fresh call: " + name,
            [&] { test_generate_prompt_cache(m, random_model(cfg, 43)); });
        run("analysis", "speculative decoding: greedy = plain greedy, hooks, trace, limits: " + name,
            [&] { test_generate_speculative(m, random_model(cfg, 43)); });
        run("analysis", "decode runtime: hooks see what they see without speculation: " + name,
            [&] { test_runtime_hooks(m, random_model(cfg, 43)); });
        run("analysis", "decode runtime: a wrong prediction is replaced by the real hook: " + name,
            [&] { test_runtime_mispredictions(m); });
    }
    run("analysis", "KV cache: 12 layers (two-digit hook names), float32 and fp16, clamped values",
        test_forward_cached_many_layers);
    run("core", "CPU attention in blocks = the whole pattern: windows, ALiBi, softcaps, GQA",
        test_cpu_attention_blocked);
    run("core", "sampler: temperature, top_k, top_p", test_sampler);
    run("core", "sampler: temperature too small for a float's 1 / T", test_tiny_temperature);
    run("analysis", "speculative sampling = the model's distribution", test_speculative_sampling);
    run("analysis", "speculative decoding: draft with a smaller vocabulary", test_speculative_small_draft);
    run("analysis", "decode runtime: one token at a time, stops, failures, prediction, backoff",
        test_runtime_interface);
    run("fp16", "linear layers on 1-16 rows: direct from the weights, fp16 = rounded float32, rows independent",
        test_linear_few_rows);
    run("fp16", "fp16 conversion", test_half_conversion);
    run("fp16", "fp16 weights: reads, linear, matmul", test_fp16_weight);
    run("fp16", "fp16 weights: loading converts as it reads", test_fp16_load);
    for (const auto& [name, cfg] : configs()) {
        const Model m16 = random_model(cfg, 42, WeightFormat::fp16);
        run("fp16", "fp16 weights = float32 with rounded weights: " + name, [&] { test_fp16_model(m16); });
        run("fp16", "fp16 weights: KV cache in fp16 = whole pass with rounded keys and values: " + name, [&] {
            set_device(Device::Cpu);
            test_fp16_kv(m16, cfg);
            set_device(Device::Both);
        });
        run("fp16", "fp16 weights: lens, DLA, patching, optimized = naive: " + name, [&] {
            test_logit_lens(m16);
            test_dla(m16);
            test_patching(m16);
            test_optimized_patching_matches_naive(m16);
        });
        run("fp16", "fp16 weights: gradients, attribution, mixed, circuits: " + name, [&] {
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
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "circuits on the GPU = CPU, chunked, retried, falling back (macOS): " + name,
            [&] { test_gpu_circuits(random_model(cfg, 46)); });
    run("gpu", "GPU block operations (macOS)", test_gpu_seq);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "GPU-resident batched passes = per-layer and plain (macOS): " + name,
            [&] { test_gpu_resident(random_model(cfg, 43)); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "attention on the GPU: batched passes = per-layer and plain (macOS): " + name, [&] {
            detail::set_gpu_attention(true);
            const Model pm = random_model(cfg, 47);
            for (bool tiled : {true, false}) {
                detail::set_gpu_attention_tiled(tiled);
                test_gpu_resident(random_model(cfg, 43));
                test_optimized_patching_matches_naive(pm, /*cpu_parts=*/tiled);  // the CPU's parts once
            }
            detail::set_gpu_attention_tiled(true);
            detail::set_gpu_attention(false);
        });
    run("gpu", "attention on the GPU: long sequences, both kernels = CPU (macOS)", test_gpu_attention_long);
    run("gpu", "tiled GPU attention = attention in double: fp32 and fp16 runs, block sizes, fp16 overflow (macOS)",
        test_gpu_attention_tiled);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu",
            "attention + its backward on the GPU: batched gradients = run_with_gradients, attribution = CPU (macOS): " +
                name,
            [&] {
                detail::set_gpu_attention(true);
                for (bool tiled : {true, false}) {
                    detail::set_gpu_attention_tiled(tiled);
                    test_gpu_gradients(random_model(cfg, 44));
                }
                detail::set_gpu_attention_tiled(true);
                test_gpu_attribution(random_model(cfg, 45));
                detail::set_gpu_attention(false);
            });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "single passes on the GPU in GPU-only mode = CPU, hooks included (macOS): " + name,
            [&] { test_gpu_single_pass(random_model(cfg, 53), 9); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "capture hooks on the GPU next to editing hooks = CPU (macOS): " + name,
            [&] { test_gpu_captures(random_model(cfg, 56), 9); });
    run("gpu", "single passes in both mode with the default thresholds (macOS)", test_gpu_single_pass_defaults);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "KV cache and generate on the GPU = forward (macOS): " + name,
            [&] { test_gpu_forward_cached(random_model(cfg, 57)); });
    run("gpu", "KV cache on the GPU: long sequences, split-key attention = CPU (macOS)", test_gpu_forward_cached_long);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "KV cache in fp16 on the GPU = whole pass with rounded keys and values (macOS): " + name, [&] {
            const Model m16 = random_model(cfg, 42, WeightFormat::fp16);
            if (!m16.gpu()) return;
            for (const Device dev : {Device::Gpu, Device::Both}) {
                set_device(dev);
                detail::set_gpu_min_rows(2);
                test_fp16_kv(m16, cfg);
                detail::set_gpu_min_rows(0);
            }
            set_device(Device::Gpu);  // speculative steps on the fp16 cache, rolled back on the GPU
            test_generate_speculative(m16, random_model(cfg, 43, WeightFormat::fp16));
            set_device(Device::Both);
        });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "GPU work committed in chunks = one command buffer, bit for bit (macOS): " + name,
            [&] { test_gpu_chunked_commits(random_model(cfg, 69)); });
    {
        uint64_t seed = 100;
        for (const auto& [name, cfg] : gpu_block_configs())
            run("gpu", "KV caches on CPU and GPU: random operations = whole passes, copies consistent (macOS): " + name,
                [&, s = seed++] {
                    test_kv_random(random_model(cfg, 63), s);
                    test_kv_random(random_model(cfg, 63, WeightFormat::fp16), s + 1000);  // the cache in fp16
                });
    }
    run("gpu", "KV cache on the GPU with fp16 multiplies: one run, its fp32 repeat finds the inputs intact (macOS)",
        test_gpu_fp16_cached_repeat);
    run("gpu", "single passes on the GPU in GPU-only mode: long sequences (macOS)", [&] {
        for (const auto& [name, cfg] : long_attention_configs()) test_gpu_single_pass(random_model(cfg, 54), 48);
    });
    run("gpu",
        "attention + its backward on the GPU: long sequences, gradients = run_with_gradients, neural accelerators "
        "on and off (macOS)",
        [&] {
            detail::set_gpu_attention(true);
            for (const bool tensor : {true, false}) {
                detail::set_gpu_attention_tensor(tensor);
                for (const auto& [name, cfg] : long_attention_configs())
                    test_gpu_gradients(random_model(cfg, 52), {48, 17, 33, 1});
            }
            detail::set_gpu_attention_tensor(true);
            detail::set_gpu_attention(false);
        });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "GPU batched gradients = run_with_gradients (macOS): " + name,
            [&] { test_gpu_gradients(random_model(cfg, 44)); });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu",
            "run_with_gradients on the GPU = CPU: every hook point, every row's logits, editing hooks (macOS): " + name,
            [&] { test_gpu_run_with_gradients(random_model(cfg, 58), 9); });
    run("gpu", "run_with_gradients on the GPU = CPU: long sequences, editing hooks (macOS)", [&] {
        for (const auto& [name, cfg] : long_attention_configs()) test_gpu_run_with_gradients(random_model(cfg, 59), 48);
    });
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "attribution + mixed patching on the GPU = CPU (macOS): " + name,
            [&] { test_gpu_attribution(random_model(cfg, 45)); });
    run("gpu", "attribution: the corrupted run matches the clean one with gradients on the GPU, fp16 too (macOS)",
        test_attribution_corrupted_run_matches_clean);
    for (const auto& [name, cfg] : gpu_block_configs())
        run("gpu", "fp16 weights on the GPU: = rounded float32 weights; gradients, attribution = CPU (macOS): " + name,
            [&] {
                test_gpu_fp16_matches_rounded(random_model(cfg, 43, WeightFormat::fp16));
                test_gpu_gradients(random_model(cfg, 44, WeightFormat::fp16));
                test_gpu_attribution(random_model(cfg, 45, WeightFormat::fp16));
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