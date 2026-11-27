// Unit tests, group "core": tensors, weights, configs, files, tokenizers, the CPU forward pass, generation.
#include "tests.h"

// RoPE scaling: what the config accepts, and the length limit of "dynamic" and "longrope", on the CPU
// and (macOS) the GPU. The frequencies themselves are checked against PyTorch and transformers by the
// golden tests (python/tests/test_golden.py, test_hf.py).
void test_rope_scaling() {
    auto parse = [](const char* s) { return RopeScaling::from_json(json::parse(s)); };
    CHECK_THROWS(parse(R"({"type": "ntk", "factor": 2})"));
    CHECK_THROWS(parse(R"({"type": "linear"})"));
    CHECK_THROWS(parse(R"({"type": "linear", "factor": 0})"));
    CHECK_THROWS(parse(R"({"type": "linear", "factor": 2, "beta_fast": 3})"));
    CHECK_THROWS(parse(
        R"({"type": "llama3", "factor": 8, "low_freq_factor": 4, "high_freq_factor": 1, "original_max_position_embeddings": 8})"));
    const RopeScaling y =
        parse(R"({"type": "yarn", "factor": 4, "original_max_position_embeddings": 8, "beta_fast": null})");
    CHECK(y.beta_fast == 32.0 && y.truncate);

    const ModelConfig c = tiny([&](ModelConfig& m) {
        m.rope_scaling = parse(R"({"type": "dynamic", "factor": 2, "original_max_position_embeddings": 10})");
    });
    const detail::RopeTable plain = detail::rope_table(tiny()), dyn = detail::rope_table(c);
    CHECK(dyn.inv_freq == plain.inv_freq && dyn.factor == 1.f && dyn.max_len == 10);  // plain RoPE up to its limit
    const Model m = random_model(c, 7);
    ActivationCache cache;
    CHECK(m.run_with_cache(seq(10, 1), cache).shape == (Shape{10, 50}));
    CHECK_THROWS(m.run_with_cache(seq(11, 1), cache));
    CHECK_THROWS(m.forward(seq(11, 1)));
    const ModelConfig wrong = tiny([&](ModelConfig& m) {  // short factors for 4 frequency pairs, head size 4 has 2
        m.rope_scaling =
            parse(R"({"type": "longrope", "short_factor": [1, 2, 3, 4], "original_max_position_embeddings": 10})");
    });
    CHECK_THROWS(detail::rope_table(wrong));
#ifdef TFI_USE_ACCELERATE
    if (m.gpu()) {  // (a Mac can have no Metal device)
        const Device before = device();
        set_device(Device::Gpu);  // single passes on the GPU: the GPU path checks the length too
        CHECK(m.run_with_cache(seq(10, 1), cache).shape == (Shape{10, 50}));
        CHECK_THROWS(m.run_with_cache(seq(11, 1), cache));
        set_device(before);
    }
#endif
}

void test_json() {
    const json::Value v = json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": null, "d": "x\"é"}, "e": true})");
    CHECK(v.at("a").array.size() == 3);
    CHECK_NEAR(v.at("a").array[2].as_number(), -300, 1e-9);
    CHECK(v.at("b").at("c").is_null());
    CHECK(v.at("b").at("d").as_string() == "x\"\xc3\xa9");
    CHECK(v.at("e").as_bool());
    CHECK_THROWS(json::parse("{\"a\": }"));
}

void test_reshape_and_linear() {
    // reshape on an rvalue takes over the buffer; on an lvalue it copies and leaves the original alone.
    Tensor t({2, 3}, {0, 1, 2, 3, 4, 5});
    const float* buf = t.ptr();
    Tensor moved = std::move(t).reshape({3, -1});
    CHECK(moved.ptr() == buf);
    CHECK(moved.shape == (Shape{3, 2}));
    const Tensor copy = moved.reshape({6});
    CHECK(copy.ptr() != moved.ptr());
    CHECK(moved.shape == (Shape{3, 2}) && moved.data[5] == 5.f);
    CHECK(copy.shape == (Shape{6}) && copy.data[5] == 5.f);
    CHECK_THROWS(moved.reshape({4}));

    // linear on a 3-D input with bias, against a direct computation.
    std::mt19937_64 rng(2);
    std::normal_distribution<float> n(0, 1);
    Tensor x({2, 3, 4}), W({5, 4}), b({5});
    for (Tensor* p : {&x, &W, &b})
        for (float& v : p->data) v = n(rng);
    const Tensor y = linear(x, W, &b);
    CHECK(y.shape == (Shape{2, 3, 5}));
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t t2 = 0; t2 < 3; ++t2)
            for (int64_t o = 0; o < 5; ++o) {
                float s = b.data[o];
                for (int64_t k = 0; k < 4; ++k) s += x.at({i, t2, k}) * W.at({o, k});
                CHECK_NEAR(y.at({i, t2, o}), s, 1e-4);
            }
    CHECK(linear(x, W).shape == (Shape{2, 3, 5}));
    CHECK_THROWS(linear(x, Tensor({5, 3})));
}

void test_matmul() {
    std::mt19937_64 rng(1);
    std::normal_distribution<float> n(0, 1);
    Tensor A({5, 7}), B({7, 3});
    for (float& v : A.data) v = n(rng);
    for (float& v : B.data) v = n(rng);
    const Tensor C = matmul(A, B), Ct = matmul(A, transpose2d(B), true);
    for (int64_t i = 0; i < 5; ++i)
        for (int64_t j = 0; j < 3; ++j) {
            float s = 0;
            for (int64_t k = 0; k < 7; ++k) s += A.at({i, k}) * B.at({k, j});
            CHECK_NEAR(C.at({i, j}), s, 1e-4);
            CHECK_NEAR(Ct.at({i, j}), s, 1e-4);
        }
    CHECK_THROWS(matmul(A, A));
}

// Every weight must have the shape the config implies: a model with any one weight a row (or an
// element) short, or missing, is refused when it is built, not read out of bounds later.
// bias as parts: parsed from true / false / a list; a model has exactly the biases its config names
// (a stray one is refused rather than used unannounced); and a qkv-only model (Qwen2's) works.
void test_bias_parts() {
    Biases b;
    b = true;
    CHECK(b.str() == "all" && b.has("norm"));
    CHECK(Biases::from_json(json::parse("false")).str() == "none");
    CHECK(Biases::from_json(json::parse(R"(["mlp", "qkv"])")).str() == "qkv,mlp");
    CHECK_THROWS(Biases::from_json(json::parse(R"(["qkv", "qkv"])")));
    CHECK_THROWS(Biases::from_json(json::parse(R"(["attn"])")));
    CHECK_THROWS(Biases::from_json(json::parse("1")));

    const ModelConfig qkv = tiny([](ModelConfig& c) { c.bias = Biases::from_json(json::parse(R"(["qkv"])")); });
    const Model m = random_model(qkv, 3);
    CHECK(m.weights().contains("blocks.0.attn.k_proj.bias") && !m.weights().contains("blocks.0.attn.o_proj.bias") &&
          !m.weights().contains("blocks.0.mlp.up.bias"));
    // q, k and v biases change the output (they are used), and match the computation spelled out
    std::map<std::string, Tensor> w;
    for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
    auto zeroed = w;
    for (auto& [name, w] : zeroed)
        if (name.find("_proj.bias") != std::string::npos) std::ranges::fill(w.data, 0.f);
    const Tokens t = seq(9, 4);
    CHECK(max_abs_diff(Model(qkv, zeroed).forward(t), m.forward(t)) > 1e-3f);
    // a bias the config doesn't name: refused, whichever part it belongs to
    for (const char* extra : {"blocks.0.attn.o_proj.bias", "blocks.1.mlp.down.bias"}) {
        auto more = w;
        more[extra] = Tensor({16}, std::vector<float>(16, 0.f));
        CHECK_THROWS(Model(qkv, more));
    }
    auto fewer = w;
    fewer.erase("blocks.1.attn.v_proj.bias");
    CHECK_THROWS(Model(qkv, fewer));
    // a layer's own setting decides for that layer
    const ModelConfig per_layer =
        tiny([](ModelConfig& c) { c.layer_overrides[1] = json::parse(R"({"bias": ["o"]})"); });
    const Model pl = random_model(per_layer, 5);
    CHECK(pl.weights().contains("blocks.1.attn.o_proj.bias") && !pl.weights().contains("blocks.0.attn.o_proj.bias"));
}

// The tokenizer's parts on their own (the whole tokenizer is checked against Hugging Face's
// tokenizers library by python/tests/test_hf.py): UTF-8, NFC / NFD, the split-pattern regex, and BPE
// rules on small hand-written tokenizer.json files.
void test_tokenizer_parts() {
    using namespace detail;
    // UTF-8: round trip, and invalid input refused (stray continuation, overlong, surrogate, truncated)
    const std::string mixed = "a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80";  // a é 中 😀
    CHECK(utf8_decode(mixed) == U32String({U'a', 0xE9, 0x4E2D, 0x1F600}) && utf8_encode(utf8_decode(mixed)) == mixed);
    for (const char* bad : {"\x80", "\xC0\xAF", "\xED\xA0\x80", "\xE4\xB8"}) CHECK_THROWS(utf8_decode(bad));
    // categories and White_Space
    CHECK(std::string(unicode_category(U'A')) == "Lu" && unicode_letter(0x4E2D) && unicode_number(0x2163) &&
          !unicode_letter(U'1'));
    CHECK(unicode_white_space(0x3000) && unicode_white_space(0x85) && !unicode_white_space(0x200B));
    // NFC / NFD: composition, Hangul, mark order, an excluded composition, and blocking
    CHECK(unicode_nfc(U"é") == U"é" && unicode_nfd(U"é") == U"é");
    CHECK(unicode_nfc(U"각") == U"각" && unicode_nfd(U"각") == U"각");
    CHECK(unicode_nfd(U"ạ́") == U"ạ́");  // dot below (220) before acute (230)
    CHECK(unicode_nfc(U"ạ́") == U"ạ́");  // ạ then the acute
    CHECK(unicode_nfc(U"क़") == U"क़");  // excluded: stays decomposed
    CHECK(unicode_nfc(U"plain ascii") == U"plain ascii");

    // regex: the subset tokenizer patterns use
    auto spans = [](const char* pattern, const char32_t* text) {
        std::vector<std::u32string> out;
        const U32String s(text);
        for (const auto& [begin, end] : Regex(pattern).find_all(s)) out.push_back(s.substr(begin, end - begin));
        return out;
    };
    using V = std::vector<std::u32string>;
    CHECK(spans(R"(\p{N}{1,3})", U"12345") == (V{U"123", U"45"}));
    CHECK(spans(R"((?i:'s|'t)|\p{L}+)", U"IT'S it's") == (V{U"IT", U"'S", U"it", U"'s"}));
    CHECK(spans(R"(\s+(?!\S)|\s+)", U"a   b") == (V{U"  ", U" "}));  // the lookahead leaves one space for " b"
    CHECK(spans(R"([^\r\n\p{L}\p{N}]?\p{L}+)", U"中文 été") == (V{U"中文", U" été"}));
    CHECK(spans(R"(a+?b|a)", U"aab") == (V{U"aab"}));
    CHECK(spans(R"(x\x{263A}é[^a-c])", U"x☺éd") == (V{U"x☺éd"}));
    CHECK(spans(R"(\s+)", std::u32string(100000, U' ').c_str()).size() == 1);  // long runs: no deep recursion
    for (const char* bad : {"(a", "a{2,1}", "\\q", "(?<name>a)", "a++", "[b-a]", "^a"}) CHECK_THROWS(Regex(bad));

    // BPE: merges by rank (leftmost on ties), byte fallback, unknowns, specials around the text
    auto tokenizer = [](const std::string& extra, const std::string& model_extra = "") {
        return Tokenizer::from_json(
            R"({"model": {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "bc": 5, "abc": 6,
            "<0xC3>": 7, "<0xA9>": 8, "▁": 9, "<s>": 10, "aa": 11}, "merges": ["a b", "ab c", "b c", "a a"])" +
            model_extra + "}" + extra + "}");
    };
    auto ids = [](const Tokenizer& t, const std::string& s, bool special = true) { return t.encode(s, special); };
    const Tokenizer plain = tokenizer("");
    CHECK(ids(plain, "abc") == (Tokens{6}) && ids(plain, "bca") == (Tokens{5, 1}) &&
          ids(plain, "aaa") == (Tokens{11, 1}));
    CHECK(ids(plain, "axz") == (Tokens{1}));  // no unk_token and no fallback: unknown characters dropped
    const Tokenizer unk = tokenizer("", R"(, "unk_token": "<unk>", "fuse_unk": true)");
    CHECK(ids(unk, "axza") == (Tokens{1, 0, 1}));
    const Tokenizer fallback = tokenizer("", R"(, "unk_token": "<unk>", "byte_fallback": true)");
    CHECK(ids(fallback, "a\xC3\xA9z") == (Tokens{1, 7, 8, 0}));  // é as its two bytes; z has no bytes in the vocabulary
    const Tokenizer templ =
        tokenizer(R"(, "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>"}},
        {"Sequence": {"id": "A"}}], "special_tokens": {"<s>": {"id": "<s>", "ids": [10], "tokens": ["<s>"]}}})");
    CHECK(ids(templ, "ab") == (Tokens{10, 4}) && ids(templ, "ab", false) == (Tokens{4}));
    // Metaspace "first": the replacement prepended only at the start of the input, not after an added token
    const Tokenizer meta =
        tokenizer(R"(, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first",
        "split": false}, "added_tokens": [{"id": 10, "content": "<s>", "special": true}])");
    CHECK(ids(meta, "a b") == (Tokens{9, 1, 9, 2}) && ids(meta, "<s>a") == (Tokens{10, 1}));
    // added tokens: leftmost-longest, rstrip eats the space after, single_word needs word boundaries
    const Tokenizer added =
        tokenizer(R"(, "added_tokens": [{"id": 20, "content": "XY"}, {"id": 21, "content": "XYZ", "rstrip": true},
        {"id": 22, "content": "Q", "single_word": true}, {"id": 99, "content": "abc"}])");
    // ids as tokenizers assigns them: past the vocabulary (12 entries) in order, whatever the file says
    CHECK(ids(added, "aXYZ  b") == (Tokens{1, 13, 2}) && ids(added, "XYa") == (Tokens{12, 1}));
    CHECK(ids(added, "aQ Q") == (Tokens{1, 14}));  // inside a word Q is not the token (and is dropped)
    CHECK(ids(added, "cabc") == (Tokens{3, 6}));   // text in the vocabulary: the vocabulary's id
    // ignore_merges: a word in the vocabulary is that token, even when the merges wouldn't reach it
    // (Llama 3); Digits: each digit alone, or runs of them
    auto small = [](const std::string& model_extra, const std::string& pre = "null") {
        return Tokenizer::from_json(R"({"pre_tokenizer": )" + pre +
                                    R"(, "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2,
            "bc": 3, "abc": 4, "1": 5, "2": 6, "12": 7}, "merges": ["b c", "1 2"])" +
                                    model_extra + "}}");
    };
    CHECK(ids(small(""), "abc") == (Tokens{0, 3}) && ids(small(R"(, "ignore_merges": true)"), "abc") == (Tokens{4}));
    CHECK(ids(small("", R"({"type": "Digits", "individual_digits": true})"), "a12") == (Tokens{0, 5, 6}));
    CHECK(ids(small("", R"({"type": "Digits", "individual_digits": false})"), "a12") == (Tokens{0, 7}));
    // what the engine doesn't implement is refused by name
    CHECK_THROWS(Tokenizer::from_json(R"({"model": {"type": "Unigram", "vocab": []}})"));
    CHECK_THROWS(tokenizer(R"(, "normalizer": {"type": "NFKC"})"));
    CHECK_THROWS(tokenizer(R"(, "pre_tokenizer": {"type": "Whitespace"})"));
    CHECK_THROWS(Tokenizer::from_json(R"({"model": {"type": "BPE", "vocab": {"a": 0}, "merges": ["a b"]}})"));
}

void test_weight_shapes_checked() {
    for (const auto& [name, cfg] : configs()) {
        const Model m = random_model(cfg, 1);
        std::map<std::string, Tensor> w;
        for (const auto& [name, weight] : m.weights()) w[name] = weight.to_tensor();
        bool all = true;
        for (const auto& e : w) {
            auto shorter = w;
            Tensor& t = shorter[e.first];
            Shape s = t.shape;
            s[0] -= 1;
            t = Tensor(s, std::vector<float>(t.data.begin(), t.data.begin() + shape_numel(s)));
            bool threw = false;
            try {
                Model bad(cfg, shorter);
            } catch (const std::runtime_error&) { threw = true; }
            if (!threw)
                std::printf("     %s: accepted %s with shape %s\n", name.c_str(), e.first.c_str(),
                            shape_str(s).c_str());
            all &= threw;
            auto missing = w;
            missing.erase(e.first);
            threw = false;
            try {
                Model bad(cfg, missing);
            } catch (const std::runtime_error&) { threw = true; }
            if (!threw) std::printf("     %s: accepted a model without %s\n", name.c_str(), e.first.c_str());
            all &= threw;
        }
        CHECK(all);
    }
}

// d_mlp 0 asks for the default width, as indranet reads it: a config saying 0 (top level or in a layer
// override) takes weights of the default width, plain and gated MLPs alike.
void test_d_mlp_zero_is_default() {
    for (const char* mlp : {"mlp", "swiglu"}) {
        const std::string base =
            std::string(R"({"vocab_size": 50, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32, )"
                        R"("mlp_multiple_of": 8, "mlp": ")") +
            mlp + "\"";
        const ModelConfig zero = ModelConfig::from_json(base + R"(, "d_mlp": 0})");
        CHECK(!zero.d_mlp);
        const ModelConfig layer =
            ModelConfig::from_json(base + R"(, "d_mlp": 24, "layer_overrides": {"1": {"d_mlp": 0}}})");
        CHECK(layer.d_mlp && *layer.d_mlp == 24 && !layer.layer_config(1).d_mlp);
        const Model source = random_model(tiny([mlp](ModelConfig& c) { c.mlp = mlp; }), 3);  // default width
        std::map<std::string, Tensor> w;
        for (const auto& [name, weight] : source.weights()) w[name] = weight.to_tensor();
        bool loads = true;
        try {
            const Model m(zero, w);
            CHECK(same_values(m.forward(seq(6, 4)), source.forward(seq(6, 4))));
        } catch (const std::exception& e) {
            loads = false;
            std::printf("     %s, d_mlp 0: %s\n", mlp, e.what());
        }
        CHECK(loads);
    }
}

// Headers whose shapes don't fit the file are refused before anything is allocated or multiplied out:
// a shape whose element count overflows 64 bits (or any product of its leading dimensions does), one
// larger than the file, a negative dimension, a byte count that doesn't match. Zero-size tensors load.
void test_safetensors_bad_shapes() {
    const std::string path = std::format("/tmp/tfi_unit_shapes_{}.safetensors", ::getpid());
    auto write = [&](const std::string& tensor_json, size_t data_bytes) {
        const std::string header = "{" + tensor_json + "}";
        std::ofstream f(path, std::ios::binary);
        uint64_t n = header.size();
        for (int i = 0; i < 8; ++i) f.put(static_cast<char>((n >> (8 * i)) & 0xff));
        f << header << std::string(data_bytes, '\0');
    };
    auto refused = [&](const std::string& tensor_json, size_t data_bytes, const std::string& why) {
        write(tensor_json, data_bytes);
        std::string error;
        try {
            load_safetensors(path);
        } catch (const std::runtime_error& e) { error = e.what(); }
        if (error.find(why) == std::string::npos)
            std::printf("     %s: got '%s'\n", tensor_json.c_str(), error.c_str());
        CHECK(error.find(why) != std::string::npos);
    };
    refused(R"("x": {"dtype": "F32", "shape": [4294967296, 4294967296], "data_offsets": [0, 0]})", 0,
            "larger than the file");
    refused(R"("x": {"dtype": "F16", "shape": [4611686018427387904, 4], "data_offsets": [0, 16]})", 16,
            "larger than the file");
    refused(R"("x": {"dtype": "F32", "shape": [1099511627776, 1099511627776, 0], "data_offsets": [0, 0]})", 0,
            "larger than the file");
    refused(R"("x": {"dtype": "I64", "shape": [3], "data_offsets": [0, 16]})", 16, "larger than the file");
    refused(R"("x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 20]})", 24, "byte size does not match");
    refused(R"("x": {"dtype": "F32", "shape": [2, -3], "data_offsets": [0, 24]})", 24, "negative dimension");
    write(
        R"("x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}, "e": {"dtype": "F32", "shape": [0, 1099511627776, 1099511627776], "data_offsets": [24, 24]})",
        24);
    const SafeTensors ok = load_safetensors(path);
    CHECK(ok.tensors.at("x").shape == Shape({2, 3}) && ok.tensors.at("x").numel() == 6);
    CHECK(ok.tensors.at("e").numel() == 0);
    std::remove(path.c_str());
    CHECK_THROWS(shape_numel({int64_t{1} << 40, int64_t{1} << 40}));
    CHECK(shape_numel({0, int64_t{1} << 40, int64_t{1} << 40}) == 0);
    CHECK(shape_numel({}) == 1 && shape_numel({3, 4}) == 12);
}

void test_vector_math() {
    auto ulp_error = [](float got, double want) {
        const float w = static_cast<float>(std::fabs(want));
        const double ulp = w < std::numeric_limits<float>::min() ? std::numeric_limits<float>::denorm_min()
                                                                 : double(std::nextafter(w, INFINITY)) - w;
        return std::fabs(double(got) - want) / ulp;
    };
    // Every 16th float from lo up to hi. Floats in increasing order are consecutive integers: k for
    // the float k steps above zero, -k for k steps below (0 is -0.0, as nextafter from below gives),
    // so jumping 16 at a time visits them without stepping through the 15 in between.
    auto order = [](float x) {
        const auto b = std::bit_cast<uint32_t>(x);
        return (b >> 31) ? -static_cast<int64_t>(b & 0x7FFFFFFFu) : static_cast<int64_t>(b);
    };
    auto from_order = [](int64_t k) {
        const uint32_t b = k <= 0 ? 0x80000000u | static_cast<uint32_t>(-k) : static_cast<uint32_t>(k);
        return std::bit_cast<float>(b);
    };
    auto sweep = [&](float lo, float hi, auto f, auto ref, double max_ulp) {
        double worst = 0;
        for (int64_t k = order(lo), end = order(hi); k <= end; k += 16) {
            const float x = from_order(k);
            worst = std::max(worst, ulp_error(f(x), ref(double(x))));
        }
        CHECK(worst <= max_ulp);
    };
    sweep(-87.3f, 88.7f, [](float x) { return vm::exp(x); }, [](double x) { return std::exp(x); }, 1.5);
    sweep(-6.f, 6.f, [](float x) { return vm::erf(x); }, [](double x) { return std::erf(x); }, 3.0);
    sweep(-12.f, 12.f, [](float x) { return vm::tanh(x); }, [](double x) { return std::tanh(x); }, 2.0);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(vm::exp(-inf) == 0.f && vm::exp(-200.f) == 0.f && vm::exp(inf) == inf && vm::exp(100.f) == inf);
    CHECK(vm::exp(0.f) == 1.f && vm::erf(0.f) == 0.f && vm::tanh(0.f) == 0.f);
    CHECK(vm::erf(inf) == 1.f && vm::erf(-inf) == -1.f && vm::tanh(inf) == 1.f && vm::tanh(-inf) == -1.f);
    CHECK(std::isnan(vm::exp(nan)) && std::isnan(vm::erf(nan)) && std::isnan(vm::tanh(nan)));
    // Softmax: masked (-inf) entries are exactly 0, the rest sum to 1.
    float row[11] = {0.5f, -inf, 3.f, -1.f, -inf, 2.f, 0.f, 1.f, -2.f, 4.f, -inf};
    softmax_(row);
    CHECK(row[1] == 0.f && row[4] == 0.f && row[10] == 0.f);
    double total = 0;
    for (float v : row) total += v;
    CHECK_NEAR(total, 1.0, 1e-6);
    CHECK_NEAR(row[9] / row[2], std::exp(1.0), 1e-5);
}

// generate()'s trace: each step = the full-pass analyses at that position on the text so far (the
// chosen token's probability and top tokens, logit lens, DLA, where each head attended), hooks included.
// A whole sequence's attention on the CPU runs in blocks of query rows, each against the keys up to its last row
// (from its first row's window start), unless a hook reads hook_pattern, which needs the whole pattern: both give
// the same logits, over 300 positions (blocks of 128, the last partial), with windows shorter and longer than a
// block, ALiBi, softcaps and grouped-query attention.
void test_cpu_attention_blocked() {
    auto cfg = [](std::function<void(ModelConfig&)> edit) {
        return tiny([&](ModelConfig& c) {
            c.max_seq_len = 320;
            edit(c);
        });
    };
    const std::vector<ModelConfig> cfgs = {
        cfg([](ModelConfig&) {}),
        cfg([](ModelConfig& c) {
            c.sliding_window = 50;
            c.attn_logit_softcap = 4.0;
        }),
        cfg([](ModelConfig& c) {
            c.positional = "alibi";
            c.n_kv_heads = 1;
            c.sliding_window = 200;
        }),
    };
    for (const ModelConfig& c : cfgs) {
        const Model m = random_model(c, 61);
        const Tokens t = seq(300, 5);
        int64_t calls = 0;
        Hooks whole;  // reading hook_pattern: the whole-pattern path
        whole.add_capture([](const std::string& n) { return n.ends_with("attn.hook_pattern"); },
                          [&](const std::string&, Tensor& p) { calls += p.shape[1] == 300 && p.shape[2] == 300; });
        check_close(m.forward(t), m.forward(t, &whole));
        CHECK(calls == c.n_layers);
    }
}

// A temperature whose reciprocal overflows a float: the largest logit alone, the lowest id among equal ones.
void test_tiny_temperature() {
    const Model m = random_model(tiny(), 42), other = random_model(tiny(), 43);
    const Tokens prompt = {5, 9, 13, 5, 9, 13, 5, 9};
    const std::vector<float> logits = {1.f, 3.f, 2.f, 3.f};
    GenerateOptions o;
    o.max_new_tokens = 6;
    o.stop_at_eos = false;
    const Tokens greedy = generate(m, prompt, o);
    for (double temp : {2.9e-39, 1e-40, 1e-300, std::numeric_limits<double>::denorm_min()})
        for (double top_p : {1.0, 0.9}) {
            o.temperature = temp;
            o.top_p = top_p;
            std::vector<double> p;
            Sampler(o).distribution(std::span(logits.data(), 4), p);
            CHECK(p == (std::vector<double>{0, 1, 0, 0}));
            o.speculative.draft_tokens = 0;
            CHECK(generate(m, prompt, o) == greedy);
            o.speculative.draft_tokens = 3;
            for (const Model* d : {static_cast<const Model*>(nullptr), &other}) {
                o.speculative.draft = d;
                CHECK(generate(m, prompt, o) == greedy);
            }
            o.speculative.draft = nullptr;
        }
    // The smallest scale that still fits a float samples as usual: equal logits, equal chances.
    o = GenerateOptions{};
    o.temperature = 3e-39;
    std::vector<double> p;
    Sampler(o).distribution(std::span(logits.data(), 4), p);
    CHECK(p == (std::vector<double>{0, 0.5, 0, 0.5}));
}

// The sampler alone: frequencies follow softmax(logits / temperature), restricted by top_k and top_p;
// NaN and -inf are never picked.
void test_sampler() {
    const std::vector<float> logits = {1.f, 2.f, 0.f, -INFINITY, std::nanf(""), 1.5f};
    auto freqs = [&](GenerateOptions o, int n) {
        Sampler s(o);
        std::vector<double> f(logits.size(), 0);
        for (int i = 0; i < n; ++i) f[s(logits)] += 1.0 / n;
        return f;
    };
    auto dist = [&](GenerateOptions o) {  // what speculative decoding's acceptance rule uses
        std::vector<double> d;
        Sampler(o).distribution(logits, d);
        return d;
    };
    auto softmax = [&](std::vector<int> keep, double temp) {
        std::vector<double> p(logits.size(), 0);
        double z = 0;
        for (int i : keep) z += std::exp(logits[i] / temp);
        for (int i : keep) p[i] = std::exp(logits[i] / temp) / z;
        return p;
    };
    const int n = 200000;
    GenerateOptions o;
    o.seed = 3;
    for (double temp : {1.0, 0.5}) {
        o.temperature = temp;
        const std::vector<double> f = freqs(o, n), p = softmax({0, 1, 2, 5}, temp);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
        const std::vector<double> d = dist(o);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(d[i], p[i], 1e-6);  // float weights
    }
    o.temperature = 1.0;
    o.top_k = 2;  // tokens 1 and 5
    {
        const std::vector<double> f = freqs(o, n), p = softmax({1, 5}, 1.0);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
        const std::vector<double> d = dist(o);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(d[i], p[i], 1e-6);  // float weights
    }
    o.top_k = 0;
    o.top_p = 0.6;  // token 1 has 0.47 of the mass, with token 5 0.76 >= 0.6
    {
        const std::vector<double> f = freqs(o, n), p = softmax({1, 5}, 1.0);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(f[i], p[i], 0.005);
        const std::vector<double> d = dist(o);
        for (size_t i = 0; i < f.size(); ++i) CHECK_NEAR(d[i], p[i], 1e-6);  // float weights
    }
    o.top_p = 0.3;  // token 1 alone
    CHECK_NEAR(freqs(o, 1000)[1], 1.0, 1e-9);
    o.top_p = 1;
    o.temperature = 0;
    CHECK_NEAR(freqs(o, 10)[1], 1.0, 1e-9);
    CHECK((dist(o) == std::vector<double>{0, 1, 0, 0, 0, 0}));  // greedy: one-hot
    {  // draw: index i with probability w[i] / sum(w), never a zero weight
        Sampler s(o);
        const std::vector<double> w = {0.5, 0, 1.5, 2};
        std::vector<double> f(4, 0);
        for (int i = 0; i < n; ++i) f[static_cast<size_t>(s.draw(w))] += 1.0 / n;
        for (size_t i = 0; i < 4; ++i) CHECK_NEAR(f[i], w[i] / 4, 0.005);
        CHECK_THROWS(s.draw({0, 0}));
    }
    const std::vector<float> tie = {3.f, 5.f, 5.f};
    Sampler greedy(o);
    CHECK(greedy(tie) == 1);  // the lowest id among equal logits
    const std::vector<float> none = {-INFINITY, std::nanf("")};
    CHECK_THROWS(greedy(none));

    // The sampler finds top_k and the nucleus without sorting: it picks exactly what sorting every token
    // would, on a vocabulary-sized row (peaked and flat), pick for pick from the same seed. The rule: the k
    // largest logits (lower id first among equal ones), then the fewest most likely of those whose weights
    // reach top_p of their total, then a draw over the kept tokens in id order.
    std::mt19937_64 gen(9);
    for (const float spread : {3.f, 0.3f}) {
        std::normal_distribution<float> normal(0.f, spread);
        std::vector<float> row(20000);
        for (float& v : row) v = normal(gen);
        for (int64_t i = 0; i < 20000; i += 97)
            row[static_cast<size_t>(i)] = row[static_cast<size_t>(i + 1)];  // equal logits
        row[5] = -INFINITY;
        row[6] = std::nanf("");
        const float top = *std::ranges::max_element(
            row, [](float x, float y) { return std::isnan(y) ? false : std::isnan(x) || x < y; });
        for (const double top_p : {0.5, 0.9, 0.99, 1.0})
            for (const int64_t top_k : {int64_t(0), int64_t(50), int64_t(5000)})
                for (const double temp : {0.7, 1.5}) {
                    GenerateOptions so;
                    so.temperature = temp;
                    so.top_p = top_p;
                    so.top_k = top_k;
                    so.seed = 11;
                    Sampler fast(so);
                    std::mt19937_64 rng(11);
                    std::vector<int64_t> by_logit;  // usable tokens, largest logit first
                    for (int64_t i = 0; i < 20000; ++i)
                        if (!std::isnan(row[static_cast<size_t>(i)]) && row[static_cast<size_t>(i)] != -INFINITY)
                            by_logit.push_back(i);
                    std::ranges::stable_sort(by_logit, std::ranges::greater{},
                                             [&](int64_t i) { return row[static_cast<size_t>(i)]; });
                    if (top_k && static_cast<size_t>(top_k) < by_logit.size())
                        by_logit.resize(static_cast<size_t>(top_k));
                    const float scale = static_cast<float>(1.0 / temp);
                    std::vector<std::pair<float, int64_t>> c;  // (weight, id), most likely first
                    double mass = 0;
                    for (int64_t i : by_logit) {
                        const float d = row[static_cast<size_t>(i)] - top;
                        c.push_back({detail::exp_weight(d * scale), i});
                        mass += c.back().first;
                    }
                    std::ranges::stable_sort(c, std::ranges::greater{}, [](const auto& x) { return x.first; });
                    size_t n = c.size();
                    if (top_p < 1) {
                        double sum = 0;
                        for (n = 0; n < c.size();) {
                            sum += c[n++].first;
                            if (sum >= top_p * mass) break;
                        }
                    }
                    std::vector<std::pair<int64_t, float>> kept;  // id order
                    for (size_t i = 0; i < n; ++i) kept.push_back({c[i].second, c[i].first});
                    std::ranges::sort(kept);
                    for (int pick = 0; pick < 100; ++pick) {
                        double total = 0;
                        for (const auto& [id, weight] : kept) total += weight;
                        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * total;
                        int64_t want = kept.back().first;
                        double acc = 0;
                        for (const auto& [id, weight] : kept) {
                            acc += weight;
                            if (u < acc) {
                                want = id;
                                break;
                            }
                        }
                        CHECK(fast(row) == want);
                    }
                }
    }
    // Cuts at equal keys and at a bucket's edge: which tokens distribution() keeps.
    auto kept_ids = [](const std::vector<float>& lg, int64_t k, double p) {
        GenerateOptions so;
        so.temperature = 1;
        so.top_k = k;
        so.top_p = p;
        std::vector<double> d;
        Sampler(so).distribution(lg, d);
        std::vector<int64_t> ids;
        for (size_t i = 0; i < d.size(); ++i)
            if (d[i] > 0) ids.push_back(static_cast<int64_t>(i));
        return ids;
    };
    const std::vector<float> flat(10, 1.f);  // equal logits: lower ids first
    CHECK((kept_ids(flat, 3, 1) == std::vector<int64_t>{0, 1, 2}));
    CHECK((kept_ids(flat, 0, 0.25) == std::vector<int64_t>{0, 1, 2}));  // 0.3 is the first total reaching 0.25
    CHECK((kept_ids(flat, 6, 0.5) == std::vector<int64_t>{0, 1, 2}));   // half of the top 6
    CHECK((kept_ids({-0.f, 0.f, 0.f, -1.f}, 2, 1) == std::vector<int64_t>{0, 1}));  // -0 = +0
    // Each logit in a bucket of its own: the count is reached exactly at a bucket's end.
    CHECK((kept_ids({1.f, 16.f, 2.f, 8.f, 4.f}, 2, 1) == std::vector<int64_t>{1, 3}));
    CHECK((kept_ids({1.f, 16.f, 2.f, 8.f, 4.f}, 3, 1) == std::vector<int64_t>{1, 3, 4}));
    // exp_weight = exp to fp32 precision over its range, 0 below -87, for NaN and -inf; exp_weights (vector
    // and scalar parts) gives exp_weight's bits exactly.
    float worst = 0;
    for (float x = -87.f; x <= 0.f; x += 0.0137f) {
        const double want = std::exp(static_cast<double>(x));
        worst = std::max(worst, static_cast<float>(std::fabs(detail::exp_weight(x) - want) / want));
    }
    CHECK(worst < 3e-7f);
    CHECK(detail::exp_weight(0.f) == 1.f && detail::exp_weight(-0.f) == 1.f);
    CHECK(detail::exp_weight(-87.5f) == 0.f && detail::exp_weight(-INFINITY) == 0.f &&
          detail::exp_weight(std::nanf("")) == 0.f);
    std::vector<float> xs = {0.f,       -1e-30f,       -0.5f,  -87.f, -86.99f, -87.01f,
                             -INFINITY, std::nanf(""), -3.25f, -10.f, -50.f};
    for (int i = 0; i < 1000; ++i) xs.push_back(-90.f * static_cast<float>(i) / 1000.f);
    std::vector<float> ws(xs.size());
    detail::exp_weights(xs, 0.f, 1.f, ws);
    for (size_t i = 0; i < xs.size(); ++i) {
        const float d = xs[i] - 0.f;
        CHECK(ws[i] == detail::exp_weight(d * 1.f));
    }
    // The same picks on every platform: weights and draws use only exactly specified arithmetic (the logits
    // here come from the generator's raw output, not a library distribution).
    {
        std::mt19937_64 g(21);
        std::vector<float> lg(1000);
        for (float& v : lg) v = static_cast<float>(static_cast<double>(g() >> 40) * 0x1.0p-24 * 12.0 - 6.0);
        GenerateOptions go;
        go.temperature = 0.9;
        go.top_k = 200;
        go.top_p = 0.95;
        go.seed = 4;
        Sampler gs(go);
        std::vector<int64_t> picks;
        for (int i = 0; i < 12; ++i) picks.push_back(gs(std::span(lg.data(), 1000)));
        CHECK((picks == std::vector<int64_t>{762, 411, 537, 56, 470, 52, 831, 37, 704, 751, 618, 408}));
    }
}