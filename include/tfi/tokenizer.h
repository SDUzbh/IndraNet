// A Hugging Face tokenizer (tokenizer.json), as the model was trained with: what
// tokenizers.Tokenizer.encode(text).ids gives, special tokens included.
//
// Supported, which covers Llama 1-3, Mistral, Qwen2/3, Phi-3, Gemma, GPT-2 and Pythia:
//   model           BPE (merges, byte_fallback, ignore_merges, unk_token, fuse_unk)
//   normalizer      NFC, NFD, Replace, Prepend, Sequence
//   pre_tokenizer   ByteLevel, Split (regex or string; every behavior), Metaspace, Digits, Sequence
//   post_processor  TemplateProcessing, ByteLevel, RobertaProcessing, BertProcessing, Sequence
//   added_tokens    matched leftmost-longest, before (or, for normalized ones, after) normalization,
//                   with lstrip / rstrip / single_word
// Anything else is refused with a message naming it. Unicode data (categories for \p{..}, NFC) is
// version 13.0 (src/unicode_tables.h); text with characters assigned later may split differently.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tfi {

class Tokenizer {
public:
    // From the text of a tokenizer.json. Throws std::runtime_error for anything unsupported.
    static Tokenizer from_json(const std::string& tokenizer_json);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // Token ids for UTF-8 text. add_special_tokens: the post-processor's tokens (e.g. Llama's
    // <|begin_of_text|>), as Hugging Face does by default.
    std::vector<int64_t> encode(const std::string& text, bool add_special_tokens = true) const;

    struct Impl;

private:
    explicit Tokenizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace tfi
