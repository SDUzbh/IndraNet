#include "tfi/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "regex.h"
#include "tfi/json.h"
#include "unicode.h"

namespace tfi {

using detail::Regex;
using detail::U32String;
using detail::utf8_decode;
using detail::utf8_encode;

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("tokenizer.json: " + what + " is not supported");
}

const json::Value* field(const json::Value& v, const char* key) {
    const json::Value* f = v.find(key);
    return f && !f->is_null() ? f : nullptr;
}
bool flag(const json::Value& v, const char* key, bool fallback) {
    const json::Value* f = field(v, key);
    return f ? f->as_bool() : fallback;
}
std::string text(const json::Value& v, const char* key, const std::string& fallback = "") {
    const json::Value* f = field(v, key);
    return f ? f->as_string() : fallback;
}

// GPT-2's byte-level alphabet: the printable character each byte is written as.
const std::vector<char32_t>& byte_chars() {
    static const std::vector<char32_t> table = [] {
        std::vector<char32_t> t(256, 0);
        std::vector<bool> direct(256, false);
        for (int b = '!'; b <= '~'; ++b) direct[b] = true;
        for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
        for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
        char32_t next = 256;
        for (int b = 0; b < 256; ++b) t[b] = direct[b] ? char32_t(b) : next++;
        return t;
    }();
    return table;
}

// A split pattern: a regex, or a literal string, or a character test (Digits).
struct Pattern {
    std::optional<Regex> regex;
    U32String literal;
    bool (*test)(char32_t) = nullptr;

    std::vector<std::pair<size_t, size_t>> find_all(const U32String& s) const {
        if (regex) return regex->find_all(s);
        std::vector<std::pair<size_t, size_t>> out;
        if (test) {
            for (size_t i = 0; i < s.size(); ++i)
                if (test(s[i])) out.push_back({i, i + 1});
            return out;
        }
        if (literal.empty()) return out;
        for (size_t i = 0; i + literal.size() <= s.size();) {
            if (s.compare(i, literal.size(), literal) == 0) {
                out.push_back({i, i + literal.size()});
                i += literal.size();
            } else {
                ++i;
            }
        }
        return out;
    }
};

Pattern pattern_of(const json::Value& p) {
    Pattern out;
    if (const json::Value* r = field(p, "Regex")) out.regex.emplace(r->as_string());
    else if (const json::Value* s = field(p, "String")) out.literal = utf8_decode(s->as_string());
    else unsupported("a split pattern other than String or Regex");
    return out;
}

enum class Behavior { Removed, Isolated, MergedWithPrevious, MergedWithNext, Contiguous };

Behavior behavior_of(const std::string& b) {
    if (b == "Removed") return Behavior::Removed;
    if (b == "Isolated") return Behavior::Isolated;
    if (b == "MergedWithPrevious") return Behavior::MergedWithPrevious;
    if (b == "MergedWithNext") return Behavior::MergedWithNext;
    if (b == "Contiguous") return Behavior::Contiguous;
    unsupported("split behavior '" + b + "'");
}

// tokenizers' NormalizedString::split: the string cut at the pattern's matches, the pieces kept as
// `behavior` says. Empty pieces are dropped (as PreTokenizedString::split does).
std::vector<std::pair<size_t, size_t>> split(const U32String& s, const Pattern& pattern, Behavior behavior, bool invert) {
    std::vector<std::pair<std::pair<size_t, size_t>, bool>> parts;  // (range, is_match), covering s
    size_t prev = 0;
    for (const auto& m : pattern.find_all(s)) {
        if (m.first > prev) parts.push_back({{prev, m.first}, invert});
        parts.push_back({m, !invert});
        prev = m.second;
    }
    if (prev < s.size() || parts.empty()) parts.push_back({{prev, s.size()}, invert});
    std::vector<std::pair<size_t, size_t>> out;
    bool previous_match = false;
    switch (behavior) {
        case Behavior::Removed:
            for (const auto& p : parts)
                if (!p.second) out.push_back(p.first);
            break;
        case Behavior::Isolated:
            for (const auto& p : parts) out.push_back(p.first);
            break;
        case Behavior::MergedWithPrevious:
            for (const auto& p : parts) {
                if (p.second && !previous_match && !out.empty()) out.back().second = p.first.second;
                else out.push_back(p.first);
                previous_match = p.second;
            }
            break;
        case Behavior::MergedWithNext:
            for (size_t k = parts.size(); k-- > 0;) {
                const auto& p = parts[k];
                if (p.second && !previous_match && !out.empty()) out.back().first = p.first.first;
                else out.push_back(p.first);
                previous_match = p.second;
            }
            std::reverse(out.begin(), out.end());
            break;
        case Behavior::Contiguous:
            for (const auto& p : parts) {
                if (p.second == previous_match && !out.empty()) out.back().second = p.first.second;
                else out.push_back(p.first);
                previous_match = p.second;
            }
            break;
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const auto& r) { return r.first == r.second; }), out.end());
    return out;
}

const char* kGpt2Pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

}  // namespace

struct Tokenizer::Impl {
    struct Normalizer {
        enum Kind { NFC, NFD, Replace, Prepend } kind;
        Pattern pattern;
        U32String content;  // Replace: the replacement; Prepend: what is prepended
    };
    struct PreTokenizer {
        enum Kind { ByteLevel, Split, Metaspace } kind;
        bool add_prefix_space = false, use_regex = true;  // ByteLevel
        Pattern pattern;                                  // Split (ByteLevel with use_regex: GPT-2's)
        Behavior behavior = Behavior::Isolated;
        bool invert = false;
        char32_t replacement = U'▁';  // Metaspace
        std::string prepend_scheme = "always";
        bool split = true;
    };
    struct Added {
        U32String content;
        int64_t id;
        bool lstrip, rstrip, single_word, normalized;
    };
    struct Piece {
        U32String text;
        int64_t id;     // an added token's id, or -1 for text still to encode
        bool at_start;  // starts at the beginning of the input (Metaspace's "first" scheme)
    };

    std::vector<Normalizer> normalizers;
    std::vector<PreTokenizer> pre_tokenizers;
    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<uint64_t, std::pair<int32_t, int64_t>> merges;  // (left id, right id) -> (rank, merged id)
    bool byte_fallback = false, ignore_merges = false, fuse_unk = false;
    std::optional<int64_t> unk;
    std::vector<Added> added;
    std::vector<int64_t> prefix, suffix;  // the post-processor's tokens around the text

    void parse_normalizer(const json::Value& n) {
        const std::string type = n.at("type").as_string();
        if (type == "Sequence") {
            for (const json::Value& s : n.at("normalizers").array) parse_normalizer(s);
        } else if (type == "NFC" || type == "NFD") {
            normalizers.push_back({type == "NFC" ? Normalizer::NFC : Normalizer::NFD, {}, {}});
        } else if (type == "Replace") {
            normalizers.push_back({Normalizer::Replace, pattern_of(n.at("pattern")), utf8_decode(n.at("content").as_string())});
        } else if (type == "Prepend") {
            normalizers.push_back({Normalizer::Prepend, {}, utf8_decode(n.at("prepend").as_string())});
        } else {
            unsupported("normalizer '" + type + "'");
        }
    }

    void parse_pre_tokenizer(const json::Value& p) {
        const std::string type = p.at("type").as_string();
        if (type == "Sequence") {
            for (const json::Value& s : p.at("pretokenizers").array) parse_pre_tokenizer(s);
            return;
        }
        PreTokenizer t{};
        if (type == "ByteLevel") {
            t.kind = PreTokenizer::ByteLevel;
            t.add_prefix_space = flag(p, "add_prefix_space", false);
            t.use_regex = flag(p, "use_regex", true);
            if (t.use_regex) t.pattern.regex.emplace(kGpt2Pattern);
        } else if (type == "Split") {
            t.kind = PreTokenizer::Split;
            t.pattern = pattern_of(p.at("pattern"));
            t.behavior = behavior_of(p.at("behavior").as_string());
            t.invert = flag(p, "invert", false);
        } else if (type == "Digits") {  // char::is_numeric: each one alone, or runs of them
            t.kind = PreTokenizer::Split;
            t.pattern.test = detail::unicode_number;
            t.behavior = flag(p, "individual_digits", false) ? Behavior::Isolated : Behavior::Contiguous;
        } else if (type == "Metaspace") {
            t.kind = PreTokenizer::Metaspace;
            const U32String r = utf8_decode(text(p, "replacement", "\xE2\x96\x81"));
            if (r.size() != 1) unsupported("a Metaspace replacement that isn't one character");
            t.replacement = r[0];
            if (field(p, "prepend_scheme")) t.prepend_scheme = text(p, "prepend_scheme");
            else t.prepend_scheme = flag(p, "add_prefix_space", true) ? "always" : "never";
            if (t.prepend_scheme != "always" && t.prepend_scheme != "first" && t.prepend_scheme != "never")
                unsupported("Metaspace prepend_scheme '" + t.prepend_scheme + "'");
            t.split = flag(p, "split", true);
        } else {
            unsupported("pre_tokenizer '" + type + "'");
        }
        pre_tokenizers.push_back(std::move(t));
    }

    void parse_model(const json::Value& m) {
        const std::string type = text(m, "type", "BPE");
        if (type != "BPE") unsupported("model type '" + type + "' (only BPE is)");
        if (!text(m, "continuing_subword_prefix").empty() || !text(m, "end_of_word_suffix").empty())
            unsupported("BPE with continuing_subword_prefix / end_of_word_suffix");
        if (const json::Value* d = field(m, "dropout"); d && d->as_number() > 0) unsupported("BPE dropout");
        byte_fallback = flag(m, "byte_fallback", false);
        ignore_merges = flag(m, "ignore_merges", false);
        fuse_unk = flag(m, "fuse_unk", false);
        const json::Value& v = m.at("vocab");
        vocab.reserve(v.keys.size());
        for (size_t i = 0; i < v.keys.size(); ++i) vocab.emplace(v.keys[i], v.values[i].as_int());
        if (const json::Value* u = field(m, "unk_token")) {
            auto it = vocab.find(u->as_string());
            if (it == vocab.end()) throw std::runtime_error("tokenizer.json: unk_token '" + u->as_string() + "' is not in the vocabulary");
            unk = it->second;
        }
        const json::Value& ms = m.at("merges");
        merges.reserve(ms.array.size());
        for (size_t r = 0; r < ms.array.size(); ++r) {
            const json::Value& e = ms.array[r];
            std::string a, b;
            if (e.is_array()) {  // tokenizers >= 0.20: ["a", "b"]
                a = e.array.at(0).as_string();
                b = e.array.at(1).as_string();
            } else {  // "a b"
                const std::string& s = e.as_string();
                const size_t sp = s.find(' ');
                if (sp == std::string::npos) throw std::runtime_error("tokenizer.json: merge '" + s + "' has no space");
                a = s.substr(0, sp);
                b = s.substr(sp + 1);
            }
            auto ia = vocab.find(a), ib = vocab.find(b), im = vocab.find(a + b);
            if (ia == vocab.end() || ib == vocab.end() || im == vocab.end())
                throw std::runtime_error("tokenizer.json: merge '" + a + " " + b + "' uses tokens not in the vocabulary");
            merges.emplace((uint64_t(uint32_t(ia->second)) << 32) | uint32_t(ib->second), std::make_pair(int32_t(r), im->second));
        }
    }

    void parse_post_processor(const json::Value& p) {
        const std::string type = p.at("type").as_string();
        if (type == "Sequence") {
            for (const json::Value& s : p.at("processors").array) parse_post_processor(s);
        } else if (type == "ByteLevel") {
            // offsets only
        } else if (type == "TemplateProcessing") {
            const json::Value& specials = p.at("special_tokens");
            bool seen_text = false;
            for (const json::Value& item : p.at("single").array) {
                if (const json::Value* s = field(item, "SpecialToken")) {
                    const json::Value& entry = specials.at(s->at("id").as_string());
                    for (const json::Value& id : entry.at("ids").array) (seen_text ? suffix : prefix).push_back(id.as_int());
                } else if (field(item, "Sequence")) {
                    if (seen_text) unsupported("a TemplateProcessing template with the text twice");
                    seen_text = true;
                } else {
                    unsupported("a TemplateProcessing item other than SpecialToken or Sequence");
                }
            }
        } else if (type == "RobertaProcessing" || type == "BertProcessing") {
            prefix.push_back(p.at("cls").array.at(1).as_int());
            suffix.push_back(p.at("sep").array.at(1).as_int());
        } else {
            unsupported("post_processor '" + type + "'");
        }
    }

    U32String normalize(U32String s) const {
        for (const Normalizer& n : normalizers) {
            switch (n.kind) {
                case Normalizer::NFC: s = detail::unicode_nfc(s); break;
                case Normalizer::NFD: s = detail::unicode_nfd(s); break;
                case Normalizer::Prepend:
                    if (!s.empty()) s = n.content + s;
                    break;
                case Normalizer::Replace: {
                    U32String out;
                    size_t prev = 0;
                    for (const auto& m : n.pattern.find_all(s)) {
                        out.append(s, prev, m.first - prev);
                        out += n.content;
                        prev = m.second;
                    }
                    out.append(s, prev, std::string::npos);
                    s = std::move(out);
                    break;
                }
            }
        }
        return s;
    }

    // Cut the added tokens with the given `normalized` flag out of pieces still to encode.
    std::vector<Piece> split_added(std::vector<Piece> pieces, bool normalized) const {
        std::vector<const Added*> tokens;
        for (const Added& a : added)
            if (a.normalized == normalized) tokens.push_back(&a);
        if (tokens.empty()) return pieces;
        std::vector<Piece> out;
        for (Piece& p : pieces) {
            if (p.id >= 0) {
                out.push_back(std::move(p));
                continue;
            }
            const U32String& s = p.text;
            size_t start = 0;  // of text not yet emitted
            for (size_t i = 0; i < s.size();) {
                const Added* best = nullptr;
                for (const Added* a : tokens) {  // leftmost, then longest
                    if (a->content.empty() || (best && a->content.size() <= best->content.size())) continue;
                    if (s.compare(i, a->content.size(), a->content) != 0) continue;
                    if (a->single_word) {
                        auto word = [](char32_t c) { return detail::unicode_letter(c) || detail::unicode_number(c) || c == '_'; };
                        const size_t e = i + a->content.size();
                        if ((i > 0 && word(s[i - 1]) && word(a->content.front())) ||
                            (e < s.size() && word(s[e]) && word(a->content.back())))
                            continue;
                    }
                    best = a;
                }
                if (!best) {
                    ++i;
                    continue;
                }
                size_t b = i, e = i + best->content.size();
                if (best->lstrip)
                    while (b > start && detail::unicode_white_space(s[b - 1])) --b;
                if (best->rstrip)
                    while (e < s.size() && detail::unicode_white_space(s[e])) ++e;
                if (b > start) out.push_back({s.substr(start, b - start), -1, p.at_start && start == 0});
                out.push_back({{}, best->id, false});
                start = i = e;
            }
            if (start < s.size()) out.push_back({s.substr(start), -1, p.at_start && start == 0});
        }
        return out;
    }

    // The pre-tokenizers, in order, each applied to every piece the previous ones left.
    std::vector<U32String> pre_tokenize(const U32String& s, bool at_start) const {
        std::vector<std::pair<U32String, bool>> pieces{{s, at_start}};
        for (const PreTokenizer& t : pre_tokenizers) {
            std::vector<std::pair<U32String, bool>> next;
            for (auto& [piece, first] : pieces) {
                U32String w = piece;
                const Pattern* pattern = nullptr;
                Behavior behavior = Behavior::Isolated;
                bool invert = false;
                if (t.kind == PreTokenizer::ByteLevel) {
                    if (t.add_prefix_space && (w.empty() || w[0] != ' ')) w.insert(w.begin(), U' ');
                    if (t.use_regex) pattern = &t.pattern;
                } else if (t.kind == PreTokenizer::Split) {
                    pattern = &t.pattern;
                    behavior = t.behavior;
                    invert = t.invert;
                } else {  // Metaspace
                    for (char32_t& c : w)
                        if (c == ' ') c = t.replacement;
                    const bool prepend = t.prepend_scheme == "always" || (t.prepend_scheme == "first" && first);
                    if (prepend && (w.empty() || w[0] != t.replacement)) w.insert(w.begin(), t.replacement);
                }
                std::vector<std::pair<size_t, size_t>> ranges{{0, w.size()}};
                if (pattern) {
                    ranges = split(w, *pattern, behavior, invert);
                } else if (t.kind == PreTokenizer::Metaspace && t.split) {
                    Pattern rep;
                    rep.literal = U32String(1, t.replacement);
                    ranges = split(w, rep, Behavior::MergedWithNext, false);
                }
                for (size_t k = 0; k < ranges.size(); ++k) {
                    U32String sub = w.substr(ranges[k].first, ranges[k].second - ranges[k].first);
                    if (sub.empty()) continue;
                    if (t.kind == PreTokenizer::ByteLevel) {  // every byte as its printable character
                        const std::string bytes = utf8_encode(sub);
                        sub.clear();
                        for (unsigned char b : bytes) sub.push_back(byte_chars()[b]);
                    }
                    next.push_back({std::move(sub), first && ranges[k].first == 0});
                }
            }
            pieces = std::move(next);
        }
        std::vector<U32String> out;
        for (auto& p : pieces) out.push_back(std::move(p.first));
        return out;
    }

    void bpe(const U32String& word, std::vector<int64_t>& out) const {
        const std::string whole = utf8_encode(word);
        if (ignore_merges) {
            auto it = vocab.find(whole);
            if (it != vocab.end()) {
                out.push_back(it->second);
                return;
            }
        }
        std::vector<int64_t> ids;
        bool last_unk = false;
        for (char32_t c : word) {
            std::string ch;
            detail::utf8_append(ch, c);
            auto it = vocab.find(ch);
            if (it != vocab.end()) {
                ids.push_back(it->second);
                last_unk = false;
                continue;
            }
            if (byte_fallback) {  // the character's UTF-8 bytes as <0xNN> tokens, if all exist
                std::vector<int64_t> bytes;
                for (unsigned char b : ch) {
                    char name[8];
                    std::snprintf(name, sizeof name, "<0x%02X>", b);
                    auto bt = vocab.find(name);
                    if (bt == vocab.end()) break;
                    bytes.push_back(bt->second);
                }
                if (bytes.size() == ch.size()) {
                    ids.insert(ids.end(), bytes.begin(), bytes.end());
                    last_unk = false;
                    continue;
                }
            }
            if (unk) {  // otherwise the character is dropped, as tokenizers does
                if (!(fuse_unk && last_unk)) ids.push_back(*unk);
                last_unk = true;
            }
        }
        // Merge the adjacent pair with the lowest rank, leftmost first, until none is left.
        while (ids.size() > 1) {
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            size_t best = 0;
            int64_t merged = 0;
            for (size_t j = 0; j + 1 < ids.size(); ++j) {
                auto it = merges.find((uint64_t(uint32_t(ids[j])) << 32) | uint32_t(ids[j + 1]));
                if (it != merges.end() && it->second.first < best_rank) {
                    best_rank = it->second.first;
                    best = j;
                    merged = it->second.second;
                }
            }
            if (best_rank == std::numeric_limits<int32_t>::max()) break;
            ids[best] = merged;
            ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best) + 1);
        }
        out.insert(out.end(), ids.begin(), ids.end());
    }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::from_json(const std::string& tokenizer_json) {
    const json::Value t = json::parse(tokenizer_json);
    auto impl = std::make_unique<Impl>();
    if (const json::Value* n = field(t, "normalizer")) impl->parse_normalizer(*n);
    if (const json::Value* p = field(t, "pre_tokenizer")) impl->parse_pre_tokenizer(*p);
    impl->parse_model(t.at("model"));
    if (const json::Value* p = field(t, "post_processor")) impl->parse_post_processor(*p);
    if (const json::Value* a = field(t, "added_tokens")) {
        // Ids as tokenizers assigns them when it loads the file (AddedVocabulary::add_tokens): an earlier
        // added token's, else the vocabulary's, else the next free one. The ids written in the file are
        // the same for any file tokenizers saved; this follows tokenizers where one isn't.
        const int64_t vocab_size = static_cast<int64_t>(impl->vocab.size());
        std::unordered_map<std::string, int64_t> seen;
        int64_t max_id = -1;
        for (const json::Value& e : a->array) {
            const std::string& content = e.at("content").as_string();
            int64_t id;
            if (auto s = seen.find(content); s != seen.end()) id = s->second;
            else if (auto v = impl->vocab.find(content); v != impl->vocab.end()) id = v->second;
            else id = max_id < 0 ? vocab_size : (max_id >= vocab_size || vocab_size == 0) ? max_id + 1 : vocab_size;
            seen.emplace(content, id);
            max_id = std::max(max_id, id);
            impl->added.push_back({utf8_decode(content), id, flag(e, "lstrip", false), flag(e, "rstrip", false),
                                   flag(e, "single_word", false), flag(e, "normalized", false)});
        }
    }
    return Tokenizer(std::move(impl));
}

std::vector<int64_t> Tokenizer::encode(const std::string& text_utf8, bool add_special_tokens) const {
    const Impl& m = *impl_;
    std::vector<Impl::Piece> pieces = m.split_added({{utf8_decode(text_utf8), -1, true}}, false);
    std::vector<Impl::Piece> normalized;
    for (Impl::Piece& p : pieces) {
        if (p.id < 0) p.text = m.normalize(std::move(p.text));
        if (p.id >= 0 || !p.text.empty()) normalized.push_back(std::move(p));
    }
    pieces = m.split_added(std::move(normalized), true);
    std::vector<int64_t> ids;
    if (add_special_tokens) ids = m.prefix;
    for (const Impl::Piece& p : pieces) {
        if (p.id >= 0) {
            ids.push_back(p.id);
            continue;
        }
        for (const U32String& w : m.pre_tokenize(p.text, p.at_start)) m.bpe(w, ids);
    }
    if (add_special_tokens) ids.insert(ids.end(), m.suffix.begin(), m.suffix.end());
    return ids;
}

}  // namespace tfi
