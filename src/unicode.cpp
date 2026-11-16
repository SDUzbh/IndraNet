#include "unicode.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "unicode_tables.h"

namespace tfi::detail {

namespace ud = unicode_data;

U32String utf8_decode(const std::string& s) {
    U32String out;
    out.reserve(s.size());
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned char b = p[i];
        int len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
        if (len == 0 || i + len > n) throw std::runtime_error("text is not valid UTF-8 (byte " + std::to_string(i) + ")");
        char32_t c = len == 1 ? b : len == 2 ? (b & 0x1F) : len == 3 ? (b & 0x0F) : (b & 0x07);
        for (int k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) throw std::runtime_error("text is not valid UTF-8 (byte " + std::to_string(i + k) + ")");
            c = (c << 6) | (p[i + k] & 0x3F);
        }
        static const char32_t kMin[] = {0, 0, 0x80, 0x800, 0x10000};
        if (c < kMin[len] || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))
            throw std::runtime_error("text is not valid UTF-8 (byte " + std::to_string(i) + ")");
        out.push_back(c);
        i += len;
    }
    return out;
}

void utf8_append(std::string& out, char32_t c) {
    if (c < 0x80) {
        out.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (c >> 6)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (c >> 12)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (c >> 18)));
        out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
}

std::string utf8_encode(const U32String& s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t c : s) utf8_append(out, c);
    return out;
}

namespace {

template <size_t N>
uint8_t range_value(const ud::Range (&table)[N], char32_t c) {
    const ud::Range* end = table + N;
    const ud::Range* it = std::upper_bound(table, end, c, [](char32_t v, const ud::Range& r) { return v < r.first; });
    if (it == table) return 0;
    --it;
    return c <= it->last ? it->value : 0;
}

uint8_t combining_class(char32_t c) { return range_value(ud::kCombiningClasses, c); }

// Hangul syllables: composed and decomposed by formula (Unicode, chapter 3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void decompose(char32_t c, U32String& out) {
    if (c >= kSBase && c < kSBase + kSCount) {
        const char32_t s = c - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount) out.push_back(kTBase + s % kTCount);
        return;
    }
    const ud::Decomposition* end = std::end(ud::kDecompositions);
    const ud::Decomposition* it = std::lower_bound(std::begin(ud::kDecompositions), end, c,
                                                   [](const ud::Decomposition& d, char32_t v) { return d.cp < v; });
    if (it == end || it->cp != c) {
        out.push_back(c);
        return;
    }
    decompose(it->first, out);
    if (it->second) decompose(it->second, out);
}

char32_t compose(char32_t a, char32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount)
        return a + (b - kTBase);
    const ud::Composition* end = std::end(ud::kCompositions);
    const ud::Composition* it = std::lower_bound(std::begin(ud::kCompositions), end, std::make_pair(a, b),
                                                 [](const ud::Composition& x, const std::pair<char32_t, char32_t>& v) {
                                                     return x.first < v.first || (x.first == v.first && x.second < v.second);
                                                 });
    return it != end && it->first == a && it->second == b ? it->cp : 0;
}

}  // namespace

const char* unicode_category(char32_t c) { return ud::kCategoryNames[range_value(ud::kCategories, c)]; }

bool unicode_letter(char32_t c) { return unicode_category(c)[0] == 'L'; }
bool unicode_number(char32_t c) { return unicode_category(c)[0] == 'N'; }

bool unicode_white_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool unicode_category_name(const std::string& name) {
    if (name.size() == 1) return std::strchr("LMNPSZC", name[0]) != nullptr;
    for (const char* n : ud::kCategoryNames)
        if (name == n) return true;
    return false;
}

bool unicode_in_category(char32_t c, const std::string& name) {
    const char* cat = unicode_category(c);
    return name.size() == 1 ? cat[0] == name[0] : name == cat;
}

U32String unicode_nfd(const U32String& s) {
    U32String out;
    out.reserve(s.size());
    for (char32_t c : s) decompose(c, out);
    // canonical order: each run of non-starters sorted (stably) by combining class
    for (size_t i = 0; i < out.size();) {
        if (!combining_class(out[i])) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < out.size() && combining_class(out[j])) ++j;
        std::stable_sort(out.begin() + i, out.begin() + j, [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
        i = j;
    }
    return out;
}

U32String unicode_nfc(const U32String& s) {
    // Text below U+0300 has nothing to decompose or compose: the common case, unchanged.
    if (std::all_of(s.begin(), s.end(), [](char32_t c) { return c < 0x300; })) return s;
    U32String d = unicode_nfd(s), out;
    out.reserve(d.size());
    size_t starter = std::string::npos;  // index in out of the last starter
    int last_class = -1;                 // combining class of the last character appended after it
    for (char32_t c : d) {
        const int cc = combining_class(c);
        if (starter != std::string::npos && (last_class < cc || (last_class == -1 && cc == 0))) {
            if (const char32_t composed = compose(out[starter], c)) {
                out[starter] = composed;
                continue;
            }
        }
        if (cc == 0) {
            starter = out.size();
            last_class = -1;
        } else {
            last_class = cc;
        }
        out.push_back(c);
    }
    return out;
}

const char* unicode_version() { return ud::kVersion; }

}  // namespace tfi::detail
