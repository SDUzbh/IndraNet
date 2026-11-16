// Unicode for the tokenizer: UTF-8, general categories, White_Space, and NFC / NFD normalization.
// Data from src/unicode_tables.h (tools/gen_unicode_tables.py).
#pragma once

#include <cstdint>
#include <string>

namespace tfi::detail {

using U32String = std::u32string;

// Throws std::runtime_error on invalid UTF-8 (overlong forms, surrogates and values past U+10FFFF too).
U32String utf8_decode(const std::string& s);
std::string utf8_encode(const U32String& s);
void utf8_append(std::string& out, char32_t c);

// General category of c as its two-letter name ("Lu", "Nd", ...; "Cn" when unassigned).
const char* unicode_category(char32_t c);
bool unicode_letter(char32_t c);       // \p{L}
bool unicode_number(char32_t c);       // \p{N}
bool unicode_white_space(char32_t c);  // the White_Space property (\s)
// \p{X} for a one- or two-letter category name ("L", "Lu", "N", "P", ...).
bool unicode_in_category(char32_t c, const std::string& name);
bool unicode_category_name(const std::string& name);  // whether \p{name} names a category

U32String unicode_nfd(const U32String& s);
U32String unicode_nfc(const U32String& s);
const char* unicode_version();

}  // namespace tfi::detail
