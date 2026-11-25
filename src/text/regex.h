// A small backtracking regex over code points, for tokenizer split patterns (the subset Hugging Face
// tokenizers' patterns use): literals, ., classes [...] with ranges and negation, \s \S \d \D \w \W,
// \p{..} / \P{..} (general categories), \r \n \t \f \v \xHH \x{H..} \uHHHH, groups (...) (?:...),
// case-insensitive groups (?i:...) (ASCII letters), alternation, ? * + {n} {n,} {n,m} (greedy, or lazy
// with a trailing ?), and lookahead (?=...) (?!...). Semantics are Perl's / Oniguruma's: alternatives
// are tried in order, repeats greedy first, and the first match found at the leftmost start wins.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "unicode.h"

namespace tfi::detail {

class Regex {
  public:
    explicit Regex(const std::string& pattern);  // throws std::runtime_error on syntax it doesn't take
    ~Regex();
    Regex(Regex&&) noexcept;
    Regex& operator=(Regex&&) noexcept;
    // Non-overlapping matches in s, left to right, as [begin, end) code point offsets. Empty matches
    // are skipped.
    std::vector<std::pair<size_t, size_t>> find_all(const U32String& s) const;
    struct Node;

  private:
    std::unique_ptr<Node> root_;
};

}  // namespace tfi::detail
