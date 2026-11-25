#include "regex.h"

#include <functional>
#include <limits>
#include <stdexcept>

namespace tfi::detail {

struct Regex::Node {
    enum Kind { Char, Class, Any, Seq, Alt, Repeat, Look } kind;
    // Char
    char32_t c = 0;
    bool icase = false;
    // Class: ranges and named sets, optionally negated
    struct Item {
        enum Type { Range, Category, Space, Digit, Word } type;
        char32_t lo = 0, hi = 0;
        std::string category;
        bool negated = false;  // \P{..}, \S, \D, \W
    };
    std::vector<Item> items;
    bool negated = false;
    // Seq / Alt children; Repeat / Look: children[0]
    std::vector<std::unique_ptr<Node>> children;
    size_t min = 0, max = 0;
    bool lazy = false;
    bool look_negative = false;

    explicit Node(Kind k) : kind(k) {}
};

namespace {

using Node = Regex::Node;
constexpr size_t kInf = std::numeric_limits<size_t>::max();

char32_t fold(char32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
char32_t upper(char32_t c) { return c >= 'a' && c <= 'z' ? c - 32 : c; }

bool item_matches(const Node::Item& it, char32_t c, bool icase) {
    bool m = false;
    switch (it.type) {
        case Node::Item::Range: {
            auto in = [&](char32_t x) { return x >= it.lo && x <= it.hi; };
            m = in(c) || (icase && (in(fold(c)) || in(upper(c))));  // (?i:...): ASCII letters in either case
            break;
        }
        case Node::Item::Category: m = unicode_in_category(c, it.category); break;
        case Node::Item::Space: m = unicode_white_space(c); break;
        case Node::Item::Digit: m = unicode_in_category(c, "Nd"); break;
        case Node::Item::Word:
            m = unicode_letter(c) || unicode_in_category(c, "M") || unicode_in_category(c, "Nd") ||
                unicode_in_category(c, "Pc");
            break;
    }
    return m != it.negated;
}

// Whether node (a single-character matcher) matches c.
bool single(const Node& n, char32_t c) {
    switch (n.kind) {
        case Node::Char: return n.icase ? fold(c) == fold(n.c) : c == n.c;
        case Node::Any: return c != '\n';
        case Node::Class: {
            bool m = false;
            for (const auto& it : n.items)
                if (item_matches(it, c, n.icase)) {
                    m = true;
                    break;
                }
            return m != n.negated;
        }
        default: return false;
    }
}

bool is_single(const Node& n) { return n.kind == Node::Char || n.kind == Node::Any || n.kind == Node::Class; }

class Parser {
  public:
    explicit Parser(const std::string& pattern) : p_(utf8_decode(pattern)), src_(pattern) {}

    std::unique_ptr<Node> parse() {
        auto n = alternation(false);
        if (i_ != p_.size()) fail("unexpected ')'");
        return n;
    }

  private:
    U32String p_;
    std::string src_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("tokenizer regex: " + why + " at offset " + std::to_string(i_) + " in " + src_);
    }
    bool more() const { return i_ < p_.size(); }
    char32_t peek() const { return p_[i_]; }
    bool eat(char32_t c) {
        if (more() && p_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }

    std::unique_ptr<Node> alternation(bool icase) {
        auto alt = std::make_unique<Node>(Node::Alt);
        alt->children.push_back(sequence(icase));
        while (eat('|')) alt->children.push_back(sequence(icase));
        if (alt->children.size() == 1) return std::move(alt->children[0]);
        return alt;
    }

    std::unique_ptr<Node> sequence(bool icase) {
        auto seq = std::make_unique<Node>(Node::Seq);
        while (more() && peek() != '|' && peek() != ')') seq->children.push_back(quantified(icase));
        return seq;
    }

    size_t number() {
        size_t v = 0;
        bool any = false;
        while (more() && peek() >= '0' && peek() <= '9') {
            v = v * 10 + (p_[i_++] - '0');
            any = true;
        }
        if (!any) fail("expected a number");
        return v;
    }

    std::unique_ptr<Node> quantified(bool icase) {
        auto atom_node = atom(icase);
        while (more()) {
            size_t lo, hi;
            if (eat('?')) lo = 0, hi = 1;
            else if (eat('*')) lo = 0, hi = kInf;
            else if (eat('+')) lo = 1, hi = kInf;
            else if (peek() == '{' && i_ + 1 < p_.size() && p_[i_ + 1] >= '0' && p_[i_ + 1] <= '9') {
                ++i_;
                lo = hi = number();
                if (eat(',')) hi = (more() && peek() == '}') ? kInf : number();
                if (!eat('}')) fail("expected '}'");
                if (hi < lo) fail("repeat {n,m} with m < n");
            } else {
                break;
            }
            if (atom_node->kind == Node::Look) fail("a lookahead can't repeat");
            auto rep = std::make_unique<Node>(Node::Repeat);
            rep->min = lo;
            rep->max = hi;
            rep->lazy = eat('?');
            if (more() && peek() == '+') fail("possessive repeats aren't supported");
            rep->children.push_back(std::move(atom_node));
            atom_node = std::move(rep);
        }
        return atom_node;
    }

    char32_t hex_escape(size_t digits) {
        char32_t v = 0;
        for (size_t k = 0; k < digits; ++k) {
            if (!more()) fail("short hex escape");
            const char32_t c = p_[i_++];
            const int d = c >= '0' && c <= '9'   ? int(c - '0')
                          : c >= 'a' && c <= 'f' ? int(c - 'a' + 10)
                          : c >= 'A' && c <= 'F' ? int(c - 'A' + 10)
                                                 : -1;
            if (d < 0) fail("bad hex digit");
            v = v * 16 + char32_t(d);
        }
        return v;
    }

    // After a backslash: a class item (\s \d \w \p{..} and negations) or a literal character.
    bool escape(Node::Item& item, char32_t& literal) {
        if (!more()) fail("trailing backslash");
        const char32_t c = p_[i_++];
        switch (c) {
            case 's':
            case 'S': item = {Node::Item::Space, 0, 0, "", c == 'S'}; return true;
            case 'd':
            case 'D': item = {Node::Item::Digit, 0, 0, "", c == 'D'}; return true;
            case 'w':
            case 'W': item = {Node::Item::Word, 0, 0, "", c == 'W'}; return true;
            case 'p':
            case 'P': {
                std::string name;
                if (eat('{')) {
                    while (more() && peek() != '}') name.push_back(static_cast<char>(p_[i_++]));
                    if (!eat('}')) fail("expected '}' after \\p{");
                } else if (more()) {
                    name.push_back(static_cast<char>(p_[i_++]));
                }
                bool negated = c == 'P';
                if (!name.empty() && name[0] == '^') {
                    negated = !negated;
                    name.erase(0, 1);
                }
                if (!unicode_category_name(name)) fail("unknown property \\p{" + name + "}");
                item = {Node::Item::Category, 0, 0, name, negated};
                return true;
            }
            case 'r': literal = '\r'; return false;
            case 'n': literal = '\n'; return false;
            case 't': literal = '\t'; return false;
            case 'f': literal = '\f'; return false;
            case 'v': literal = '\v'; return false;
            case 'x':
                if (eat('{')) {
                    char32_t v = 0;
                    while (more() && peek() != '}') v = v * 16 + hex_escape(1);
                    if (!eat('}')) fail("expected '}'");
                    literal = v;
                } else {
                    literal = hex_escape(2);
                }
                return false;
            case 'u': literal = hex_escape(4); return false;
            default:
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
                    fail(std::string("unsupported escape \\") + static_cast<char>(c));
                literal = c;  // an escaped symbol stands for itself
                return false;
        }
    }

    std::unique_ptr<Node> atom(bool icase) {
        const char32_t c = p_[i_++];
        if (c == '(') {
            std::unique_ptr<Node> inner;
            if (eat('?')) {
                if (eat(':')) {
                    inner = alternation(icase);
                } else if (eat('i')) {
                    if (!eat(':')) fail("only (?i:...) groups are supported");
                    inner = alternation(true);
                } else if (peek() == '=' || peek() == '!') {
                    auto look = std::make_unique<Node>(Node::Look);
                    look->look_negative = p_[i_++] == '!';
                    look->children.push_back(alternation(icase));
                    inner = std::move(look);
                } else {
                    fail("unsupported group (?" + std::string(1, static_cast<char>(peek())) + "...)");
                }
            } else {
                inner = alternation(icase);
            }
            if (!eat(')')) fail("expected ')'");
            return inner;
        }
        if (c == '[') return char_class(icase);
        if (c == '.') return std::make_unique<Node>(Node::Any);
        if (c == '^' || c == '$') fail("anchors aren't supported");
        if (c == '*' || c == '+' || c == '?' || c == '{' || c == ')') fail("nothing to repeat");
        auto n = std::make_unique<Node>(Node::Char);
        n->icase = icase;
        if (c == '\\') {
            Node::Item item{};
            char32_t lit = 0;
            if (escape(item, lit)) {
                n->kind = Node::Class;
                n->items.push_back(item);
                return n;
            }
            n->c = lit;
        } else {
            n->c = c;
        }
        return n;
    }

    std::unique_ptr<Node> char_class(bool icase) {
        auto n = std::make_unique<Node>(Node::Class);
        n->icase = icase;
        n->negated = eat('^');
        bool first = true;
        while (more() && (peek() != ']' || first)) {
            first = false;
            char32_t lo;
            if (eat('\\')) {
                Node::Item item{};
                if (escape(item, lo)) {
                    n->items.push_back(item);
                    continue;
                }
            } else {
                lo = p_[i_++];
                if (lo == '[') fail("nested classes aren't supported");
            }
            char32_t hi = lo;
            if (more() && peek() == '-' && i_ + 1 < p_.size() && p_[i_ + 1] != ']') {
                ++i_;
                if (eat('\\')) {
                    Node::Item item{};
                    if (escape(item, hi)) fail("a range can't end in a class");
                } else {
                    hi = p_[i_++];
                }
                if (hi < lo) fail("range out of order");
            }
            n->items.push_back({Node::Item::Range, lo, hi, "", false});
        }
        if (!eat(']')) fail("expected ']'");
        return n;
    }
};

using Cont = std::function<bool(size_t)>;

class Matcher {
  public:
    explicit Matcher(const U32String& s) : s_(s) {}

    bool match(const Node& n, size_t i, const Cont& k) const {
        switch (n.kind) {
            case Node::Char:
            case Node::Any:
            case Node::Class: return i < s_.size() && single(n, s_[i]) && k(i + 1);
            case Node::Seq: return seq(n, 0, i, k);
            case Node::Alt:
                for (const auto& c : n.children)
                    if (match(*c, i, k)) return true;
                return false;
            case Node::Look: {
                const bool found = match(*n.children[0], i, [](size_t) { return true; });
                return found != n.look_negative && k(i);
            }
            case Node::Repeat: {
                const Node& child = *n.children[0];
                if (is_single(child)) {  // a loop, not recursion: long runs can't overflow the stack
                    size_t run = 0;
                    while (run < n.max && i + run < s_.size() && single(child, s_[i + run])) ++run;
                    if (run < n.min) return false;
                    if (n.lazy) {
                        for (size_t r = n.min; r <= run; ++r)
                            if (k(i + r)) return true;
                    } else {
                        for (size_t r = run + 1; r-- > n.min;)
                            if (k(i + r)) return true;
                    }
                    return false;
                }
                return repeat(n, 0, i, k);
            }
        }
        return false;
    }

  private:
    const U32String& s_;

    bool seq(const Node& n, size_t idx, size_t i, const Cont& k) const {
        if (idx == n.children.size()) return k(i);
        return match(*n.children[idx], i, [&](size_t j) { return seq(n, idx + 1, j, k); });
    }

    bool repeat(const Node& n, size_t count, size_t i, const Cont& k) const {
        auto again = [&]() {
            return count < n.max && match(
                                        *n.children[0], i,
                                        [&](size_t j) {
                                            return j != i &&
                                                   repeat(n, count + 1, j, k);  // an empty iteration ends the repeat
                                        });
        };
        if (n.lazy) return (count >= n.min && k(i)) || again();
        return again() || (count >= n.min && k(i));
    }
};

}  // namespace

Regex::Regex(const std::string& pattern) : root_(Parser(pattern).parse()) {}
Regex::~Regex() = default;
Regex::Regex(Regex&&) noexcept = default;
Regex& Regex::operator=(Regex&&) noexcept = default;

std::vector<std::pair<size_t, size_t>> Regex::find_all(const U32String& s) const {
    std::vector<std::pair<size_t, size_t>> out;
    const Matcher m(s);
    for (size_t pos = 0; pos < s.size();) {
        size_t end = 0;
        if (m.match(*root_, pos,
                    [&](size_t j) {
                        end = j;
                        return true;
                    }) &&
            end > pos) {
            out.push_back({pos, end});
            pos = end;
        } else {
            ++pos;
        }
    }
    return out;
}

}  // namespace tfi::detail
