// The decode runtime: generate()'s decoding, a token at a time, with speculative decoding that hooks
// don't see (GenerateOptions::speculative).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "tfi/generate.h"

namespace tfi {

// One decoding, a token at a time. Each next() returns the next new token (nothing once it has stopped), and
// the outside sees the same sequence of events whether or not it speculates:
//   the prompt's hooks, token 1, token 1's hooks, token 2, token 2's hooks, ...
// A token's hooks run on the next() after it is returned, so a token that ends the text never reaches them.
// Inside, a pass may compute guesses at the following tokens. Hooks never run on a guess: its row gets a
// prediction of what they will do, learned from their earlier calls. Once the guess is confirmed, the real
// hooks get its activations, in order, one row per call. If they do what was predicted the row stands;
// if not, the token is computed again with their real output. So hooks can keep state, pick rows, draw random
// numbers or use another token's activation, unchanged. Hooks whose edits can't be predicted run with little or
// no speculation. The model, options' draft, hooks and trace must outlive the runtime; options.on_token is
// generate()'s, not used here. After an exception from a hook or the model, next() returns nothing.
class DecodeRuntime {
  public:
    DecodeRuntime(const Model& model, const Tokens& prompt, const GenerateOptions& options = {}, const Hooks* hooks = nullptr,
                  std::vector<TraceStep>* trace = nullptr);
    ~DecodeRuntime();
    DecodeRuntime(const DecodeRuntime&) = delete;
    DecodeRuntime& operator=(const DecodeRuntime&) = delete;
    std::optional<int64_t> next();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tfi
