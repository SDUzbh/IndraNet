// What the model's own source files share (model.cpp, grad.cpp, gpu_lowering.cpp): hook helpers, hook point and
// parameter names, and the GPU route of run_with_gradients. Internal.
#pragma once

#include <cstdint>
#include <format>
#include <functional>
#include <string>

#include "tfi/model.h"

namespace tfi::detail {

inline void hook(const Hooks* hooks, const std::string& name, Tensor& act) {
    if (hooks) hooks->run(name, act);
}

inline std::string layer_prefix(int64_t layer) { return std::format("blocks.{}.", layer); }

// Model::run_with_gradients on the GPU (gpu_lowering.cpp), where a single pass of the tokens would go there
// (placement.h: gradients_go_to_gpu); false, nothing done, otherwise. grad.cpp asks it first.
bool gradients_on_gpu(const Model& m, const Tokens& tokens, const std::function<Tensor(const Tensor&)>& dlogits,
                      ActivationCache& grads, const NameFilter& filter, ActivationCache* acts, const Hooks* hooks,
                      Tensor& logits);

// Names of a block's hook points and parameters, by index (`using namespace detail::names` brings them in).
namespace names {

// A block's hook points, in the order of Model::hook_names_.
enum HookPoint {
    kHookResidPre,
    kHookAttnIn,
    kHookAttnOut,
    kHookMlpIn,
    kHookMlpOut,
    kHookResidMid,
    kHookResidPost,
    kHookQ,
    kHookK,
    kHookV,
    kHookPattern,
    kHookZ,
    kHookPre,
    kHookPost,
    kHookPoints
};
inline constexpr const char* kHookSuffix[kHookPoints] = {
    "hook_resid_pre",    "hook_attn_in",    "hook_attn_out", "hook_mlp_in",  "hook_mlp_out",
    "hook_resid_mid",    "hook_resid_post", "attn.hook_q",   "attn.hook_k",  "attn.hook_v",
    "attn.hook_pattern", "attn.hook_z",     "mlp.hook_pre",  "mlp.hook_post"};
// A block's parameters, in the order of Model::param_names_.
enum Param {
    kParamQ,
    kParamK,
    kParamV,
    kParamO,
    kParamUp,
    kParamGate,
    kParamDown,
    kParamLn1,
    kParamLn1Post,
    kParamLn2,
    kParamLn2Post,
    kParamQNorm,
    kParamKNorm,
    kParams
};
inline constexpr const char* kParamPrefix[kParams] = {
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.up",      "mlp.gate",   "mlp.down",
    "ln1",         "ln1_post",    "ln2",         "ln2_post",    "attn.q_norm", "attn.k_norm"};

}  // namespace names
}  // namespace tfi::detail
