// The sampler's arithmetic (generate.cpp), exposed for tests. Internal.
#pragma once

#include <span>

namespace tfi::detail {

// The sampler's weights (Sampler, generate.cpp): w[i] = exp((logits[i] - top) * scale) in float, for logits
// at most `top`; 0 where that exponent is below -87 (exp under fp32's smallest normal), and for NaN and
// -inf. The same bits on every platform: exp_weight's operations, four at a time on ARM.
float exp_weight(float x);  // exp(x) for x <= 0, 0 below -87 and for NaN
void exp_weights(std::span<const float> logits, float top, float scale, std::span<float> w);

}  // namespace tfi::detail
