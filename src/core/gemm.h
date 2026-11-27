// The float32 matrix multiply the CPU's passes are built on (Accelerate's BLAS on macOS). Internal.
#pragma once

#include <cstdint>

namespace tfi::detail {

// C(m,n) = A(m,k) @ op(B) with row strides lda/ldb/ldc; op(B) = B (k,n) or, with trans_b, B(n,k)^T.
// With accumulate, C += A @ op(B).
void gemm(bool trans_b, int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb,
          float* C, int64_t ldc, bool accumulate = false);

}  // namespace tfi::detail
