// Reader for safetensors files (F32, F16, BF16 converted to float32; I64 kept as integers).
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tfi/tensor.h"

namespace tfi {

struct IntTensor {
    Shape shape;
    std::vector<int64_t> data;
};

struct SafeTensors {
    std::map<std::string, Tensor> tensors;
    std::map<std::string, IntTensor> int_tensors;
    std::map<std::string, std::string> metadata;
};

SafeTensors load_safetensors(const std::string& path);
// Only the header's string metadata (no tensor is read).
std::map<std::string, std::string> read_safetensors_metadata(const std::string& path);
// The same, but each float tensor goes to `take` as soon as it is read, instead of into
// SafeTensors::tensors: a caller that converts tensors (e.g. to fp16) never holds them all as float32.
using TensorSink = std::function<void(const std::string& name, Tensor&& t)>;
// A float tensor (F32, F16 or BF16) before it is read: read(offset, n, dst) copies bytes
// [offset, offset + n) of its data, as stored (little-endian), any part, any number of times.
struct RawTensor {
    std::string dtype;
    Shape shape;
    uint64_t bytes = 0;
    std::function<void(uint64_t offset, uint64_t n, void* dst)> read;
};
// Called first for each float tensor: returning true means it read the tensor itself (e.g. straight
// into its own storage, converting as it goes), and the tensor goes nowhere else.
using RawSink = std::function<bool(const std::string& name, const RawTensor& t)>;
SafeTensors load_safetensors(const std::string& path, const TensorSink& take, const RawSink& take_raw = nullptr);

}  // namespace tfi
