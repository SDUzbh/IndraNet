#include "tfi/safetensors.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "tfi/json.h"
#include "half.h"

namespace tfi {

namespace {

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

SafeTensors load_safetensors(const std::string& path) { return load_safetensors(path, nullptr); }

SafeTensors load_safetensors(const std::string& path, const TensorSink& take, const RawSink& take_raw) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t header_len = 0;
    unsigned char len_bytes[8];
    if (!f.read(reinterpret_cast<char*>(len_bytes), 8)) throw std::runtime_error(path + ": file too short");
    for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
    if (header_len > (uint64_t(1) << 30)) throw std::runtime_error(path + ": implausible header size");
    std::string header(header_len, '\0');
    if (!f.read(&header[0], static_cast<std::streamsize>(header_len))) throw std::runtime_error(path + ": truncated header");
    const json::Value root = json::parse(header);
    const std::streamoff data_start = 8 + static_cast<std::streamoff>(header_len);

    f.seekg(0, std::ios::end);
    const uint64_t data_size = static_cast<uint64_t>(f.tellg()) - static_cast<uint64_t>(data_start);

    SafeTensors out;
    for (size_t k = 0; k < root.keys.size(); ++k) {
        const std::string& name = root.keys[k];
        const json::Value& info = root.values[k];
        if (name == "__metadata__") {
            for (size_t m = 0; m < info.keys.size(); ++m) out.metadata[info.keys[m]] = info.values[m].as_string();
            continue;
        }
        const std::string& dtype = info.at("dtype").as_string();
        Shape shape;
        for (const auto& d : info.at("shape").array) shape.push_back(d.as_int());
        const uint64_t begin = static_cast<uint64_t>(info.at("data_offsets").array.at(0).as_int());
        const uint64_t end = static_cast<uint64_t>(info.at("data_offsets").array.at(1).as_int());
        if (end < begin || end > data_size) throw std::runtime_error(path + ": bad offsets for " + name);
        for (int64_t dim : shape)
            if (dim < 0) throw std::runtime_error(path + ": negative dimension for " + name);
        const int64_t n = shape_numel(shape);
        const uint64_t width = dtype == "I64" ? 8 : dtype == "F32" ? 4 : (dtype == "F16" || dtype == "BF16") ? 2 : 0;
        if (width == 0) throw std::runtime_error(path + ": unsupported dtype " + dtype + " for " + name);
        if (static_cast<uint64_t>(n) > data_size / width || end - begin != static_cast<uint64_t>(n) * width)
            throw std::runtime_error(path + ": " + name + " byte size does not match its shape");
        // Straight into the tensor's own storage: no intermediate byte buffer.
        f.seekg(data_start + static_cast<std::streamoff>(begin));
        auto read_into = [&](void* dst) {
            if (n && !f.read(static_cast<char*>(dst), static_cast<std::streamsize>(end - begin)))
                throw std::runtime_error(path + ": cannot read " + name);
        };
        if (dtype == "I64") {
            IntTensor t{shape, std::vector<int64_t>(static_cast<size_t>(n))};
            read_into(t.data.data());
            out.int_tensors[name] = std::move(t);
            continue;
        }
        if (take_raw) {
            const uint64_t size = end - begin;
            const RawTensor raw{dtype, shape, size, [&](uint64_t offset, uint64_t count, void* dst) {
                                    if (offset > size || count > size - offset)
                                        throw std::out_of_range(path + ": read past the end of " + name);
                                    f.seekg(data_start + static_cast<std::streamoff>(begin + offset));
                                    if (count && !f.read(static_cast<char*>(dst), static_cast<std::streamsize>(count)))
                                        throw std::runtime_error(path + ": cannot read " + name);
                                }};
            if (take_raw(name, raw)) continue;
            f.seekg(data_start + static_cast<std::streamoff>(begin));
        }
        std::vector<float> values;
        if (dtype == "F32") {
            values.resize(static_cast<size_t>(n));
            read_into(values.data());
        } else {
            std::vector<uint16_t> bits(static_cast<size_t>(n));
            read_into(bits.data());
            values.resize(static_cast<size_t>(n));
            if (dtype == "F16") detail::halves_to_floats(bits.data(), values.data(), n);
            else
                for (int64_t i = 0; i < n; ++i) values[i] = bf16_to_f32(bits[i]);
        }
        Tensor t(shape, std::move(values));
        if (take) take(name, std::move(t));
        else out.tensors[name] = std::move(t);
    }
    return out;
}

}  // namespace tfi
