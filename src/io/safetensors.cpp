#include "tfi/safetensors.h"

#include <bit>
#include <fcntl.h>
#include <format>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "../model/half.h"
#include "tfi/json.h"

namespace tfi {

namespace {

float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(uint32_t(h) << 16); }

}  // namespace

SafeTensors load_safetensors(const std::string& path) { return load_safetensors(path, nullptr); }

namespace {

// The JSON header; f is left at the first data byte.
std::string read_header(std::ifstream& f, const std::string& path) {
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t header_len = 0;
    unsigned char len_bytes[8];
    if (!f.read(reinterpret_cast<char*>(len_bytes), 8)) throw std::runtime_error(path + ": file too short");
    for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
    if (header_len > (uint64_t(1) << 30)) throw std::runtime_error(path + ": implausible header size");
    std::string header(header_len, '\0');
    if (!f.read(&header[0], static_cast<std::streamsize>(header_len)))
        throw std::runtime_error(path + ": truncated header");
    return header;
}

// The whole file, read-only and shared with the page cache; nullptr when it can't be mapped (an
// empty or special file, or mmap failing), and the reader falls back to reading.
std::shared_ptr<const void> map_file(const std::string& path, uint64_t& size) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st{};
    void* base = MAP_FAILED;
    if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
        size = static_cast<uint64_t>(st.st_size);
        base = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_SHARED, fd, 0);
    }
    ::close(fd);  // the mapping stays valid without the descriptor
    if (base == MAP_FAILED) return nullptr;
    const size_t length = static_cast<size_t>(size);
    return std::shared_ptr<const void>(base, [length](const void* p) { ::munmap(const_cast<void*>(p), length); });
}

}  // namespace

std::map<std::string, std::string> read_safetensors_metadata(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    const json::Value root = json::parse(read_header(f, path));
    std::map<std::string, std::string> out;
    if (const json::Value* meta = root.find("__metadata__"))
        for (size_t m = 0; m < meta->keys.size(); ++m) out[meta->keys[m]] = meta->values[m].as_string();
    return out;
}

SafeTensors load_safetensors(const std::string& path, const TensorSink& take, const RawSink& take_raw) {
    std::ifstream f(path, std::ios::binary);
    const std::string header = read_header(f, path);
    const uint64_t header_len = header.size();
    const json::Value root = json::parse(header);
    const std::streamoff data_start = 8 + static_cast<std::streamoff>(header_len);

    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    const uint64_t data_size = file_size - static_cast<uint64_t>(data_start);
    uint64_t mapped_size = 0;
    const std::shared_ptr<const void> mapping = map_file(path, mapped_size);
    const char* data =
        mapping && mapped_size == file_size ? static_cast<const char*>(mapping.get()) + data_start : nullptr;
    // Bytes [at, at + n) of the data section: from the mapping, or read from the file.
    auto copy = [&](uint64_t at, uint64_t n, void* dst, const std::string& name) {
        if (n == 0) return;
        if (data) {
            std::memcpy(dst, data + at, static_cast<size_t>(n));
            return;
        }
        f.clear();
        f.seekg(data_start + static_cast<std::streamoff>(at));
        if (!f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n)))
            throw std::runtime_error(std::format("{}: cannot read {}", path, name));
    };

    SafeTensors out;
    for (size_t k = 0; k < root.keys.size(); ++k) {
        const std::string& name = root.keys[k];
        const json::Value& info = root.values[k];
        if (name == "__metadata__") {
            for (size_t m = 0; m < info.keys.size(); ++m) out.metadata[info.keys[m]] = info.values[m].as_string();
            continue;
        }
        if (name.starts_with("__padding")) continue;  // alignment filler (see the header)
        const std::string& dtype = info.at("dtype").as_string();
        Shape shape;
        for (const auto& d : info.at("shape").array) shape.push_back(d.as_int());
        const uint64_t begin = static_cast<uint64_t>(info.at("data_offsets").array.at(0).as_int());
        const uint64_t end = static_cast<uint64_t>(info.at("data_offsets").array.at(1).as_int());
        if (end < begin || end > data_size) throw std::runtime_error(std::format("{}: bad offsets for {}", path, name));
        const uint64_t width = dtype == "I64" ? 8 : dtype == "F32" ? 4 : (dtype == "F16" || dtype == "BF16") ? 2 : 0;
        if (width == 0) throw std::runtime_error(std::format("{}: unsupported dtype {} for {}", path, dtype, name));
        // The element count, multiplied in order with a check at each step, so that a header's shape can't
        // overflow it: no product of leading dimensions may exceed the elements the file has bytes for
        // (once one is 0 the rest can't matter, as for shape_numel).
        for (int64_t dim : shape)
            if (dim < 0) throw std::runtime_error(std::format("{}: negative dimension for {}", path, name));
        const uint64_t max_n = data_size / width;
        uint64_t count = 1;
        for (int64_t dim : shape) {
            if (count != 0 && static_cast<uint64_t>(dim) > max_n / count)
                throw std::runtime_error(std::format("{}: {} has a shape larger than the file", path, name));
            count *= static_cast<uint64_t>(dim);
        }
        const int64_t n = static_cast<int64_t>(count);  // at most data_size / width
        if (end - begin != count * width)
            throw std::runtime_error(std::format("{}: {} byte size does not match its shape", path, name));
        // Straight into the tensor's own storage: no intermediate byte buffer.
        auto read_into = [&](void* dst) { copy(begin, end - begin, dst, name); };
        if (dtype == "I64") {
            IntTensor t{shape, std::vector<int64_t>(static_cast<size_t>(n))};
            read_into(t.data.data());
            out.int_tensors[name] = std::move(t);
            continue;
        }
        if (take_raw) {
            const uint64_t size = end - begin;
            RawTensor raw{dtype,
                          shape,
                          size,
                          [&](uint64_t offset, uint64_t count, void* dst) {
                              if (offset > size || count > size - offset)
                                  throw std::out_of_range(std::format("{}: read past the end of {}", path, name));
                              copy(begin + offset, count, dst, name);
                          },
                          nullptr,
                          nullptr};
            if (data) {
                raw.mapped = data + begin;
                raw.mapping = mapping;
            }
            if (take_raw(name, raw)) continue;
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
