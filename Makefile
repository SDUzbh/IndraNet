# Build the tfi engine (library), the indra CLI and the test tools with make (no CMake needed).
#   make            -> build/libtfi.a build/indra build/tfi_golden build/tfi_tests
#   make test       -> C++ unit tests (golden tests against PyTorch run from pytest)
#   make python     -> the Python bindings in place, python/tfi/_tfi*.so, for working on them (CMake + nanobind:
#                      pip install nanobind; PYTHON=... picks the interpreter, python3 by default). To use them,
#                      pip install . instead (pyproject.toml).
#   make install    -> libtfi.a, the headers and a pkg-config file under PREFIX (/usr/local by default), so that
#                      c++ app.cpp $(pkg-config --cflags --libs tfi) builds against the engine
#   make test GROUPS=gpu,fp16 FILTER=attention -> only those groups / tests whose name contains the text
#                      (groups: core analysis grad fp16 gpu; build/tfi_tests --list shows them)
# CXXFLAGS set on the command line adds to the required flags instead of replacing them:
#   make CXXFLAGS="-O1 -g -fsanitize=address,undefined" test
CXXFLAGS ?= -O2
override CXXFLAGS += -std=c++20 -Wall -Wextra -Iinclude -pthread
LDLIBS :=

BUILD := build
LIB_SRCS := $(wildcard src/*.cpp src/*/*.cpp)
LIB_OBJS := $(LIB_SRCS:src/%.cpp=$(BUILD)/obj/%.o)

# macOS 26 or newer, built with its SDK (Xcode 26): Accelerate for matrix multiplies, Metal 4 for the GPU path
# (src/gpu/gpu.mm).
ifeq ($(shell uname -s),Darwin)
  override CXXFLAGS += -DTFI_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -mmacosx-version-min=26.0
  LDLIBS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
  LIB_OBJS += $(BUILD)/obj/gpu/gpu.o
endif
HEADERS := $(wildcard include/tfi/*.h src/*.h src/*/*.h)

all: $(BUILD)/libtfi.a $(BUILD)/indra $(BUILD)/tfi_golden $(BUILD)/tfi_tests

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/%.o: src/%.mm $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -fobjc-arc -c $< -o $@

$(BUILD)/libtfi.a: $(LIB_OBJS)
	ar rcs $@ $^

$(BUILD)/indra: tools/indra.cpp $(BUILD)/libtfi.a $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libtfi.a $(LDLIBS) -o $@

$(BUILD)/tfi_golden: tests/golden.cpp $(BUILD)/libtfi.a $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libtfi.a $(LDLIBS) -o $@

$(BUILD)/tfi_tests: tests/unit.cpp $(BUILD)/libtfi.a $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libtfi.a $(LDLIBS) -o $@

test: $(BUILD)/tfi_tests
	$(BUILD)/tfi_tests $(if $(GROUPS),--group $(GROUPS)) $(if $(FILTER),--filter "$(FILTER)")

PYTHON ?= python3
python:
	cmake -S . -B $(BUILD)/python -DTFI_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$$(command -v $(PYTHON))" >/dev/null
	cmake --build $(BUILD)/python --target _tfi --parallel

PREFIX ?= /usr/local
install: $(BUILD)/libtfi.a
	mkdir -p $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(PREFIX)/include/tfi
	cp $(BUILD)/libtfi.a $(DESTDIR)$(PREFIX)/lib/
	cp include/tfi/*.h $(DESTDIR)$(PREFIX)/include/tfi/
	printf 'prefix=%s\nName: tfi\nDescription: IndraNet reverse-engineering engine\nVersion: 0.1.0\nCflags: -std=c++20 -I$${prefix}/include\nLibs: -L$${prefix}/lib -ltfi -pthread %s\n' \
	  '$(PREFIX)' '$(LDLIBS)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/tfi.pc

clean:
	rm -rf $(BUILD)

.PHONY: all test python install clean
