/* hok - v0.0.0 - Public Domain - https://github.com/jusqua/hok */

#pragma once

#include <sycl/sycl.hpp>

namespace hok::detail {

static constexpr auto s_channels = 4;

template<int dimensions>
inline constexpr auto get_linear_id(const sycl::range<dimensions>& extent, const sycl::id<dimensions>& index) {
    size_t id = 0;
    if constexpr (dimensions == 1) {
        id = index[0];
    } else if constexpr (dimensions == 2) {
        id = index[0] * extent[1] + index[1];
    } else if constexpr (dimensions == 3) {
        id = index[0] * extent[1] * extent[2] + index[1] * extent[2] + index[2];
    } else {
        static_assert(false, "ND not implemented yet");
    }
    return id;
}

template<int dimensions>
inline constexpr auto get_linear_id(const sycl::item<dimensions>& item, const sycl::id<dimensions>& relative_index, const sycl::range<dimensions>& alignment) {
    static_assert(dimensions > 0 && dimensions < 4, "ND not implemented yet");

    sycl::id<dimensions> index = item.get_id();
    sycl::range<dimensions> extent = item.get_range();

    if constexpr (dimensions > 0) {
        index[0] = sycl::clamp(static_cast<int>(index[0]) + static_cast<int>(relative_index[0]) - static_cast<int>(alignment[0]), 0, static_cast<int>(extent[0]) - 1);
    }
    if constexpr (dimensions > 1) {
        index[1] = sycl::clamp(static_cast<int>(index[1]) + static_cast<int>(relative_index[1]) - static_cast<int>(alignment[1]), 0, static_cast<int>(extent[1]) - 1);
    }
    if constexpr (dimensions > 2) {
        index[2] = sycl::clamp(static_cast<int>(index[2]) + static_cast<int>(relative_index[2]) - static_cast<int>(alignment[2]), 0, static_cast<int>(extent[2]) - 1);
    }

    return get_linear_id(extent, index);
}

template<int dimensions>
inline constexpr auto get_linear_id(const sycl::item<dimensions>& item) {
    return get_linear_id(item.get_range(), item.get_id());
}

inline constexpr auto read(const float* data, size_t index) {
    auto value = sycl::float4{0.0f};
    for(int i = 0; i < s_channels; ++i) {
        value[i] = data[index * s_channels + i];
    }
    return value;
}

inline constexpr auto read(const uint8_t* data, size_t index) {
    auto value = sycl::float4{0.0f};
    for(int i = 0; i < s_channels; ++i) {
        value[i] = static_cast<float>(data[index * s_channels + i]) / 255;
    }
    return value;
}

template<typename T, int dimensions>
inline constexpr auto read(const T* data, const sycl::item<dimensions>& item) {
    return read(data, item.get_linear_id());
}

inline constexpr auto write(float* data, size_t index, const sycl::float4& value) {
    for(int i = 0; i < s_channels; ++i) {
        data[index * s_channels + i] = value[i];
    }
}

inline constexpr auto write(uint8_t* data, size_t index, const sycl::float4& value) {
    for(int i = 0; i < s_channels; ++i) {
        data[index * s_channels + i] = static_cast<uint8_t>(value[i] * 255);
    }
}

template<typename T, int dimensions>
inline constexpr auto write(T* data, const sycl::item<dimensions>& item, const sycl::float4& value) {
    write(data, item.get_linear_id(), value);
}

template <typename F, int dimensions>
inline constexpr auto map(const sycl::range<dimensions>& range, const F&& apply) {
    if constexpr (dimensions == 1) {
        for (size_t i = 0; i < range[0]; ++i) {
            apply(sycl::id<dimensions>{i});
        }
    } else if constexpr (dimensions == 2) {
        for (size_t i = 0; i < range[0]; ++i) {
            for (size_t j = 0; j < range[1]; ++j) {
                apply(sycl::id<dimensions>{i, j});
            }
        }
    } else if constexpr (dimensions == 3) {
        for (size_t i = 0; i < range[0]; ++i) {
            for (size_t j = 0; j < range[1]; ++j) {
                for (size_t k = 0; k < range[2]; ++k) {
                    apply(sycl::id<dimensions>{i, j, k});
                }
            }
        }
    } else {
        static_assert(false, "ND not implemented yet");
    }
}

template<typename F>
class unary_kernel_impl {
public:
    unary_kernel_impl(const float* in, float* out, F&& fn)
        : m_in(in), m_out(out), m_fn(std::move(fn)) {}

    template<int Dims>
    void operator()(sycl::item<Dims> item) const {
        auto px = detail::read(m_in, item);
        detail::write(m_out, item, m_fn(px));
    }

private:
    const float* m_in;
    float* m_out;
    F m_fn;
};

template<typename F>
class binary_kernel_impl {
public:
    binary_kernel_impl(const float* in1, const float* in2, float* out, F&& fn)
        : m_in1(in1), m_in2(in2), m_out(out), m_fn(std::move(fn)) {}

    template<int Dims>
    void operator()(sycl::item<Dims> item) const {
        detail::write(m_out, item, m_fn(detail::read(m_in1, item), detail::read(m_in2, item)));
    }

private:
    const float* m_in1;
    const float* m_in2;
    float* m_out;
    F m_fn;
};

template <int dimensions, typename F>
class window_kernel_impl {
public:
    window_kernel_impl(const float* in, float* out, const sycl::range<dimensions>& wextent, const float* wdata, const sycl::float4& init, F&& fn)
        : m_wextent(wextent), m_in(in), m_out(out), m_wdata(wdata), m_init(init), m_whalo(wextent / 2), m_fn(std::move(fn)) {}

    void operator()(sycl::item<dimensions> item) const {
        auto result = m_init;
        detail::map(m_wextent, [&](sycl::id<dimensions> wid) {
            auto px = detail::read(m_in, detail::get_linear_id(item, wid, m_whalo));
            auto value = m_wdata[detail::get_linear_id(m_wextent, wid)];
            m_fn(result, px, value);
        });
        detail::write(m_out, item, result);
    }

private:
    const float* m_in;
    float* m_out;
    F m_fn;

    const sycl::range<dimensions> m_wextent;
    const sycl::range<dimensions> m_whalo;
    const float* m_wdata;
    const sycl::float4 m_init;
};

} // namespace hok::detail

namespace hok {

inline auto gray(const float* input_data, float* output_data) {
    return detail::unary_kernel_impl(input_data, output_data, [](sycl::float4& px) {
        float gray = px.x() * 0.299f + px.y() * 0.587f + px.z() * 0.114f;
        px.x() = px.y() = px.z() = gray;
        return px;
    });
}

template<int dimensions>
[[nodiscard]] inline auto gray(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, gray(input_data, output_data));
}

inline auto thresh(const float* input_data, float* output_data, float threshold) {
    return detail::unary_kernel_impl(input_data, output_data, [threshold](sycl::float4& px) {
        px.x() = px.x() > threshold ? 1.0f : 0.0f;
        px.y() = px.y() > threshold ? 1.0f : 0.0f;
        px.z() = px.z() > threshold ? 1.0f : 0.0f;
        return px;
    });
}

template<int dimensions>
[[nodiscard]] inline auto thresh(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, float threshold, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, thresh(input_data, output_data, threshold));
}

inline auto min(const float* input1_data, const float* input2_data, float* output_data) {
    return detail::binary_kernel_impl(input1_data, input2_data, output_data, [](sycl::float4& px, sycl::float4& px2) {
        return sycl::min(px, px2);
    });
}

template<int dimensions>
[[nodiscard]] inline auto min(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input1_data, const float* input2_data, float* output_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, min(input1_data, input2_data, output_data));
}

inline auto max(const float* input1_data, const float* input2_data, float* output_data) {
    return detail::binary_kernel_impl(input1_data, input2_data, output_data, [](sycl::float4& px, sycl::float4& px2) {
        return sycl::max(px, px2);
    });
}

template<int dimensions>
[[nodiscard]] inline auto max(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input1_data, const float* input2_data, float* output_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, max(input1_data, input2_data, output_data));
}

inline auto sum(const float* input1_data, const float* input2_data, float* output_data) {
    return detail::binary_kernel_impl(input1_data, input2_data, output_data, [](sycl::float4& px, sycl::float4& px2) {
        return sycl::min(px + px2, sycl::float4(1.0f));
    });
}

template<int dimensions>
[[nodiscard]] inline auto sum(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input1_data, const float* input2_data, float* output_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, sum(input1_data, input2_data, output_data));
}

inline auto sub(const float* input1_data, const float* input2_data, float* output_data) {
    return detail::binary_kernel_impl(input1_data, input2_data, output_data, [](sycl::float4& px, sycl::float4& px2) {
        return sycl::max(px - px2, sycl::float4(0.0f));
    });
}

template<int dimensions>
[[nodiscard]] inline auto sub(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input1_data, const float* input2_data, float* output_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, sub(input1_data, input2_data, output_data));
}

template<int dimensions>
inline auto convolve(const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data) {
    return detail::window_kernel_impl(input_data, output_data, window_extent, window_data, sycl::float4(0), [](sycl::float4& acc, sycl::float4& px, float val) {
        acc += px * val;
    });
}

template<int dimensions>
[[nodiscard]] inline auto convolve(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, convolve(input_data, output_data, window_extent, window_data));
}

template<int dimensions>
inline auto erode(const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data) {
    return detail::window_kernel_impl(input_data, output_data, window_extent, window_data, sycl::float4(1), [](sycl::float4& acc, sycl::float4& px, float val) {
        if (val != 0.0f && acc[0] + acc[1] + acc[2] > px[0] + px[1] + px[2]) {
            acc = px;
        }
    });
}

template<int dimensions>
[[nodiscard]] inline auto erode(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, erode(input_data, output_data, window_extent, window_data));
}

template<int dimensions>
inline auto dilate(const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data) {
    return detail::window_kernel_impl(input_data, output_data, window_extent, window_data, sycl::float4(0), [](sycl::float4& acc, sycl::float4& px, float val) {
        if (val != 0.0f && px[0] + px[1] + px[2] > acc[0] + acc[1] + acc[2]) {
            acc = px;
        }
    });
}

template<int dimensions>
[[nodiscard]] inline auto dilate(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return queue.parallel_for(io_extent, events, dilate(input_data, output_data, window_extent, window_data));
}

template<int dimensions>
[[nodiscard]] inline auto binary(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, float threshold, const std::vector<sycl::event>& events = {}) {
    return thresh(queue, io_extent, output_data, output_data, threshold, { gray(queue, io_extent, input_data, output_data, events) });
}

template<int dimensions>
[[nodiscard]] inline auto geodesic_erode(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, const float* mask_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return max(queue, io_extent, mask_data, output_data, output_data, { erode(queue, io_extent, input_data, output_data, window_extent, window_data, events) });
}

template<int dimensions>
[[nodiscard]] inline auto geodesic_dilate(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, const float* mask_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return min(queue, io_extent, mask_data, output_data, output_data, { dilate(queue, io_extent, input_data, output_data, window_extent, window_data, events) });
}

template<int dimensions>
[[nodiscard]] inline auto open(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, float* buffer_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return dilate(queue, io_extent, buffer_data, output_data, window_extent, window_data, { erode(queue, io_extent, input_data, buffer_data, window_extent, window_data, events) });
}

template<int dimensions>
[[nodiscard]] inline auto close(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, float* buffer_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return erode(queue, io_extent, buffer_data, output_data, window_extent, window_data, { dilate(queue, io_extent, input_data, buffer_data, window_extent, window_data, events) });
}

template<int dimensions>
[[nodiscard]] inline auto white_tophat(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, float* buffer_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return sub(queue, io_extent, input_data, buffer_data, output_data, { open(queue, io_extent, input_data, buffer_data, output_data, window_extent, window_data, events) });
}

template<int dimensions>
[[nodiscard]] inline auto black_tophat(sycl::queue& queue, const sycl::range<dimensions>& io_extent, const float* input_data, float* output_data, float* buffer_data, const sycl::range<dimensions>& window_extent, const float* window_data, const std::vector<sycl::event>& events = {}) {
    return sub(queue, io_extent, buffer_data, input_data, output_data, { close(queue, io_extent, input_data, buffer_data, output_data, window_extent, window_data, events) });
}

}  // namespace hok

/*
    ------------------------------------------------------------------------------
    This software is available under 2 licenses -- choose whichever you prefer.
    ------------------------------------------------------------------------------
    ALTERNATIVE A - MIT License
    Copyright (c) 2026 Ádrian Gama
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    ------------------------------------------------------------------------------
    ALTERNATIVE B - Public Domain (www.unlicense.org)
    This is free and unencumbered software released into the public domain.
    Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
    software, either in source code form or as a compiled binary, for any purpose,
    commercial or non-commercial, and by any means.
    In jurisdictions that recognize copyright laws, the author or authors of this
    software dedicate any and all copyright interest in the software to the public
    domain. We make this dedication for the benefit of the public at large and to
    the detriment of our heirs and successors. We intend this dedication to be an
    overt act of relinquishment in perpetuity of all present and future rights to
    this software under copyright law.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ------------------------------------------------------------------------------
*/
