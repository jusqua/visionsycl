/* hok - v0.0.0 - Public Domain - https://github.com/jusqua/hok */

#pragma once

#include <sycl/sycl.hpp>

// TODO: Document
namespace hok {

namespace detail {

// XXX: Exist another way to do this?
template <size_t base, size_t exp>
struct pow {
    static const size_t value = base * pow<base, exp - 1>::value;
};
template <size_t base>
struct pow<base, 0> {
    static const size_t value = 1;
};

template <size_t base, size_t exp>
constexpr auto pow_v() {
    return pow<base, exp>::value;
}

static constexpr auto s_pi = 3.14159265358979323846;
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

inline auto abs_diff(const sycl::float4& px1, const sycl::float4& px2) {
    auto diff = px1 - px2;
    for (auto i = 0; i < 4; i++) {
        if (diff[i] < 0.0f) {
            diff[i] = -diff[i];
        }
    }
    return diff;
}

inline auto sqr_abs_diff(const sycl::float4& px1, const sycl::float4& px2) {
    auto diff = abs_diff(px1, px2);
    return diff * diff;
}

template<int dimensions>
inline auto sum_sqr(const sycl::id<dimensions>& id, const sycl::range<dimensions>& halo) {
    auto result = 0.0;
    for (auto i = 0; i < dimensions; i++) {
        auto diff = static_cast<int>(id[i]) - static_cast<int>(halo[i]);
        result += diff * diff;
    }
    return result;
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

template<int dimensions>
inline auto repeat(size_t value) {
    auto range = sycl::range<dimensions>{};
    for (auto i = 0; i < dimensions; i++) {
        range[i] = value;
    }
    return range;
}

} // namespace detail

// TODO: Create some kernel launch utility
namespace wrapper {

template<int dimensions = 1, typename T, typename F>
inline constexpr auto unary(const T* input, T* output, F&& fn) {
    return [=](sycl::item<dimensions> item) {
        auto px = detail::read(input, item);
        auto val = fn(px);
        detail::write(output, item, val);
    };
}

template<int dimensions = 1, typename T, typename F>
inline constexpr auto binary(const T* input1, const T* input2, T* output, F&& fn) {
    return [=](sycl::item<dimensions> item) {
        auto px1 = detail::read(input1, item);
        auto px2 = detail::read(input2, item);
        auto val = fn(px1, px2);
        detail::write(output, item, val);
    };
}

template<int dimensions, typename T, typename F>
inline constexpr auto map(const T* input, T* output, const sycl::range<dimensions>& extent, const sycl::float4& init, F&& fn) {
    auto halo = extent / 2;

    return [=](sycl::item<dimensions> item) {
        auto acc = init;

        detail::map(extent, [&](sycl::id<dimensions> id) {
            auto px = detail::read(input, detail::get_linear_id(item, id, halo));
            fn(acc, px, id);
        });

        detail::write(output, item, acc);
    };
}

template<int dimensions, typename T, typename F>
inline constexpr auto map(const T* input, T* output, const sycl::range<dimensions>& extent, F&& fn) {
    return map(input, output, extent, sycl::float4{0.0f}, fn);
}

} // namespace wrapper

namespace strategy {

enum class gray {
    mean,
    luminance_bt601,
    luminance_bt709,
    decomposition_min,
    decomposition_max,
    desaturation,
    red,
    green,
    blue,
};

enum class thresh {
    normal,
    invert,
};

} // namespace strategy

template<int dimensions = 1, typename T>
inline auto intensity(const T* input, T* output, float factor) {
    return wrapper::unary<dimensions>(input, output,
        [=](const sycl::float4& px) {
            return sycl::float4{
                sycl::clamp(px.x() + factor, 0.0f, 1.0f),
                sycl::clamp(px.y() + factor, 0.0f, 1.0f),
                sycl::clamp(px.z() + factor, 0.0f, 1.0f),
                px.w()
            };
        }
    );
}

template<int dimensions = 1, typename T>
inline auto contrast(const T* input, T* output, float factor) {
    return wrapper::unary<dimensions>(input, output,
        [=](const sycl::float4& px) {
            return sycl::float4{
                sycl::clamp(px.x() * factor, 0.0f, 1.0f),
                sycl::clamp(px.y() * factor, 0.0f, 1.0f),
                sycl::clamp(px.z() * factor, 0.0f, 1.0f),
                px.w()
            };
        }
    );
}

template<int dimensions = 1, typename T>
inline auto invert(const T* input, T* output) {
    return wrapper::unary<dimensions>(input, output,
        [=](const sycl::float4& px) {
            return sycl::float4{
                1.0f - px.x(),
                1.0f - px.y(),
                1.0f - px.z(),
                px.w()
            };
        }
    );
}

template<int dimensions = 1, strategy::gray strategy = strategy::gray::luminance_bt601, typename T>
inline auto gray(const T* input, T* output) {
    return wrapper::unary<dimensions>(input, output,
        [=](const sycl::float4& px) {
            auto g = 0.0f;
            switch (strategy) {
                case strategy::gray::mean:
                    g = (px.x() + px.y()+ px.z()) / 3.f; break;
                case strategy::gray::luminance_bt601:
                    g = px.x() * 0.299f + px.y() * 0.587f + px.z() * 0.114f; break;
                case strategy::gray::luminance_bt709:
                    g = px.x() * 0.2126f + px.y() * 0.7152f + px.z() * 0.0722f; break;
                case strategy::gray::decomposition_min:
                    g = sycl::min(px.x(), sycl::min(px.y(), px.z())); break;
                case strategy::gray::decomposition_max:
                    g = sycl::max(px.x(), sycl::max(px.y(), px.z())); break;
                case strategy::gray::desaturation:
                    g = (sycl::max(px.x(), sycl::max(px.y(), px.z()))
                        + sycl::min(px.x(), sycl::min(px.y(), px.z())))
                        / 2.f; break;
                case strategy::gray::red:
                    g = px.x(); break;
                case strategy::gray::green:
                    g = px.y(); break;
                case strategy::gray::blue:
                    g = px.z(); break;
            }
            return sycl::float4{g, g, g, px.w()};
        }
    );
}

template<int dimensions = 1, strategy::thresh strategy = strategy::thresh::normal, typename T>
inline auto thresh(const T* input, T* output, float threshold = 0.5f) {
    return wrapper::unary<dimensions>(input, output,
        [=](const sycl::float4& px) {
            if constexpr (strategy == strategy::thresh::normal) {
                return sycl::float4{
                    px.x() > threshold ? 1.0f : 0.0f,
                    px.y() > threshold ? 1.0f : 0.0f,
                    px.z() > threshold ? 1.0f : 0.0f,
                    px.w()
                };
            } else if constexpr (strategy == strategy::thresh::invert) {
                return sycl::float4{
                    px.x() < threshold ? 1.0f : 0.0f,
                    px.y() < threshold ? 1.0f : 0.0f,
                    px.z() < threshold ? 1.0f : 0.0f,
                    px.w()
                };
            }
        }
    );
}

template<int dimensions = 1,
        strategy::thresh thresh_strategy = strategy::thresh::normal,
        strategy::gray gray_strategy = strategy::gray::luminance_bt601,
        typename T>
inline constexpr auto binary(const T* input, T* output, float threshold = 0.5f) {
    return std::make_tuple(
        gray<dimensions, gray_strategy>(input, output),
        thresh<dimensions, thresh_strategy>(output, output, threshold)
    );
}

template<int dimensions = 1, typename T>
inline auto min(const T* input1, const T* input2, T* output) {
    return wrapper::binary<dimensions>(input1, input2, output,
        [=](const sycl::float4& px1, const sycl::float4& px2) {
            return sycl::min(px1, px2);
        }
    );
}

template<int dimensions = 1, typename T>
inline auto max(const T* input1, const T* input2, T* output) {
    return wrapper::binary<dimensions>(input1, input2, output,
        [=](const sycl::float4& px1, const sycl::float4& px2) {
            return sycl::max(px1, px2);
        }
    );
}

template<int dimensions = 1, typename T>
inline auto sum(const T* input1, const T* input2, T* output) {
    return wrapper::binary<dimensions>(input1, input2, output,
        [=](const sycl::float4& px1, const sycl::float4& px2) {
            return sycl::float4{
                sycl::min(1.0f, px1.x() + px2.x()),
                sycl::min(1.0f, px1.y() + px2.y()),
                sycl::min(1.0f, px1.z() + px2.z()),
                1.0f
            };
        }
    );
}

template<int dimensions = 1, typename T>
inline auto sub(const T* input1, const T* input2, T* output) {
    return wrapper::binary<dimensions>(input1, input2, output,
        [=](const sycl::float4& px1, const sycl::float4& px2) {
            return sycl::float4{
                sycl::max(0.0f, px1.x() - px2.x()),
                sycl::max(0.0f, px1.y() - px2.y()),
                sycl::max(0.0f, px1.z() - px2.z()),
                1.0f
            };
        }
    );
}

template<int dimensions = 1, typename T>
inline auto mul(const T* input1, const T* input2, T* output) {
    return wrapper::binary<dimensions>(input1, input2, output,
        [=](const sycl::float4& px1, const sycl::float4& px2) {
            return sycl::float4{
                sycl::min(1.0f, px1.x() * px2.x()),
                sycl::min(1.0f, px1.y() * px2.y()),
                sycl::min(1.0f, px1.z() * px2.z()),
                1.0f
            };
        }
    );
}

template<int dimensions, typename T>
inline auto gaussian(const T* input, T* output, double sigma) {
    auto radius = static_cast<size_t>(2 * detail::s_pi * sigma + 1);
    auto extent = detail::repeat<dimensions>(radius);
    auto halo = extent / 2;

    auto coeff = -.5 / sigma / sigma;
    auto normal = 1.0;
    for (auto i = 0; i < dimensions; i++)
        normal *= sycl::sqrt(2 * detail::s_pi * sigma * sigma);

    return wrapper::map(input, output, extent,
        [=](sycl::float4& acc, const sycl::float4& px, const sycl::id<dimensions>& id) {
            auto pixel_dist = detail::sum_sqr(id, halo);
            auto weight = sycl::exp(pixel_dist * coeff) / normal;

            acc += px * weight;
        }
    );
}

template<int dimensions, typename T>
inline auto bilateral(const T* input, T* output, double sigma_space, double sigma_color) {
    auto radius = static_cast<size_t>(2 * detail::s_pi * sigma_space + 1);
    auto extent = detail::repeat<dimensions>(radius);
    auto halo = extent / 2;

    auto space_coeff = -.5 / sigma_space / sigma_space;
    auto color_coeff = -.5 / sigma_color / sigma_color;

    return [=](sycl::item<dimensions> item) {
        auto acc = sycl::float4{};
        auto wacc = sycl::float4{};
        auto cpx = detail::read(input, item);

        detail::map(extent, [&](sycl::id<dimensions> id) {
            auto px = detail::read(input, detail::get_linear_id(item, id, halo));
            auto px_diff = detail::sqr_abs_diff(cpx, px);
            auto px_dist = detail::sum_sqr(id, halo);

            auto weight = sycl::exp(px_dist * space_coeff + px_diff * color_coeff);
            acc += px * weight;
            wacc += weight;
        });

        acc = acc / wacc;
        detail::write(output, item, acc);
    };
}

template<int dimensions, typename T>
inline auto mean(const T* input, T* output, size_t radius = 1) {
    auto extent = detail::repeat<dimensions>(2 * radius + 1);
    auto halo = extent / 2;

    return [=](sycl::item<dimensions> item) {
        auto acc = sycl::float4{};

        detail::map(extent, [&](sycl::id<dimensions> id) {
            auto px = detail::read(input, detail::get_linear_id(item, id, halo));
            acc += px;
        });

        acc /= extent.size();
        detail::write(output, item, acc);
    };
}

// TODO: make a median filter based on a quantile estimator to handle get median without the need to set comptime radius
template<int dimensions, size_t radius = 1, typename T>
inline auto median(const T* input, T* output) {
    constexpr auto size = detail::pow_v<2 * radius + 1, dimensions>();
    auto extent = detail::repeat<dimensions>(2 * radius + 1);
    auto halo = extent / 2;

    return [=](sycl::item<dimensions> item) {
        auto buf = sycl::marray<sycl::float4, size>{};
        auto c = 0;

        detail::map(extent, [&](sycl::id<dimensions> id) {
            auto px = detail::read(input, detail::get_linear_id(item, id, halo));
            buf[c++] = px;
        });

        for (size_t i = 0; i < size - 1; i++) {
            auto swapped = false;
            for (size_t j = 0; j < size - i - 1; j++) {
                if (buf[j].x() + buf[j].y() + buf[j].z() <= buf[j + 1].x() + buf[j + 1].y() + buf[j + 1].z())
                    continue;

                swapped = true;
                auto tmp = buf[j];
                buf[j] = buf[j + 1];
                buf[j + 1] = tmp;
            }

            if (!swapped) break;
        }

        detail::write(output, item, buf[(c - 1) / 2]);
    };
}

template<int dimensions, typename T>
inline auto convolve(const T* input, T* output, const sycl::range<dimensions>& filter_extent, const float* filter) {
    return wrapper::map(input, output, filter_extent,
        [=](sycl::float4& acc, const sycl::float4& px, const sycl::id<dimensions>& id) {
            auto weight = filter[detail::get_linear_id(filter_extent, id)];
            acc += px * weight;
        }
    );
}

template<int dimensions, typename T>
inline auto erode(const T* input, T* output, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return wrapper::map(input, output, strel_extent,
        [=](sycl::float4& acc, const sycl::float4& px, const sycl::id<dimensions>& id) {
            auto enabled = strel[detail::get_linear_id(strel_extent, id)];
            if (!enabled || (acc.x() + acc.y() + acc.z() >= px.x() + px.y() + px.z())) return;

            acc = px;
        }
    );
}

template<int dimensions, typename T>
inline auto dilate(const T* input, T* output, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return wrapper::map(input, output, strel_extent, sycl::float4{1.0f},
        [=](sycl::float4& acc, const sycl::float4& px, const sycl::id<dimensions>& id) {
            auto enabled = strel[detail::get_linear_id(strel_extent, id)];
            if (!enabled || (acc.x() + acc.y() + acc.z() <= px.x() + px.y() + px.z())) return;

            acc = px;
        }
    );
}

template<int dimensions, typename T>
inline auto open(const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return std::make_tuple(
        erode<dimensions>(input, buffer, strel_extent, strel),
        dilate<dimensions>(buffer, output, strel_extent, strel)
    );
}

template<int dimensions, typename T>
inline auto close(const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return std::make_tuple(
        dilate<dimensions>(input, buffer, strel_extent, strel),
        erode<dimensions>(buffer, output, strel_extent, strel)
    );
}

template<int dimensions, typename T>
inline auto white_tophat(const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return std::tuple_cat(
        open<dimensions>(input, buffer, output, strel_extent, strel),
        std::make_tuple(sub<dimensions>(input, buffer, output))
    );
}

template<int dimensions, typename T>
inline auto black_tophat(const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return std::tuple_cat(
        close<dimensions>(input, buffer, output, strel_extent, strel),
        std::make_tuple(sub<dimensions>(buffer, input, output))
    );
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
