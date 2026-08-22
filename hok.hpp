/* hok - v0.0.0 - Public Domain - https://github.com/jusqua/hok */

#pragma once

#include <cmath>
#include <sycl/detail/builtins/builtins.hpp>
#include <sycl/marray.hpp>
#include <sycl/sycl.hpp>

namespace hok::strategy {

enum class gray {
    average,
    luminance_bt601,
    luminance_bt709,
    decomposition_min,
    decomposition_max,
    desaturation,
    red,
    green,
    blue,
};

}

namespace hok::detail {

namespace meta {

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

template<strategy::gray strategy>
inline constexpr auto gray(const sycl::float4& pixel) -> float {
    auto gray = 0.0f;
    switch (strategy) {
        case strategy::gray::average:
            gray = (pixel.x() + pixel.y()+ pixel.z()) / 3.f; break;
        case strategy::gray::luminance_bt601:
            gray = pixel.x() * 0.299f + pixel.y() * 0.587f + pixel.z() * 0.114f; break;
        case strategy::gray::luminance_bt709:
            gray = pixel.x() * 0.2126f + pixel.y() * 0.7152f + pixel.z() * 0.0722f; break;
        case strategy::gray::decomposition_min:
            gray = sycl::min(pixel.x(), sycl::min(pixel.y(), pixel.z())); break;
        case strategy::gray::decomposition_max:
            gray = sycl::max(pixel.x(), sycl::max(pixel.y(), pixel.z())); break;
        case strategy::gray::desaturation:
            gray = (sycl::max(pixel.x(), sycl::max(pixel.y(), pixel.z())) + sycl::min(pixel.x(), sycl::min(pixel.y(), pixel.z()))) / 2.f; break;
        case strategy::gray::red:
            gray = pixel.x(); break;
        case strategy::gray::green:
            gray = pixel.y(); break;
        case strategy::gray::blue:
            gray = pixel.z(); break;
    }
    return gray;
}

} // namespace hok::detail

// TODO: Document the kernels
// TODO: Provide a better wrapper for nd_item handling
namespace hok {

template<int dimensions = 1, typename T>
inline auto intensity(const sycl::range<dimensions>& io_extent, const T* input, T* output, float factor) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px = detail::read(input, item);
            auto val = px + factor;
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto contrast(const sycl::range<dimensions>& io_extent, const T* input, T* output, float factor) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px = detail::read(input, item);
            auto val = px * factor;
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto invert(const sycl::range<dimensions>& io_extent, const T* input, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px = detail::read(input, item);
            auto val = 1.0f - px;
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, strategy::gray strategy = strategy::gray::luminance_bt601, typename T>
inline auto gray(const sycl::range<dimensions>& io_extent, const T* input, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px = detail::read(input, item);
            auto g = detail::gray<strategy>(px);
            auto val = sycl::float4{g, g, g, px.w()};
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto thresh(const sycl::range<dimensions>& io_extent, const T* input, T* output, float threshold) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px = detail::read(input, item);
            auto val = sycl::float4{
                px.x() > threshold ? 1.0f : 0.0f,
                px.y() > threshold ? 1.0f : 0.0f,
                px.z() > threshold ? 1.0f : 0.0f,
                px.w()
            };
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, strategy::gray strategy = strategy::gray::luminance_bt601, typename T>
inline auto binary(const sycl::range<dimensions>& io_extent, const T* input, T* output, float threshold) {
    return [=](sycl::handler& cgh) {
        gray<dimensions, strategy>(io_extent, input, output)(cgh);
        thresh<dimensions>(io_extent, input, output, threshold)(cgh);
    };
}

template<int dimensions = 1, typename T>
inline auto min(const sycl::range<dimensions>& io_extent, const T* input1, const T* input2, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px1 = detail::read(input1, item);
            auto px2 = detail::read(input2, item);
            auto val = sycl::min(px1, px2);
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto max(const sycl::range<dimensions>& io_extent, const T* input1, const T* input2, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px1 = detail::read(input1, item);
            auto px2 = detail::read(input2, item);
            auto val = sycl::max(px1, px2);
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto sum(const sycl::range<dimensions>& io_extent, const T* input1, const T* input2, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px1 = detail::read(input1, item);
            auto px2 = detail::read(input2, item);
            auto val = sycl::min(sycl::float4{1.0f}, px1 + px2);
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto sub(const sycl::range<dimensions>& io_extent, const T* input1, const T* input2, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px1 = detail::read(input1, item);
            auto px2 = detail::read(input2, item);
            auto val = sycl::max(sycl::float4{0.0f}, px1 - px2);
            detail::write(output, item, val);
        });
    };
}

template<int dimensions = 1, typename T>
inline auto mul(const sycl::range<dimensions>& io_extent, const T* input1, const T* input2, T* output) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto px1 = detail::read(input1, item);
            auto px2 = detail::read(input2, item);
            auto val = sycl::min(sycl::float4{1.0f}, px1 * px2);
            detail::write(output, item, val);
        });
    };
}

template<int dimensions, typename T>
inline auto gaussian(const sycl::range<dimensions>& io_extent, const T* input, T* output, double sigma) {
    return [=](sycl::handler& cgh) {
        auto radius = static_cast<size_t>(2 * detail::s_pi * sigma + 1);
        auto extent = detail::repeat<dimensions>(radius);
        auto halo = extent / 2;

        auto coeff = -.5 / sigma / sigma;
        auto normal = 1.0;
        for (auto i = 0; i < dimensions; i++) {
            normal *= sycl::sqrt(2 * detail::s_pi * sigma * sigma);
        }

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto result = sycl::float4(0);

            detail::map(extent, [&](sycl::id<dimensions> id) {
                auto px = detail::read(input, detail::get_linear_id(item, id, halo));
                auto pixel_dist = detail::sum_sqr(id, halo);
                auto weight = sycl::exp(pixel_dist * coeff) / normal;

                result += px * weight;
            });

            detail::write(output, item, result);
        });
    };
}

template<int dimensions, typename T>
inline auto bilateral(const sycl::range<dimensions>& io_extent, const T* input, T* output, double sigma_space, double sigma_color) {
    return [=](sycl::handler& cgh) {
        auto radius = static_cast<size_t>(2 * detail::s_pi * sigma_space + 1);
        auto extent = detail::repeat<dimensions>(radius);
        auto halo = extent / 2;

        auto space_coeff = -.5 / sigma_space / sigma_space;
        auto color_coeff = -.5 / sigma_color / sigma_color;

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto result_sum = sycl::float4{};
            auto weight_sum = sycl::float4{};
            auto curr_pixel = detail::read(input, item);

            detail::map(extent, [&](sycl::id<dimensions> id) {
                auto rel_pixel = detail::read(input, detail::get_linear_id(item, id, halo));
                auto pixel_diff = detail::sqr_abs_diff(curr_pixel, rel_pixel);
                auto pixel_dist = detail::sum_sqr(id, halo);

                auto weight = sycl::exp(pixel_dist * space_coeff + pixel_diff * color_coeff);
                result_sum += rel_pixel * weight;
                weight_sum += weight;
            });

            detail::write(output, item, result_sum / weight_sum);
        });
    };
}

template<int dimensions, typename T>
inline auto average(const sycl::range<dimensions>& io_extent, const T* input, T* output, size_t radius) {
    return [=](sycl::handler& cgh) {
        auto extent = detail::repeat<dimensions>(2 * radius + 1);
        auto halo = extent / 2;

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto result = sycl::float4(0);

            detail::map(extent, [&](sycl::id<dimensions> id) {
                auto px = detail::read(input, detail::get_linear_id(item, id, halo));
                result += px;
            });

            result /= extent.size();
            detail::write(output, item, result);
        });
    };
}

template<size_t radius, int dimensions, typename T>
inline auto median(const sycl::range<dimensions>& io_extent, const T* input, T* output) {
    return [=](sycl::handler& cgh) {
        constexpr auto buffer_size = detail::meta::pow_v<2 * radius + 1, dimensions>();
        auto extent = detail::repeat<dimensions>(2 * radius + 1);
        auto halo = extent / 2;

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto count = 0;
            auto buffer = sycl::marray<sycl::float4, buffer_size>{};

            detail::map(extent, [&](sycl::id<dimensions> id) {
                auto px = detail::read(input, detail::get_linear_id(item, id, halo));
                buffer[count++] = px;
            });

            for (auto i = 0; i < count - 1; i++) {
                auto swapped = false;
                for (auto j = 0; j < count - i - 1; j++) {
                    if (buffer[j].x() + buffer[j].y() + buffer[j].z() <= buffer[j + 1].x() + buffer[j + 1].y() + buffer[j + 1].z())
                        continue;

                    swapped = true;
                    auto tmp = buffer[j];
                    buffer[j] = buffer[j + 1];
                    buffer[j + 1] = tmp;
                }

                if (!swapped) break;
            }

            detail::write(output, item, buffer[(count - 1) / 2]);
        });
    };
}

template<int dimensions, typename T>
inline auto convolve(const sycl::range<dimensions>& io_extent, const T* input, T* output, const sycl::range<dimensions>& window_extent, const float* window) {
    return [=](sycl::handler& cgh) {
        auto window_halo = window_extent / 2;

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto result = sycl::float4{};

            detail::map(window_extent, [&](sycl::id<dimensions> id) {
                auto px = detail::read(input, detail::get_linear_id(item, id, window_halo));
                auto weight = window[detail::get_linear_id(window_extent, id)];
                result += px * weight;
            });

            detail::write(output, item, result);
        });
    };
}

template<int dimensions, typename T>
inline auto erode(const sycl::range<dimensions>& io_extent, const T* input, T* output, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        auto strel_halo = strel_extent;

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto result = sycl::float4{0.0f};
            auto sum = 0.0f;

            detail::map(strel_extent, [&](sycl::id<dimensions> id) {
                auto px = detail::read(input, detail::get_linear_id(item, id, strel_halo));
                auto enabled = strel[detail::get_linear_id(strel_extent, id)];
                if (!enabled) return;

                auto new_sum = px.x() + px.y() + px.z();
                if (sum > new_sum) {
                    sum = new_sum;
                    result = px;
                }
            });

            detail::write(output, item, result);
        });
    };
}

template<int dimensions, typename T>
inline auto dilate(const sycl::range<dimensions>& io_extent, const T* input, T* output, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        auto strel_halo = strel_extent;

        cgh.parallel_for(io_extent, [=](sycl::item<dimensions> item) {
            auto result = sycl::float4{1.0f};
            auto sum = 3.0f;

            detail::map(strel_extent, [&](sycl::id<dimensions> id) {
                auto px = detail::read(input, detail::get_linear_id(item, id, strel_halo));
                auto enabled = strel[detail::get_linear_id(strel_extent, id)];
                if (!enabled) return;

                auto new_sum = px.x() + px.y() + px.z();
                if (sum < new_sum) {
                    sum = new_sum;
                    result = px;
                }
            });

            detail::write(output, item, result);
        });
    };
}

template<int dimensions, typename T>
inline auto geodesic_erode(const sycl::range<dimensions>& io_extent, const T* marker, const T* mark, T* output, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, erode<dimensions>(marker, output, strel_extent, strel));
        cgh.parallel_for(io_extent, max(output, mark, output));
    };
}

template<int dimensions, typename T>
inline auto geodesic_dilate(const sycl::range<dimensions>& io_extent, const T* marker, const T* mark, T* output, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        cgh.parallel_for(io_extent, dilate<dimensions>(marker, output, strel_extent, strel));
        cgh.parallel_for(io_extent, min(output, mark, output));
    };
}

template<int dimensions, typename T>
inline auto open(const sycl::range<dimensions>& io_extent, const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        erode(io_extent, input, buffer, strel_extent, strel)(cgh);
        dilate(io_extent, buffer, output, strel_extent, strel)(cgh);
    };
}

template<int dimensions, typename T>
inline auto close(const sycl::range<dimensions>& io_extent, const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        dilate(io_extent, input, buffer, strel_extent, strel)(cgh);
        erode(io_extent, buffer, output, strel_extent, strel)(cgh);
    };
}

template<int dimensions, typename T>
inline auto white_tophat(const sycl::range<dimensions>& io_extent, const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        open(io_extent, input, buffer, output, strel_extent, strel)(cgh);
        sub(io_extent, input, buffer, output)(cgh);
    };
}

template<int dimensions, typename T>
inline auto black_tophat(const sycl::range<dimensions>& io_extent, const T* input, T* output, T* buffer, const sycl::range<dimensions>& strel_extent, const bool* strel) {
    return [=](sycl::handler& cgh) {
        open(io_extent, input, buffer, output, strel_extent, strel)(cgh);
        sub(io_extent, buffer, input, output)(cgh);
    };
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
