#pragma once

#include "helpers.hpp"
#include "types.hpp"

#include <cuda_fp16.h>

#include <cstddef>

namespace smollnet {

#ifdef __CUDACC__
#define SMOLLNET_HOST_DEVICE __host__ __device__
#else
#define SMOLLNET_HOST_DEVICE
#endif

inline constexpr bool is_supported_float_dtype(DataType dtype) noexcept {
  return dtype == DataType::f16 || dtype == DataType::f32;
}

inline DataType accumulation_dtype(DataType dtype) {
  ASSERT(is_supported_float_dtype(dtype),
         fmt::format("Unsupported dtype {}", get_name(dtype)));
  return DataType::f32;
}

template <typename T> struct ScalarTraits;

template <> struct ScalarTraits<float> {
  static constexpr DataType dtype = DataType::f32;

  SMOLLNET_HOST_DEVICE static float to_float(float value) { return value; }
  SMOLLNET_HOST_DEVICE static float from_float(float value) { return value; }
};

template <> struct ScalarTraits<__half> {
  static constexpr DataType dtype = DataType::f16;

  SMOLLNET_HOST_DEVICE static float to_float(__half value) {
    return __half2float(value);
  }

  SMOLLNET_HOST_DEVICE static __half from_float(float value) {
    return __float2half(value);
  }
};

template <typename T>
SMOLLNET_HOST_DEVICE float scalar_to_float(T value) {
  return ScalarTraits<T>::to_float(value);
}

template <typename T>
SMOLLNET_HOST_DEVICE T scalar_from_float(float value) {
  return ScalarTraits<T>::from_float(value);
}

inline float load_scalar(const void *data, DataType dtype, size_t index) {
  switch (dtype) {
  case DataType::f16:
    return scalar_to_float(static_cast<const __half *>(data)[index]);
  case DataType::f32:
    return static_cast<const float *>(data)[index];
  default:
    ASSERT(false, fmt::format("Unsupported dtype {}", get_name(dtype)));
  }

  __builtin_unreachable();
}

inline void store_scalar(void *data, DataType dtype, size_t index,
                         float value) {
  switch (dtype) {
  case DataType::f16:
    static_cast<__half *>(data)[index] = scalar_from_float<__half>(value);
    return;
  case DataType::f32:
    static_cast<float *>(data)[index] = value;
    return;
  default:
    ASSERT(false, fmt::format("Unsupported dtype {}", get_name(dtype)));
  }

  __builtin_unreachable();
}

template <typename Fn> void dispatch_float_dtype(DataType dtype, Fn &&fn) {
  switch (dtype) {
  case DataType::f16:
    fn.template operator()<__half>();
    return;
  case DataType::f32:
    fn.template operator()<float>();
    return;
  default:
    ASSERT(false, fmt::format("Unsupported dtype {}", get_name(dtype)));
  }

  __builtin_unreachable();
}

} // namespace smollnet
