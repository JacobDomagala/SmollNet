#pragma once

#include <cstddef>
#include <cstdint>

namespace smollnet {

enum class Device
{
   CUDA,
   CPU
};

enum class DataType
{
   f8,
   f16,
   f32,
   f64,

   i8,
   i16,
   i32,
   i64
};

constexpr const char*
get_name(DataType t) noexcept
{
   switch (t)
   {
      case DataType::f8:
         return "f8";
      case DataType::f16:
         return "f16";
      case DataType::f32:
         return "f32";
      case DataType::f64:
         return "f64";
      case DataType::i8:
         return "i8";
      case DataType::i16:
         return "i16";
      case DataType::i32:
         return "i32";
      case DataType::i64:
         return "i64";
      default:
         return "UnknownType";
   }
}


inline constexpr size_t
element_size(DataType t) noexcept
{
   switch (t)
   {
      case DataType::f8:
      case DataType::i8:
         return 1;

      case DataType::f16:
      case DataType::i16:
         return 2;

      case DataType::f32:
      case DataType::i32:
         return 4;

      case DataType::f64:
      case DataType::i64:
         return 8;
   }
   __builtin_unreachable();
}

inline constexpr size_t
product(const int64_t* dims, size_t num) noexcept
{
   size_t r = 1;
   for (int d = 0; d < num; d++)
   {
      r *= static_cast< size_t >(dims[d]);
   }
   return r;
}

} // namespace smollnet
