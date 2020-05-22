#pragma once

#include "mt.hpp"
#include "aligned_allocator.hpp"
#include "basic_numeric_matrix.hpp"

#include <vector>
#include <random>
#include <optional>
#include <cstdint>

namespace Tsetlini
{


static constexpr unsigned int alignment = 64;

using real_type = float;
using seed_type = std::random_device::result_type;
using none_type = std::nullopt_t;
using size_type = decltype (sizeof (0));

// aligned vectors
template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, alignment>>;
using aligned_vector_int = aligned_vector<int>;
using aligned_vector_char = aligned_vector<char>;
using aligned_vector_float = aligned_vector<float>;

using aligned_vector_int32 = aligned_vector<std::int32_t>;
using aligned_vector_int16 = aligned_vector<std::int16_t>;
using aligned_vector_int8 = aligned_vector<std::int8_t>;

template<typename T>
using numeric_matrix = basic_numeric_matrix<T, 32>;
using numeric_matrix_int32 = numeric_matrix<std::int32_t>;
using numeric_matrix_int16 = numeric_matrix<std::int16_t>;
using numeric_matrix_int8 = numeric_matrix<std::int8_t>;

using label_type = int;
using label_vector_type = std::vector<label_type>;
using feedback_vector_type = aligned_vector_int;

static constexpr real_type ONE = 1.;
static constexpr real_type ZERO = 0.;


} // namespace Tsetlini
