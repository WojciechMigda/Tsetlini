#pragma once

#include "mt.hpp"
#include "aligned_allocator.hpp"

#include <vector>
#include <random>

namespace Tsetlin
{

static constexpr unsigned int alignment = 64;

using real_type = float;
using seed_type = std::mt19937::result_type;

// aligned vectors
template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, alignment>>;
using aligned_vector_int = aligned_vector<int>;
using aligned_vector_char = aligned_vector<char>;
using aligned_vector_float = aligned_vector<float>;

using y_vector_type = std::vector<int>;
using feedback_vector_type = aligned_vector_int;

static constexpr real_type ONE = 1.;
static constexpr real_type ZERO = 0.;

} // namespace Tsetlin
