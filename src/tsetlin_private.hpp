#pragma once

#include "tsetlin_status_code.hpp"
#include "tsetlin_state.hpp"
#include "tsetlin_types.hpp"

#include "neither/either.hpp"

#include <vector>

namespace Tsetlin
{


status_message_t
fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int epochs)
    __attribute__((warn_unused_result));


status_message_t
partial_fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int epochs)
    __attribute__((warn_unused_result));


neither::Either<status_message_t, label_vector_type>
predict_impl(
    ClassifierState const & state,
    std::vector<aligned_vector_char> const & X)
    __attribute__((warn_unused_result));


neither::Either<status_message_t, aligned_vector_int>
predict_raw_impl(
    ClassifierState const & state,
    aligned_vector_char const & sample)
    __attribute__((warn_unused_result));


neither::Either<status_message_t, std::vector<aligned_vector_int>>
predict_raw_impl(
    ClassifierState const & state,
    std::vector<aligned_vector_char> const & X)
    __attribute__((warn_unused_result));


} // namespace Tsetlin
