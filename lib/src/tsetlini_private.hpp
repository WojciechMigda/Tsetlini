#pragma once

#ifndef LIB_SRC_TSETLINI_PRIVATE_HPP_
#define LIB_SRC_TSETLINI_PRIVATE_HPP_

#include "tsetlini_status_code.hpp"
#include "estimator_state.hpp"
#include "tsetlini_types.hpp"

#include "neither/either.hpp"

#include <vector>


namespace Tsetlini
{


[[nodiscard]]
status_message_t
fit_impl(
    ClassifierStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs);


[[nodiscard]]
status_message_t
partial_fit_impl(
    ClassifierStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs);


[[nodiscard]]
neither::Either<status_message_t, label_vector_type>
predict_impl(
    ClassifierStateClassic const & state,
    std::vector<aligned_vector_char> const & X);


[[nodiscard]]
neither::Either<status_message_t, aligned_vector_int>
predict_raw_impl(
    ClassifierStateClassic const & state,
    aligned_vector_char const & sample);


[[nodiscard]]
neither::Either<status_message_t, std::vector<aligned_vector_int>>
predict_raw_impl(
    ClassifierStateClassic const & state,
    std::vector<aligned_vector_char> const & X);


[[nodiscard]]
status_message_t
fit_impl(
    RegressorStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    response_vector_type const & y,
    unsigned int epochs);


[[nodiscard]]
status_message_t
partial_fit_impl(
    RegressorStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    response_vector_type const & y,
    unsigned int epochs);


[[nodiscard]]
neither::Either<status_message_t, response_vector_type>
predict_impl(
    RegressorStateClassic const & state,
    std::vector<aligned_vector_char> const & X);


} // namespace Tsetlini

#endif /* LIB_SRC_TSETLINI_PRIVATE_HPP_ */
