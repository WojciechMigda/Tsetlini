#pragma once

#include "neither/either.hpp"

#include "tsetlini_params.hpp"
#include "tsetlini_status_code.hpp"

#include "tsetlini_types.hpp"
#include "tsetlini_state.hpp"

#include <vector>

namespace Tsetlini
{

using namespace neither;

struct ClassifierClassic
{
    [[nodiscard]]
    status_message_t
    fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y,
        int max_number_of_labels, unsigned int epochs = 100);

    [[nodiscard]]
    status_message_t
    partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y,
        int max_number_of_labels, unsigned int epochs = 100);

    [[nodiscard]]
    Either<status_message_t, real_type>
    evaluate(std::vector<aligned_vector_char> const & X, label_vector_type const & y) const;

    [[nodiscard]]
    Either<status_message_t, label_type>
    predict(aligned_vector_char const & sample) const;

    [[nodiscard]]
    Either<status_message_t, label_vector_type>
    predict(std::vector<aligned_vector_char> const & X) const;

    [[nodiscard]]
    Either<status_message_t, aligned_vector_int>
    predict_raw(aligned_vector_char const & sample) const;

    [[nodiscard]]
    Either<status_message_t, std::vector<aligned_vector_int>>
    predict_raw(std::vector<aligned_vector_char> const & X) const;


    params_t read_params() const;
    ClassifierState read_state() const;

    ClassifierClassic(ClassifierState const & state);

friend Either<status_message_t, ClassifierClassic> make_classifier_classic(std::string const & json_params);


private:
    ClassifierState m_state;

    ClassifierClassic(params_t const & params);
    ClassifierClassic(params_t && params);
};

Either<status_message_t, ClassifierClassic> make_classifier_classic(std::string const & json_params = "{}");


} // namespace Tsetlini
