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


struct RegressorClassic
{
    /*
     * X is a vector of vectors of chars, each vector can hold either 0 or 1,
     * y is a vector of integers. Values of y must fall within [0, T] range.
     *   It is a responsibility of the user to normalize y. It is not mandatory
     *   that y spans entire [0, T] range, as that may vary while training
     *   online, especially if input training data evolves over time.
     * epochs specifies number of epochs to iterate training
     */
    [[nodiscard]]
    status_message_t
    fit(std::vector<aligned_vector_char> const & X, response_vector_type const & y,
        unsigned int epochs = 100);

    [[nodiscard]]
    status_message_t
    partial_fit(std::vector<aligned_vector_char> const & X, response_vector_type const & y,
        unsigned int epochs = 100);

//    [[nodiscard]]
//    Either<status_message_t, real_type>
//    evaluate(std::vector<aligned_vector_char> const & X, label_vector_type const & y) const;

    [[nodiscard]]
    Either<status_message_t, response_type>
    predict(aligned_vector_char const & sample) const;

    [[nodiscard]]
    Either<status_message_t, response_vector_type>
    predict(std::vector<aligned_vector_char> const & X) const;


    params_t read_params() const;
    RegressorState read_state() const;

    RegressorClassic(RegressorState const & state);

friend Either<status_message_t, RegressorClassic> make_regressor_classic(std::string const & json_params);


private:
    RegressorState m_state;

    RegressorClassic(params_t const & params);
    RegressorClassic(params_t && params);
};

/*
 * Accepted parameters:
 *
 * number_of_regressor_clauses - total number of clauses to use for training
 * number of states - number of states [-N, +N] that a single TA can attain
 * s - specificity parameter
 * threshold - threshold for total count of votes cast by clauses
 * boost_true_positive_feedback - controls learning // TODO
 * n_jobs - number of paraller threads
 * verbose - verbosity flag
 * counting_type - underlying signed integral type used to represent single TA state, 'auto' means smallest width type
 * clause_output_tile_size - positive integer which describes how many clause outputs are processed as a single tile
 * random_state - seed value for PRN generation
 *
 * number_of_features - [internal] number of features used for initial fit. Used for for X verification in partial fit.
 */
Either<status_message_t, RegressorClassic> make_regressor_classic(std::string const & json_params = "{}");


} // namespace Tsetlini
