#pragma once

#ifndef LIB_INCLUDE_TSETLINI_HPP_
#define LIB_INCLUDE_TSETLINI_HPP_

#include "tsetlini_params.hpp"
#include "tsetlini_status_code.hpp"

#include "tsetlini_types.hpp"
#include "estimator_state_fwd.hpp"
#include "tsetlini_strong_params.hpp"
#include "either.hpp"

#include <vector>


#define TSETLINI_VERSION_MAJOR 0
#define TSETLINI_VERSION_MINOR 0
#define TSETLINI_VERSION_PATCH 9
// TSETLINI_VERSION : 0xMMmmPP
#define TSETLINI_VERSION (TSETLINI_VERSION_MAJOR * 0x10000 + \
                          TSETLINI_VERSION_MINOR *   0x100 + \
                          TSETLINI_VERSION_PATCH *     0x1)


namespace Tsetlini
{


struct ClassifierClassic
{
    [[nodiscard]]
    status_message_t
    fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs = number_of_epochs_t{100});

    [[nodiscard]]
    status_message_t
    partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs = number_of_epochs_t{100});

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
    decision_function(aligned_vector_char const & sample) const;

    [[nodiscard]]
    Either<status_message_t, std::vector<aligned_vector_int>>
    decision_function(std::vector<aligned_vector_char> const & X) const;


    params_t read_params() const;
    ClassifierStateClassicPtr clone_state() const;


    ClassifierClassic(ClassifierStateClassic const & state);
    ClassifierClassic(ClassifierClassic &&);
    ClassifierClassic & operator=(ClassifierClassic &&) = default;

friend Either<status_message_t, ClassifierClassic> make_classifier_classic(std::string const & json_params);


private:
    ClassifierStateClassicPtr m_state_p;

    ClassifierClassic(params_t const & params);
    ClassifierClassic(params_t && params);
};

Either<status_message_t, ClassifierClassic> make_classifier_classic(std::string const & json_params = "{}");


////////////////////////////////////////////////////////////////////////////////

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
        number_of_epochs_t epochs = number_of_epochs_t{100});

    [[nodiscard]]
    status_message_t
    partial_fit(std::vector<aligned_vector_char> const & X, response_vector_type const & y,
        number_of_epochs_t epochs = number_of_epochs_t{100});

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
    RegressorStateClassicPtr clone_state() const;

    RegressorClassic(RegressorStateClassic const & state);
    RegressorClassic(RegressorClassic &&);
    RegressorClassic & operator=(RegressorClassic &&) = default;

friend Either<status_message_t, RegressorClassic> make_regressor_classic(std::string const & json_params);


private:
    RegressorStateClassicPtr m_state_p;

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
 * weights - boolean flag whether to employ integer weighting of clauses
 * max_weight - integer upper bound on individual clause weights, default=max(int)
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


////////////////////////////////////////////////////////////////////////////////

struct ClassifierBitwise
{
    [[nodiscard]]
    status_message_t
    fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs = number_of_epochs_t{100});

    [[nodiscard]]
    status_message_t
    partial_fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs = number_of_epochs_t{100});

    [[nodiscard]]
    Either<status_message_t, real_type>
    evaluate(std::vector<bit_vector_uint64> const & X, label_vector_type const & y) const;

    [[nodiscard]]
    Either<status_message_t, label_type>
    predict(bit_vector_uint64 const & sample) const;

    [[nodiscard]]
    Either<status_message_t, label_vector_type>
    predict(std::vector<bit_vector_uint64> const & X) const;

    [[nodiscard]]
    Either<status_message_t, aligned_vector_int>
    decision_function(bit_vector_uint64 const & sample) const;

    [[nodiscard]]
    Either<status_message_t, std::vector<aligned_vector_int>>
    decision_function(std::vector<bit_vector_uint64> const & X) const;


    params_t read_params() const;
    ClassifierStateBitwisePtr clone_state() const;

    ClassifierBitwise(ClassifierStateBitwise const & state);
    ClassifierBitwise(ClassifierBitwise &&);
    ClassifierBitwise & operator=(ClassifierBitwise &&) = default;

friend Either<status_message_t, ClassifierBitwise> make_classifier_bitwise(std::string const & json_params);


private:
    ClassifierStateBitwisePtr m_state_p;

    ClassifierBitwise(params_t const & params);
    ClassifierBitwise(params_t && params);
};


Either<status_message_t, ClassifierBitwise> make_classifier_bitwise(std::string const & json_params = "{}");


////////////////////////////////////////////////////////////////////////////////

struct RegressorBitwise
{
    /*
     * TODO
     * X is a bit vector composed of 64-bit unsigned integers,
     * y is a vector of integers. Values of y must fall within [0, T] range.
     *   It is a responsibility of the user to normalize y. It is not mandatory
     *   that y spans entire [0, T] range, as that may vary while training
     *   online, especially if input training data evolves over time.
     * epochs specifies number of epochs to iterate training
     */
    [[nodiscard]]
    status_message_t
    fit(std::vector<bit_vector_uint64> const & X, response_vector_type const & y,
        number_of_epochs_t epochs = number_of_epochs_t{100});

    [[nodiscard]]
    status_message_t
    partial_fit(std::vector<bit_vector_uint64> const & X, response_vector_type const & y,
        number_of_epochs_t epochs = number_of_epochs_t{100});

//    [[nodiscard]]
//    Either<status_message_t, real_type>
//    evaluate(std::vector<aligned_vector_char> const & X, label_vector_type const & y) const;

    [[nodiscard]]
    Either<status_message_t, response_type>
    predict(bit_vector_uint64 const & sample) const;

    [[nodiscard]]
    Either<status_message_t, response_vector_type>
    predict(std::vector<bit_vector_uint64> const & X) const;


    params_t read_params() const;
    RegressorStateBitwisePtr clone_state() const;

    RegressorBitwise(RegressorStateBitwise const & state);
    RegressorBitwise(RegressorBitwise &&);
    RegressorBitwise & operator=(RegressorBitwise &&) = default;

friend Either<status_message_t, RegressorBitwise> make_regressor_bitwise(std::string const & json_params);


private:
    RegressorStateBitwisePtr m_state_p;

    RegressorBitwise(params_t const & params);
    RegressorBitwise(params_t && params);
};

/*
 * Accepted parameters:
 *
 * number_of_regressor_clauses - total number of clauses to use for training
 * number of states - number of states [-N, +N] that a single TA can attain
 * s - specificity parameter
 * threshold - threshold for total count of votes cast by clauses
 * weights - boolean flag whether to employ integer weighting of clauses
 * max_weight - integer upper bound on individual clause weights, default=max(int)
 * boost_true_positive_feedback - controls learning // TODO
 * n_jobs - number of paraller threads
 * verbose - verbosity flag
 * counting_type - underlying signed integral type used to represent single TA state, 'auto' means smallest width type
 * clause_output_tile_size - positive integer which describes how many clause outputs are processed as a single tile
 * random_state - seed value for PRN generation
 *
 * number_of_features - [internal] number of features used for initial fit. Used for for X verification in partial fit.
 */
Either<status_message_t, RegressorBitwise> make_regressor_bitwise(std::string const & json_params = "{}");


} // namespace Tsetlini


#endif /* LIB_INCLUDE_TSETLINI_HPP_ */
