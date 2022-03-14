#pragma once

#ifndef LIB_INCLUDE_TSETLINI_HPP_
#define LIB_INCLUDE_TSETLINI_HPP_

#include "tsetlini_params.hpp"
#include "tsetlini_status_code.hpp"

#include "tsetlini_types.hpp"
#include "estimator_state_fwd.hpp"
#include "tsetlini_strong_params.hpp"
#include "arg_extract.hpp"
#include "either.hpp"
#include "is_subset_of.hpp"

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
    /*
     * Fit the model according to the given training data.
     *
     * X: matrix of shape (n_samples, n_features)
     *    Training vector, where n_samples is the number of samples and
     *    n_features is the number of features. All values must be either
     *    0 or 1.
     *
     * y: array of shape (n_samples,)
     *    Target vector relative to X.
     *
     * max_number_of_labels: integer value of type max_number_of_labels_t
     *    Number of labels to prepare the model for. If, however, y contains
     *    more labels, the value will be adjusted accordingly to accommodate
     *    all of them.
     *
     * number_of_epochs: unsigned integer value of type max_number_of_labels_t
     *    Number of epochs to train the model for. If not provided, the default
     *    value of 100 is used.
     *
     * Arguments except X and y can be passed in arbitrary order.
     */
    template<typename ...Args>
    [[nodiscard]]
    inline
    status_message_t
    fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, Args && ...args);

    /*
     * Update the model over the given data.
     *
     * X: matrix of shape (n_samples, n_features)
     *    Training vector, where n_samples is the number of samples and
     *    n_features is the number of features. All values must be either
     *    0 or 1.
     *
     * y: array of shape (n_samples,)
     *    Target vector relative to X.
     *
     * max_number_of_labels: integer value of type max_number_of_labels_t
     *    Number of labels to prepare the model for in case partial_fit was
     *    called on an untrained model. If, however, y contains more labels,
     *    the value will be adjusted accordingly to accommodate all of them.
     *    This parameter has no meaning if the model has been already trained.
     *
     * number_of_epochs: unsigned integer value of type max_number_of_labels_t
     *    Number of epochs to train the model for. If not provided, the default
     *    value of 100 is used.
     *
     * Arguments except X and y can be passed in arbitrary order.
     */
    template<typename ...Args>
    [[nodiscard]]
    inline
    status_message_t
    partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, Args && ...args);

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

    friend Either<status_message_t, ClassifierClassic> make_classifier_classic_from_json(std::string const & json_params);

    template<typename ...Args>
    friend Either<status_message_t, ClassifierClassic> make_classifier_classic(Args && ...args);


private:
    ClassifierStateClassicPtr m_state_p;

    ClassifierClassic(params_t const & params);
    ClassifierClassic(params_t && params);

    status_message_t
    _fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs);

    status_message_t
    _partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs);

};


Either<status_message_t, ClassifierClassic> make_classifier_classic_from_json(std::string const & json_params = "{}");


template<typename ...Args>
Either<status_message_t, ClassifierClassic> make_classifier_classic(Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            number_of_physical_classifier_clauses_per_label_t,
            number_of_states_t,
            specificity_t,
            threshold_t,
            weighted_flag_t,
            max_weight_t,
            boost_tpf_t,
            number_of_jobs_t,
            verbosity_t,
            counting_type_t,
            clause_output_tile_size_t,
            random_seed_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return
        make_classifier_params_from_args(
            arg::extract_or<number_of_physical_classifier_clauses_per_label_t>(number_of_physical_classifier_clauses_per_label_t{12}, args...)
            , arg::extract_or<number_of_states_t>(number_of_states_t{100}, args...)
            , arg::extract_or<specificity_t>(specificity_t{2.0f}, args...)
            , arg::extract_or<threshold_t>(threshold_t{15}, args...)
            , arg::extract_or<weighted_flag_t>(weighted_flag_t{false}, args...)
            , arg::extract_or<max_weight_t>(MAX_WEIGHT_DEFAULT, args...)
            , arg::extract_or<boost_tpf_t>(boost_tpf_t{false}, args...)
            , arg::extract_or<number_of_jobs_t>(number_of_jobs_t{-1}, args...)
            , arg::extract_or<verbosity_t>(verbosity_t{false}, args...)
            , arg::extract_or<counting_type_t>(counting_type_t{std::string("auto")}, args...)
            , arg::extract_or<clause_output_tile_size_t>(clause_output_tile_size_t{16}, args...)
            , arg::maybe_extract<random_seed_t>(args...)
        )
        .rightMap([](params_t && params){ return ClassifierClassic(params); })
        ;
}


template<typename ...Args>
status_message_t
ClassifierClassic::fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            max_number_of_labels_t,
            number_of_epochs_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return _fit(X, y, arg::extract<max_number_of_labels_t>(args...), arg::extract_or<number_of_epochs_t>(number_of_epochs_t{100}, args...));
}


template<typename ...Args>
status_message_t
ClassifierClassic::partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            max_number_of_labels_t,
            number_of_epochs_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return _partial_fit(X, y, arg::extract<max_number_of_labels_t>(args...), arg::extract_or<number_of_epochs_t>(number_of_epochs_t{100}, args...));
}


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

    friend Either<status_message_t, RegressorClassic> make_regressor_classic_from_json(std::string const & json_params);

    template<typename ...Args>
    friend Either<status_message_t, RegressorClassic> make_regressor_classic(Args && ...args);


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
Either<status_message_t, RegressorClassic> make_regressor_classic_from_json(std::string const & json_params = "{}");


template<typename ...Args>
Either<status_message_t, RegressorClassic> make_regressor_classic(Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            number_of_physical_regressor_clauses_t,
            number_of_states_t,
            specificity_t,
            threshold_t,
            weighted_flag_t,
            max_weight_t,
            boost_tpf_t,
            number_of_jobs_t,
            verbosity_t,
            counting_type_t,
            clause_output_tile_size_t,
            loss_fn_name_t,
            loss_fn_C1_t,
            box_muller_flag_t,
            random_seed_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return
        make_regressor_params_from_args(
            arg::extract_or<number_of_physical_regressor_clauses_t>(number_of_physical_regressor_clauses_t{20}, args...)
            , arg::extract_or<number_of_states_t>(number_of_states_t{100}, args...)
            , arg::extract_or<specificity_t>(specificity_t{2.0f}, args...)
            , arg::extract_or<threshold_t>(threshold_t{15}, args...)
            , arg::extract_or<weighted_flag_t>(weighted_flag_t{true}, args...)
            , arg::extract_or<max_weight_t>(MAX_WEIGHT_DEFAULT, args...)
            , arg::extract_or<boost_tpf_t>(boost_tpf_t{false}, args...)
            , arg::extract_or<number_of_jobs_t>(number_of_jobs_t{-1}, args...)
            , arg::extract_or<verbosity_t>(verbosity_t{false}, args...)
            , arg::extract_or<counting_type_t>(counting_type_t{std::string("auto")}, args...)
            , arg::extract_or<clause_output_tile_size_t>(clause_output_tile_size_t{16}, args...)
            , arg::extract_or<loss_fn_name_t>(loss_fn_name_t{"MSE"}, args...)
            , arg::extract_or<loss_fn_C1_t>(loss_fn_C1_t{0.0f}, args...)
            , arg::extract_or<box_muller_flag_t>(box_muller_flag_t{false}, args...)
            , arg::maybe_extract<random_seed_t>(args...)
        )
        .rightMap([](params_t && params){ return RegressorClassic(params); })
        ;
}


////////////////////////////////////////////////////////////////////////////////

struct ClassifierBitwise
{
    /*
     * Fit the model according to the given training data.
     *
     * X: bit matrix of shape (n_samples, n_features)
     *    Training vector, where n_samples is the number of samples and
     *    n_features is the number of features.
     *
     * y: array of shape (n_samples,)
     *    Target vector relative to X.
     *
     * max_number_of_labels: integer value of type max_number_of_labels_t
     *    Number of labels to prepare the model for. If, however, y contains
     *    more labels, the value will be adjusted accordingly to accommodate
     *    all of them.
     *
     * number_of_epochs: unsigned integer value of type max_number_of_labels_t
     *    Number of epochs to train the model for. If not provided, the default
     *    value of 100 is used.
     *
     * Arguments except X and y can be passed in arbitrary order.
     */
    template<typename ...Args>
    [[nodiscard]]
    inline
    status_message_t
    fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y, Args && ...args);

    /*
     * Update the model over the given data.
     *
     * X: bit matrix of shape (n_samples, n_features)
     *    Training vector, where n_samples is the number of samples and
     *    n_features is the number of features.
     *
     * y: array of shape (n_samples,)
     *    Target vector relative to X.
     *
     * max_number_of_labels: integer value of type max_number_of_labels_t
     *    Number of labels to prepare the model for in case partial_fit was
     *    called on an untrained model. If, however, y contains more labels,
     *    the value will be adjusted accordingly to accommodate all of them.
     *    This parameter has no meaning if the model has been already trained.
     *
     * number_of_epochs: unsigned integer value of type max_number_of_labels_t
     *    Number of epochs to train the model for. If not provided, the default
     *    value of 100 is used.
     *
     * Arguments except X and y can be passed in arbitrary order.
     */
    template<typename ...Args>
    [[nodiscard]]
    inline
    status_message_t
    partial_fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y, Args && ...args);

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

    friend Either<status_message_t, ClassifierBitwise> make_classifier_bitwise_from_json(std::string const & json_params);

    template<typename ...Args>
    friend Either<status_message_t, ClassifierBitwise> make_classifier_bitwise(Args && ...args);


private:
    ClassifierStateBitwisePtr m_state_p;

    ClassifierBitwise(params_t const & params);
    ClassifierBitwise(params_t && params);

    status_message_t
    _fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs);

    status_message_t
    _partial_fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y,
        max_number_of_labels_t max_number_of_labels, number_of_epochs_t epochs);
};


Either<status_message_t, ClassifierBitwise> make_classifier_bitwise_from_json(std::string const & json_params = "{}");


template<typename ...Args>
Either<status_message_t, ClassifierBitwise> make_classifier_bitwise(Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            number_of_physical_classifier_clauses_per_label_t,
            number_of_states_t,
            specificity_t,
            threshold_t,
            weighted_flag_t,
            max_weight_t,
            boost_tpf_t,
            number_of_jobs_t,
            verbosity_t,
            counting_type_t,
            clause_output_tile_size_t,
            random_seed_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return
        make_classifier_params_from_args(
            arg::extract_or<number_of_physical_classifier_clauses_per_label_t>(number_of_physical_classifier_clauses_per_label_t{12}, args...)
            , arg::extract_or<number_of_states_t>(number_of_states_t{100}, args...)
            , arg::extract_or<specificity_t>(specificity_t{2.0f}, args...)
            , arg::extract_or<threshold_t>(threshold_t{15}, args...)
            , arg::extract_or<weighted_flag_t>(weighted_flag_t{false}, args...)
            , arg::extract_or<max_weight_t>(MAX_WEIGHT_DEFAULT, args...)
            , arg::extract_or<boost_tpf_t>(boost_tpf_t{false}, args...)
            , arg::extract_or<number_of_jobs_t>(number_of_jobs_t{-1}, args...)
            , arg::extract_or<verbosity_t>(verbosity_t{false}, args...)
            , arg::extract_or<counting_type_t>(counting_type_t{std::string("auto")}, args...)
            , arg::extract_or<clause_output_tile_size_t>(clause_output_tile_size_t{16}, args...)
            , arg::maybe_extract<random_seed_t>(args...)
        )
        .rightMap([](params_t && params){ return ClassifierBitwise(params); })
        ;
}


template<typename ...Args>
status_message_t
ClassifierBitwise::fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y, Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            max_number_of_labels_t,
            number_of_epochs_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return _fit(X, y, arg::extract<max_number_of_labels_t>(args...), arg::extract_or<number_of_epochs_t>(number_of_epochs_t{100}, args...));
}


template<typename ...Args>
status_message_t
ClassifierBitwise::partial_fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y, Args && ...args)
{
    static_assert(meta::is_subset_of<
        std::tuple<
            max_number_of_labels_t,
            number_of_epochs_t>,
        std::decay_t<Args>...>,
        "Passed argument of type outside of accepted type set");

    return _partial_fit(X, y, arg::extract<max_number_of_labels_t>(args...), arg::extract_or<number_of_epochs_t>(number_of_epochs_t{100}, args...));
}


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
