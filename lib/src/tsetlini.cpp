#define LOG_MODULE "tsetlini"
#include "logger.hpp"

#include "estimator_state_cache.hpp"
#include "tsetlini_private.hpp"
#include "tsetlini_algo_classic.hpp"
#include "tsetlini_algo_common.hpp"

#include "tsetlini_params.hpp"
#include "tsetlini.hpp"
#include "mt.hpp"
#include "assume_aligned.hpp"
#include "params_companion.hpp"
#include "tsetlini_types.hpp"
#include "estimator_state.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_estimator_state_private.hpp"

#include "neither/either.hpp"

#include <utility>
#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <string>
#include <numeric>
#include <cstddef>


using namespace neither;
using namespace std::string_literals;


namespace Tsetlini
{

namespace
{


template<typename T>
bool check_all_0s_or_1s(aligned_vector<T> const & vec)
{
    return std::all_of(vec.cbegin(), vec.cend(), [](T v){ return v == 0 || v == 1; });
}


template<typename T>
bool check_all_0s_or_1s(bit_vector<T> const & vec)
{
    return true;
}


template<typename RowType, typename T>
status_message_t check_X_y(
    std::vector<RowType> const & X,
    std::vector<T> const & y)
{
    if (X.empty())
    {
        return {StatusCode::S_VALUE_ERROR, "X cannot be empty"};
    }
    else if (y.empty())
    {
        return {StatusCode::S_VALUE_ERROR, "y cannot be empty"};
    }
    else if (y.size() != X.size())
    {
        return {StatusCode::S_VALUE_ERROR,
            "X and y must have the same lengths, got " + std::to_string(X.size()) +
            " and " + std::to_string(y.size())};
    }
    else if (auto first = X.cbegin();
        not std::all_of(std::next(first), X.cend(),
            [&first](auto const & row){ return row.size() == first->size(); }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "All rows of X must have the same number of feature columns"};
    }
    else if (not std::all_of(X.cbegin(), X.cend(), [](auto const & row){ return check_all_0s_or_1s(row); }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values of 0 and 1 can be used in X"};
    }

    return {StatusCode::S_OK, ""};
}


status_message_t check_response_y(response_vector_type const & y, int const T)
{
    if (not std::all_of(y.cbegin(), y.cend(), [T](auto v){ return v >= 0 and v <= T; }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values within [0, threshold] range can be used in y"};
    }

    return {StatusCode::S_OK, ""};
}


status_message_t check_labels(label_vector_type const & labels)
{
    if (*std::min_element(labels.cbegin(), labels.cend()) < 0)
    {
        return {StatusCode::S_VALUE_ERROR, "Labels in y cannot be negative"};
    }

    return {StatusCode::S_OK, ""};
}


status_message_t check_labels(label_vector_type const & labels, int max_label)
{
    auto [lo, hi] = std::minmax_element(labels.cbegin(), labels.cend());

    if (*lo < 0)
    {
        return {StatusCode::S_VALUE_ERROR, "Labels in y cannot be negative"};
    }
    else if (*hi > max_label)
    {
        return {StatusCode::S_VALUE_ERROR,
            "Max allowed label is " + std::to_string(max_label) +
            " but y has value as high as " + std::to_string(*hi)};
    }

    return {StatusCode::S_OK, ""};
}


label_vector_type unique_labels(label_vector_type const & y)
{
    std::unordered_set<label_type> uniq(y.cbegin(), y.cend());

    return label_vector_type(uniq.cbegin(), uniq.cend());
}


bool is_fitted(TAState::value_type const & ta_state)
{
    return std::visit([](auto const & ta_state_values){ return ta_state_values.shape().first != 0; }, ta_state.matrix);
}


bool is_fitted(TAStateWithSignum::value_type const & ta_state)
{
    return std::visit([](auto const & ta_state_values){ return ta_state_values.shape().first != 0; }, ta_state.matrix);
}


template<typename EstimatorStateType, typename SampleType>
status_message_t check_for_predict(
    EstimatorStateType const & state,
    std::vector<SampleType> const & X)
{
    if (not is_fitted(state.ta_state))
    {
        return {StatusCode::S_NOT_FITTED_ERROR,
            "This model instance is not fitted yet. Call fit or partial_fit before using this method"};
    }
    else if (X.size() == 0)
    {
        return {StatusCode::S_VALUE_ERROR, "Cannot predict for empty X"};
    }
    else if (auto first = X.cbegin();
        not std::all_of(std::next(first), X.cend(),
            [&first](auto const & sample){ return sample.size() == first->size(); }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "All samples in X must have the same number of feature columns"};
    }
    else if (X.front().size() - Params::number_of_features(state.m_params) != 0)
    {
        return {StatusCode::S_VALUE_ERROR,
            "Predict called with X, which number of features " + std::to_string(X.front().size()) +
            " does not match that from prior fit " + std::to_string(Params::number_of_features(state.m_params))};
    }
    else if (not std::all_of(X.cbegin(), X.cend(), [](auto const & sample){ return check_all_0s_or_1s(sample); }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values of 0 and 1 can be used in X"};
    }

    return {StatusCode::S_OK, ""};
}


template<typename EstimatorStateType, typename SampleType>
status_message_t check_for_predict(
    EstimatorStateType const & state,
    SampleType const & sample)
{
    if (not is_fitted(state.ta_state))
    {
        return {StatusCode::S_NOT_FITTED_ERROR,
            "This model instance is not fitted yet. Call fit or partial_fit before using this method"};
    }
    else if (sample.size() - Params::number_of_features(state.m_params) != 0)
    {
        return {StatusCode::S_VALUE_ERROR,
            "Predict called with sample, which size " + std::to_string(sample.size()) +
            " does not match number of features from prior fit " + std::to_string(Params::number_of_features(state.m_params))};
    }
    else if (not check_all_0s_or_1s(sample))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values of 0 and 1 can be used in sample for prediction"};
    }

    return {StatusCode::S_OK, ""};
}


template<typename SampleType, typename TAStateValueType>
void classifier_update_impl(
    SampleType const & X,
    label_type const target_label,
    label_type const opposite_label,

    int const number_of_pos_neg_clauses_per_label,
    int const threshold,
    int const number_of_clauses,
    int const number_of_states,
    real_type s,
    int const boost_true_positive_feedback,
    int const n_jobs,

    IRNG & igen,
    FRNG & fgen,
    TAStateValueType & ta_state,
    ClassifierStateCache::value_type & cache,

    int clause_output_tile_size
    )
{
    {
        auto const [output_ix_begin, output_ix_end] = clause_range_for_label(target_label, number_of_pos_neg_clauses_per_label);

        calculate_clause_output(
            X,
            cache.clause_output,
            output_ix_begin,
            output_ix_end,
            ta_state,
            n_jobs,
            clause_output_tile_size
        );
    }

    {
        auto const [output_ix_begin, output_ix_end] = clause_range_for_label(opposite_label, number_of_pos_neg_clauses_per_label);

        calculate_clause_output(
            X,
            cache.clause_output,
            output_ix_begin,
            output_ix_end,
            ta_state,
            n_jobs,
            clause_output_tile_size
        );
    }

    sum_up_label_votes(
        cache.clause_output,
        cache.label_sum,
        target_label,
        number_of_pos_neg_clauses_per_label,
        threshold);

    sum_up_label_votes(
        cache.clause_output,
        cache.label_sum,
        opposite_label,
        number_of_pos_neg_clauses_per_label,
        threshold);



    calculate_classifier_feedback_to_clauses(
        cache.feedback_to_clauses,
        target_label,
        opposite_label,
        cache.label_sum[target_label],
        cache.label_sum[opposite_label],
        number_of_pos_neg_clauses_per_label,
        threshold,
        fgen);

    {
        auto const [input_ix_begin, input_ix_end] = clause_range_for_label(target_label, number_of_pos_neg_clauses_per_label);

        train_classifier_automata(
            ta_state,
            input_ix_begin,
            input_ix_end,
            cache.feedback_to_clauses.data(),
            cache.clause_output.data(),
            number_of_states,
            X,
            boost_true_positive_feedback,
            igen,
            cache.ct
        );
    }

    {
        auto const [input_ix_begin, input_ix_end] = clause_range_for_label(opposite_label, number_of_pos_neg_clauses_per_label);

        train_classifier_automata(
            ta_state,
            input_ix_begin,
            input_ix_end,
            cache.feedback_to_clauses.data(),
            cache.clause_output.data(),
            number_of_states,
            X,
            boost_true_positive_feedback,
            igen,
            cache.ct
        );
    }
}


template<typename ClassifierStateType, typename SampleType>
Either<status_message_t, real_type>
evaluate_classifier_impl(
    ClassifierStateType const & state,
    std::vector<SampleType> const & X,
    label_vector_type const & y)
{
    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_classifier_clauses(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    int errors = 0;

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output_for_predict(
            X[it],
            state.cache.clause_output,
            number_of_clauses / 2,
            state.ta_state,
            n_jobs,
            clause_output_tile_size);

        sum_up_all_label_votes(
            state.cache.clause_output,
            state.cache.label_sum,
            number_of_labels,
            number_of_pos_neg_clauses_per_label,
            threshold);


        const int max_class = std::distance(
            state.cache.label_sum.cbegin(),
            std::max_element(state.cache.label_sum.cbegin(), state.cache.label_sum.cend()));

        errors += (max_class != y[it]);
    }

    real_type const rv = ONE - ONE * errors / number_of_examples;

    return Either<status_message_t, real_type>::rightOf(rv);
}


template<typename ClassifierStateType, typename SampleType>
Either<status_message_t, label_type>
predict_classifier_impl(ClassifierStateType const & state, SampleType const & sample)
{
    if (auto sm = check_for_predict(state, sample);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, label_type>::leftOf(std::move(sm));
    }

    auto const n_jobs = Params::n_jobs(state.m_params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(state.m_params);

    calculate_clause_output_for_predict(
        sample,
        state.cache.clause_output,
        Params::number_of_classifier_clauses(state.m_params) / 2,
        state.ta_state,
        n_jobs,
        clause_output_tile_size);

    sum_up_all_label_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        Params::number_of_labels(state.m_params),
        Params::number_of_pos_neg_clauses_per_label(state.m_params),
        Params::threshold(state.m_params));

    label_type rv = std::distance(
        state.cache.label_sum.cbegin(),
        std::max_element(state.cache.label_sum.cbegin(), state.cache.label_sum.cend()));

    return Either<status_message_t, label_type>::rightOf(rv);
}


template<typename RegressorStateType, typename SampleType>
Either<status_message_t, response_type>
predict_regressor_impl(RegressorStateType const & state, SampleType const & sample)
{
    if (auto sm = check_for_predict(state, sample);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, label_type>::leftOf(std::move(sm));
    }

    auto const n_jobs = Params::n_jobs(state.m_params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(state.m_params);

    calculate_clause_output_for_predict(
        sample,
        state.cache.clause_output,
        Params::number_of_classifier_clauses(state.m_params) / 2,
        state.ta_state,
        n_jobs,
        clause_output_tile_size);

    response_type rv = sum_up_regressor_votes(state.cache.clause_output, Params::threshold(state.m_params), state.ta_state.weights);

    return Either<status_message_t, label_type>::rightOf(rv);
}


template<typename ClassifierStateType, typename SampleType>
Either<status_message_t, label_vector_type>
predict_classifier_impl(ClassifierStateType const & state, std::vector<SampleType> const & X)
{
    if (auto sm = check_for_predict(state, X);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, label_vector_type>::leftOf(std::move(sm));
    }

    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_classifier_clauses(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    label_vector_type rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output_for_predict(
            X[it],
            state.cache.clause_output,
            number_of_clauses / 2,
            state.ta_state,
            n_jobs,
            clause_output_tile_size);

        sum_up_all_label_votes(
            state.cache.clause_output,
            state.cache.label_sum,
            number_of_labels,
            number_of_pos_neg_clauses_per_label,
            threshold);


        const int max_class = std::distance(
            state.cache.label_sum.cbegin(),
            std::max_element(state.cache.label_sum.cbegin(), state.cache.label_sum.cend()));

        rv[it] = max_class;
    }

    return Either<status_message_t, label_vector_type>::rightOf(rv);
}


template<typename ClassifierStateType, typename SampleType>
Either<status_message_t, aligned_vector_int>
predict_classifier_raw_impl(ClassifierStateType const & state, SampleType const & sample)
{
    if (auto sm = check_for_predict(state, sample);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, aligned_vector_int>::leftOf(std::move(sm));
    }

    // let it crash - no state validation for now

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_classifier_clauses(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);


    calculate_clause_output_for_predict(
        sample,
        state.cache.clause_output,
        number_of_clauses / 2,
        state.ta_state,
        n_jobs,
        clause_output_tile_size);

    sum_up_all_label_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        number_of_labels,
        number_of_pos_neg_clauses_per_label,
        threshold);

    return Either<status_message_t, aligned_vector_int>::rightOf(state.cache.label_sum);
}


template<typename ClassifierStateType, typename SampleType>
Either<status_message_t, std::vector<aligned_vector_int>>
predict_classifier_raw_impl(ClassifierStateType const & state, std::vector<SampleType> const & X)
{
    if (auto sm = check_for_predict(state, X);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, std::vector<aligned_vector_int>>::leftOf(std::move(sm));
    }

    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_classifier_clauses(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    std::vector<aligned_vector_int> rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output_for_predict(
            X[it],
            state.cache.clause_output,
            number_of_clauses / 2,
            state.ta_state,
            n_jobs,
            clause_output_tile_size);

        sum_up_all_label_votes(
            state.cache.clause_output,
            state.cache.label_sum,
            number_of_labels,
            number_of_pos_neg_clauses_per_label,
            threshold);

        rv[it] = state.cache.label_sum;
    }

    return Either<status_message_t, std::vector<aligned_vector_int>>::rightOf(rv);
}


template<typename Gen>
void generate_opposite_y(
    label_vector_type const & y,
    label_vector_type & opposite_y,
    int number_of_labels,
    Gen & g)
{
    for (auto it = 0u; it < y.size(); ++it)
    {
        opposite_y[it] = (y[it] + 1 + g() % (number_of_labels - 1)) % number_of_labels;
    }
}


template<typename ClassifierStateType, typename TAStateValueType, typename SampleType>
status_message_t
fit_classifier_online_impl(
    ClassifierStateType & state,
    TAStateValueType & ta_state,
    std::vector<SampleType> const & X,
    label_vector_type const & y,
    unsigned int epochs)
{
    auto labels = unique_labels(y);

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_classifier_clauses(params);
    auto const number_of_states = Params::number_of_states(params);
    auto const s = Params::s(params);
    auto const boost_true_positive_feedback = Params::boost_true_positive_feedback(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const verbose = Params::verbose(params);

    if (auto sm = check_labels(labels, number_of_labels);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    auto const number_of_examples = X.size();

    std::vector<int> ix(number_of_examples);

    label_vector_type opposite_y(y.size());

    for (unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        LOG(info) << "Epoch " << epoch + 1 << '\n';

        generate_opposite_y(y, opposite_y, number_of_labels, state.igen);

        std::iota(ix.begin(), ix.end(), 0);
        std::shuffle(ix.begin(), ix.end(), state.igen);

        state.cache.ct.populate(s, state.igen);

        for (auto i = 0u; i < number_of_examples; ++i)
        {
            classifier_update_impl(
                X[ix[i]],
                y[ix[i]],
                opposite_y[ix[i]],

                number_of_pos_neg_clauses_per_label,
                threshold,
                number_of_clauses,
                number_of_states,
                s,
                boost_true_positive_feedback,
                n_jobs,

                state.igen,
                state.fgen,
                ta_state,
                state.cache,

                clause_output_tile_size
            );
        }
    }

    return {S_OK, ""};
}


template<typename ClassifierStateType, typename SampleType>
status_message_t
fit_classifier_impl_T(
    ClassifierStateType & state,
    std::vector<SampleType> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    auto labels = unique_labels(y);

    int const number_of_labels = std::max(
        *std::max_element(labels.cbegin(), labels.cend()) + 1,
        max_number_of_labels);

    if (auto sm = check_labels(labels);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    // Let it crash for now
//    validate_params();

    state.m_params["number_of_labels"] = param_value_t(number_of_labels);

    int const number_of_features = X.front().size();
    state.m_params["number_of_features"] = param_value_t(number_of_features);

    initialize_state(state);

    return fit_classifier_online_impl(state, state.ta_state, X, y, epochs);
}


template<typename ClassifierStateType, typename SampleType>
status_message_t
fit_classifier_impl(
    ClassifierStateType & state,
    std::vector<SampleType> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    if (auto sm = check_X_y(X, y);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    return fit_classifier_impl_T(state, X, y, max_number_of_labels, epochs);
}


template<typename SampleType, typename TAStateValueType>
void regressor_update_impl(
    SampleType const & X,
    response_type const target_response,

    int const threshold,
    int const number_of_clauses,
    int const number_of_states,
    real_type s,
    int const boost_true_positive_feedback,
    int const n_jobs,

    IRNG & igen,
    FRNG & fgen,
    TAStateValueType & ta_state,
    RegressorStateCache::value_type & cache,

    int clause_output_tile_size
    )
{
    calculate_clause_output(
        X,
        cache.clause_output,
        0,
        number_of_clauses / 2,
        ta_state,
        n_jobs,
        clause_output_tile_size
    );

    auto const votes = sum_up_regressor_votes(cache.clause_output, threshold, ta_state.weights);
    int const response_error = votes - target_response;

    train_regressor_automata(
        ta_state,
        0,
        number_of_clauses / 2,
        cache.clause_output.data(),
        number_of_states,
        response_error,
        X,
        boost_true_positive_feedback,
        igen,
        threshold,
        cache.ct
    );
}


template<typename RegressorStateType, typename TAStateValueType, typename SampleType>
status_message_t
fit_regressor_online_impl(
    RegressorStateType & state,
    TAStateValueType & ta_state,
    std::vector<SampleType> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    auto const & params = state.m_params;

    auto const number_of_clauses = Params::number_of_regressor_clauses(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_states = Params::number_of_states(params);
    auto const s = Params::s(params);
    auto const boost_true_positive_feedback = Params::boost_true_positive_feedback(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const verbose = Params::verbose(params);

    if (auto sm = check_response_y(y, threshold);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    auto const number_of_examples = X.size();

    std::vector<int> ix(number_of_examples);

    for (unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        LOG(info) << "Epoch " << epoch + 1 << '\n';

        std::iota(ix.begin(), ix.end(), 0);
        std::shuffle(ix.begin(), ix.end(), state.igen);

        state.cache.ct.populate(s, state.igen);

        for (auto i = 0u; i < number_of_examples; ++i)
        {
            regressor_update_impl(
                X[ix[i]],
                y[ix[i]],

                threshold,
                number_of_clauses,
                number_of_states,
                s,
                boost_true_positive_feedback,
                n_jobs,

                state.igen,
                state.fgen,
                ta_state,
                state.cache,

                clause_output_tile_size
            );
        }
    }

    return {S_OK, ""};
}


template<typename RegressorStateType, typename SampleType>
status_message_t
fit_regressor_impl_T(
    RegressorStateType & state,
    std::vector<SampleType> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    if (auto sm = check_response_y(y, Params::threshold(state.m_params));
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    // Let it crash for now
//    validate_params();

    int const number_of_features = X.front().size();
    state.m_params["number_of_features"] = param_value_t(number_of_features);

    initialize_state(state);

    return fit_regressor_online_impl(state, state.ta_state, X, y, epochs);
}


template<typename RegressorStateType, typename SampleType>
status_message_t
fit_regressor_impl(
    RegressorStateType & state,
    std::vector<SampleType> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    if (auto sm = check_X_y(X, y);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    return fit_regressor_impl_T(state, X, y, epochs);
}


template<typename RegressorStateType, typename SampleType>
Either<status_message_t, response_vector_type>
predict_regressor_impl(RegressorStateType const & state, std::vector<SampleType> const & X)
{
    if (auto sm = check_for_predict(state, X);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, response_vector_type>::leftOf(std::move(sm));
    }

    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_regressor_clauses(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    response_vector_type rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output_for_predict(
            X[it],
            state.cache.clause_output,
            number_of_clauses / 2,
            state.ta_state,
            n_jobs,
            clause_output_tile_size);

        auto const votes = sum_up_regressor_votes(state.cache.clause_output, threshold, state.ta_state.weights);

        rv[it] = votes;
    }

    return Either<status_message_t, response_vector_type>::rightOf(rv);
}


} // anonymous


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


Either<status_message_t, label_vector_type>
predict_impl(ClassifierStateClassic const & state, std::vector<aligned_vector_char> const & X)
{
    return predict_classifier_impl(state, X);
}


Either<status_message_t, aligned_vector_int>
predict_raw_impl(ClassifierStateClassic const & state, aligned_vector_char const & sample)
{
    return predict_classifier_raw_impl(state, sample);
}


Either<status_message_t, std::vector<aligned_vector_int>>
predict_raw_impl(ClassifierStateClassic const & state, std::vector<aligned_vector_char> const & X)
{
    return predict_classifier_raw_impl(state, X);
}


status_message_t
partial_fit_impl(
    ClassifierStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    if (is_fitted(state.ta_state))
    {
        if (auto sm = check_X_y(X, y);
            sm.first != StatusCode::S_OK)
        {
            return sm;
        }

        return fit_classifier_online_impl(state, state.ta_state, X, y, epochs);
    }
    else
    {
        return fit_impl(state, X, y, max_number_of_labels, epochs);
    }
}


ClassifierClassic::ClassifierClassic(params_t const & params) :
    m_state(params)
{
}


ClassifierClassic::ClassifierClassic(params_t && params) :
    m_state(params)
{
}


ClassifierClassic::ClassifierClassic(ClassifierStateClassic const & state) :
    m_state(state)
{
}


Either<status_message_t, label_type>
predict_impl(ClassifierStateClassic const & state, aligned_vector_char const & sample)
{
    return predict_classifier_impl(state, sample);
}


Either<status_message_t, label_type>
ClassifierClassic::predict(aligned_vector_char const & sample) const
{
    return predict_impl(m_state, sample);
}


Either<status_message_t, label_vector_type>
ClassifierClassic::predict(std::vector<aligned_vector_char> const & X) const
{
    return predict_impl(m_state, X);
}


Either<status_message_t, aligned_vector_int>
ClassifierClassic::predict_raw(aligned_vector_char const & sample) const
{
    return predict_classifier_raw_impl(m_state, sample);
}


Either<status_message_t, std::vector<aligned_vector_int>>
ClassifierClassic::predict_raw(std::vector<aligned_vector_char> const & X) const
{
    return predict_raw_impl(m_state, X);
}


Either<status_message_t, real_type>
evaluate_impl(
    ClassifierStateClassic const & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y)
{
    return evaluate_classifier_impl(state, X, y);
}


Either<status_message_t, real_type>
ClassifierClassic::evaluate(std::vector<aligned_vector_char> const & X, label_vector_type const & y) const
{
    return evaluate_impl(m_state, X, y);
}


status_message_t
ClassifierClassic::partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return partial_fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


status_message_t
ClassifierClassic::fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


status_message_t
fit_impl(
    ClassifierStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    return fit_classifier_impl(state, X, y, max_number_of_labels, epochs);
}


params_t ClassifierClassic::read_params() const
{
    return m_state.m_params;
}


ClassifierStateClassic ClassifierClassic::read_state() const
{
    return m_state;
}


Either<status_message_t, ClassifierClassic>
make_classifier_classic(std::string const & json_params)
{
    auto rv =
        make_classifier_params_from_json(json_params)
        .rightMap([](params_t && params){ return ClassifierClassic(params); })
        ;

    return rv;
}


////////////////////////////////////////////////////////////////////////////////


RegressorClassic::RegressorClassic(params_t const & params) :
    m_state(params)
{
}


RegressorClassic::RegressorClassic(params_t && params) :
    m_state(params)
{
}


RegressorClassic::RegressorClassic(RegressorStateClassic const & state) :
    m_state(state)
{
}


Either<status_message_t, RegressorClassic>
make_regressor_classic(std::string const & json_params)
{
    auto rv =
        make_regressor_params_from_json(json_params)
        .rightMap([](params_t && params){ return RegressorClassic(params); })
        ;

    return rv;
}


status_message_t
RegressorClassic::fit(std::vector<aligned_vector_char> const & X, response_vector_type const & y, unsigned int epochs)
{
    return fit_impl(m_state, X, y, epochs);
}


status_message_t
partial_fit_impl(
    RegressorStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    if (is_fitted(state.ta_state))
    {
        if (auto sm = check_X_y(X, y);
            sm.first != StatusCode::S_OK)
        {
            return sm;
        }

        return fit_regressor_online_impl(state, state.ta_state, X, y, epochs);
    }
    else
    {
        return fit_impl(state, X, y, epochs);
    }
}


status_message_t
fit_impl(
    RegressorStateClassic & state,
    std::vector<aligned_vector_char> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    return fit_regressor_impl(state, X, y, epochs);
}


status_message_t
RegressorClassic::partial_fit(std::vector<aligned_vector_char> const & X, response_vector_type const & y, unsigned int epochs)
{
    return partial_fit_impl(m_state, X, y, epochs);
}


Either<status_message_t, response_vector_type>
predict_impl(RegressorStateClassic const & state, std::vector<aligned_vector_char> const & X)
{
    return predict_regressor_impl(state, X);
}


Either<status_message_t, response_vector_type>
RegressorClassic::predict(std::vector<aligned_vector_char> const & X) const
{
    return predict_impl(m_state, X);
}


Either<status_message_t, response_type>
predict_impl(RegressorStateClassic const & state, aligned_vector_char const & sample)
{
    return predict_regressor_impl(state, sample);
}


Either<status_message_t, response_type>
RegressorClassic::predict(aligned_vector_char const & sample) const
{
    return predict_impl(m_state, sample);
}


params_t RegressorClassic::read_params() const
{
    return m_state.m_params;
}


RegressorStateClassic RegressorClassic::read_state() const
{
    return m_state;
}


////////////////////////////////////////////////////////////////////////////////


status_message_t
fit_impl(
    ClassifierStateBitwise & state,
    std::vector<bit_vector_uint64> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    return fit_classifier_impl(state, X, y, max_number_of_labels, epochs);
}


status_message_t
ClassifierBitwise::fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


status_message_t
partial_fit_impl(
    ClassifierStateBitwise & state,
    std::vector<bit_vector_uint64> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    if (is_fitted(state.ta_state))
    {
        if (auto sm = check_X_y(X, y);
            sm.first != StatusCode::S_OK)
        {
            return sm;
        }

        return fit_classifier_online_impl(state, state.ta_state, X, y, epochs);
    }
    else
    {
        return fit_impl(state, X, y, max_number_of_labels, epochs);
    }
}


status_message_t
ClassifierBitwise::partial_fit(std::vector<bit_vector_uint64> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return partial_fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


Either<status_message_t, label_type>
predict_impl(ClassifierStateBitwise const & state, bit_vector_uint64 const & sample)
{
    return predict_classifier_impl(state, sample);
}


Either<status_message_t, label_type>
ClassifierBitwise::predict(bit_vector_uint64 const & sample) const
{
    return predict_impl(m_state, sample);
}


Either<status_message_t, label_vector_type>
predict_impl(ClassifierStateBitwise const & state, std::vector<bit_vector_uint64> const & X)
{
    return predict_classifier_impl(state, X);
}


Either<status_message_t, label_vector_type>
ClassifierBitwise::predict(std::vector<bit_vector_uint64> const & X) const
{
    return predict_impl(m_state, X);
}


Either<status_message_t, aligned_vector_int>
predict_raw_impl(ClassifierStateBitwise const & state, bit_vector_uint64 const & sample)
{
    return predict_classifier_raw_impl(state, sample);
}


Either<status_message_t, aligned_vector_int>
ClassifierBitwise::predict_raw(bit_vector_uint64 const & sample) const
{
    return predict_raw_impl(m_state, sample);
}


Either<status_message_t, std::vector<aligned_vector_int>>
predict_raw_impl(ClassifierStateBitwise const & state, std::vector<bit_vector_uint64> const & X)
{
    return predict_classifier_raw_impl(state, X);
}


Either<status_message_t, std::vector<aligned_vector_int>>
ClassifierBitwise::predict_raw(std::vector<bit_vector_uint64> const & X) const
{
    return predict_raw_impl(m_state, X);
}


params_t ClassifierBitwise::read_params() const
{
    return m_state.m_params;
}


ClassifierStateBitwise ClassifierBitwise::read_state() const
{
    return m_state;
}


ClassifierBitwise::ClassifierBitwise(ClassifierStateBitwise const & state) :
    m_state(state)
{
}


Either<status_message_t, ClassifierBitwise>
make_classifier_bitwise(std::string const & json_params)
{
    auto rv =
        make_classifier_params_from_json(json_params)
        .rightMap([](params_t && params){ return ClassifierBitwise(params); })
        ;

    return rv;
}


ClassifierBitwise::ClassifierBitwise(params_t const & params) :
    m_state(params)
{
}


ClassifierBitwise::ClassifierBitwise(params_t && params) :
    m_state(params)
{
}


Either<status_message_t, real_type>
evaluate_impl(
    ClassifierStateBitwise const & state,
    std::vector<bit_vector_uint64> const & X,
    label_vector_type const & y)
{
    return evaluate_classifier_impl(state, X, y);
}


Either<status_message_t, real_type>
ClassifierBitwise::evaluate(std::vector<bit_vector_uint64> const & X, label_vector_type const & y) const
{
    return evaluate_impl(m_state, X, y);
}


////////////////////////////////////////////////////////////////////////////////


status_message_t
fit_impl(
    RegressorStateBitwise & state,
    std::vector<bit_vector_uint64> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    return fit_regressor_impl(state, X, y, epochs);
}


status_message_t
RegressorBitwise::fit(std::vector<bit_vector_uint64> const & X, response_vector_type const & y, unsigned int epochs)
{
    return fit_impl(m_state, X, y, epochs);
}


status_message_t
partial_fit_impl(
    RegressorStateBitwise & state,
    std::vector<bit_vector_uint64> const & X,
    response_vector_type const & y,
    unsigned int epochs)
{
    if (is_fitted(state.ta_state))
    {
        if (auto sm = check_X_y(X, y);
            sm.first != StatusCode::S_OK)
        {
            return sm;
        }

        return fit_regressor_online_impl(state, state.ta_state, X, y, epochs);
    }
    else
    {
        return fit_impl(state, X, y, epochs);
    }
}


status_message_t
RegressorBitwise::partial_fit(std::vector<bit_vector_uint64> const & X, response_vector_type const & y, unsigned int epochs)
{
    return partial_fit_impl(m_state, X, y, epochs);
}


Either<status_message_t, response_type>
predict_impl(RegressorStateBitwise const & state, bit_vector_uint64 const & sample)
{
    return predict_regressor_impl(state, sample);
}


Either<status_message_t, response_type>
RegressorBitwise::predict(bit_vector_uint64 const & sample) const
{
    return predict_impl(m_state, sample);
}


Either<status_message_t, response_vector_type>
predict_impl(RegressorStateBitwise const & state, std::vector<bit_vector_uint64> const & X)
{
    return predict_regressor_impl(state, X);
}


Either<status_message_t, response_vector_type>
RegressorBitwise::predict(std::vector<bit_vector_uint64> const & X) const
{
    return predict_impl(m_state, X);
}


params_t RegressorBitwise::read_params() const
{
    return m_state.m_params;
}


RegressorStateBitwise RegressorBitwise::read_state() const
{
    return m_state;
}


RegressorBitwise::RegressorBitwise(RegressorStateBitwise const & state) :
    m_state(state)
{
}


Either<status_message_t, RegressorBitwise>
make_regressor_bitwise(std::string const & json_params)
{
    auto rv =
        make_regressor_params_from_json(json_params)
        .rightMap([](params_t && params){ return RegressorBitwise(params); })
        ;

    return rv;
}


RegressorBitwise::RegressorBitwise(params_t const & params) :
    m_state(params)
{
}


RegressorBitwise::RegressorBitwise(params_t && params) :
    m_state(params)
{
}


} // namespace Tsetlini
