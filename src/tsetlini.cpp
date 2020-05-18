#define LOG_MODULE "tsetlini"
#include "logger.hpp"

#include "tsetlini_private.hpp"
#include "tsetlini_algo.hpp"

#include "tsetlini_params.hpp"
#include "tsetlini.hpp"
#include "mt.hpp"
#include "assume_aligned.hpp"
#include "params_companion.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_state.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_classifier_state_private.hpp"

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


status_message_t check_X_y(
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y)
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
    else if (not std::all_of(X.cbegin(), X.cend(),
        [](auto const & row)
        {
            return std::all_of(
                row.cbegin(), row.cend(), [](auto v){ return v == 0 || v == 1; });
        }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values of 0 and 1 can be used in X"};
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


bool is_fitted(ClassifierState const & state)
{
    return std::visit([](auto const & ta_state){ return ta_state.shape().first != 0; }, state.ta_state);
}


status_message_t check_for_predict(
    ClassifierState const & state,
    std::vector<aligned_vector_char> const & X)
{
    if (not is_fitted(state))
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
            [&first](auto const & row){ return row.size() == first->size(); }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "All rows of X must have the same number of feature columns"};
    }
    else if (X.front().size() - Params::number_of_features(state.m_params) != 0)
    {
        return {StatusCode::S_VALUE_ERROR,
            "Predict called with X, which number of features " + std::to_string(X.front().size()) +
            " does not match that from prior fit " + std::to_string(Params::number_of_features(state.m_params))};
    }
    else if (not std::all_of(X.cbegin(), X.cend(),
        [](auto const & row)
        {
            return std::all_of(
                row.cbegin(), row.cend(), [](auto v){ return v == 0 || v == 1; });
        }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values of 0 and 1 can be used in X"};
    }

    return {StatusCode::S_OK, ""};
}


status_message_t check_for_predict(
    ClassifierState const & state,
    aligned_vector_char const & sample)
{
    if (not is_fitted(state))
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
    else if (not std::all_of(sample.cbegin(), sample.cend(), [](auto v){ return v == 0 || v == 1; }))
    {
        return {StatusCode::S_VALUE_ERROR,
            "Only values of 0 and 1 can be used in sample for prediction"};
    }

    return {StatusCode::S_OK, ""};
}


template<typename state_type, typename row_type>
void update_impl(
    row_type const & X,
    label_type target_label,
    label_type opposite_label,

    int const number_of_pos_neg_clauses_per_label,
    int const threshold,
    int const number_of_clauses,
    int const number_of_features,
    int const number_of_states,
    real_type s,
    int const boost_true_positive_feedback,
    int const n_jobs,

    FRNG & fgen,
    numeric_matrix<state_type> & ta_state,
    ClassifierState::Cache & cache,

    int clause_output_tile_size
    )
{
    calculate_clause_output(
        X,
        cache.clause_output,
        number_of_clauses,
        number_of_features,
        ta_state,
        n_jobs,
        clause_output_tile_size
    );

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


    std::fill(cache.feedback_to_clauses.begin(), cache.feedback_to_clauses.end(), 0);


    const auto S_inv = ONE / s;

    const auto THR2_inv = (ONE / (threshold * 2));
    const auto THR_pos = THR2_inv * (threshold - cache.label_sum[target_label]);
    const auto THR_neg = THR2_inv * (threshold + cache.label_sum[opposite_label]);

    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_pos)
        {
            continue;
        }

        // Type I Feedback
        cache.feedback_to_clauses[pos_clause_index(target_label, j, number_of_pos_neg_clauses_per_label)] = 1;
    }
    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_pos)
        {
            continue;
        }

        // Type II Feedback
        cache.feedback_to_clauses[neg_clause_index(target_label, j, number_of_pos_neg_clauses_per_label)] = -1;
    }

    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_neg)
        {
            continue;
        }

        cache.feedback_to_clauses[pos_clause_index(opposite_label, j, number_of_pos_neg_clauses_per_label)] = -1;
    }
    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_neg)
        {
            continue;
        }

        cache.feedback_to_clauses[neg_clause_index(opposite_label, j, number_of_pos_neg_clauses_per_label)] = 1;
    }


    train_automata_batch(
        ta_state,
        0,
        number_of_clauses,
        cache.feedback_to_clauses.data(),
        cache.clause_output.data(),
        number_of_features,
        number_of_states,
        S_inv,
        X,
        boost_true_positive_feedback,
        cache.fcache[0]
    );
}


Either<status_message_t, real_type>
evaluate_impl(
    ClassifierState const & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y)
{
    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    int errors = 0;

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        std::visit([&](auto & ta_state)
            {
                calculate_clause_output_for_predict(
                    X[it],
                    state.cache.clause_output,
                    number_of_clauses,
                    number_of_features,
                    ta_state,
                    n_jobs,
                    clause_output_tile_size);
            }, state.ta_state);

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


Either<status_message_t, label_type>
predict_impl(ClassifierState const & state, aligned_vector_char const & sample)
{
    if (auto sm = check_for_predict(state, sample);
        sm.first != StatusCode::S_OK)
    {
        return Either<status_message_t, label_type>::leftOf(std::move(sm));
    }

    auto const n_jobs = Params::n_jobs(state.m_params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(state.m_params);

    std::visit([&](auto & ta_state)
        {
            calculate_clause_output_for_predict(
                sample,
                state.cache.clause_output,
                Params::number_of_clauses(state.m_params),
                Params::number_of_features(state.m_params),
                ta_state,
                n_jobs,
                clause_output_tile_size);
        }, state.ta_state);

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


} // anonymous


Either<status_message_t, label_vector_type>
predict_impl(ClassifierState const & state, std::vector<aligned_vector_char> const & X)
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
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    label_vector_type rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        std::visit([&](auto & ta_state)
            {
                calculate_clause_output_for_predict(
                    X[it],
                    state.cache.clause_output,
                    number_of_clauses,
                    number_of_features,
                    ta_state,
                    n_jobs,
                    clause_output_tile_size);
            }, state.ta_state);

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


Either<status_message_t, aligned_vector_int>
predict_raw_impl(ClassifierState const & state, aligned_vector_char const & sample)
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
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);


    std::visit([&](auto & ta_state)
        {
            calculate_clause_output_for_predict(
                sample,
                state.cache.clause_output,
                number_of_clauses,
                number_of_features,
                ta_state,
                n_jobs,
                clause_output_tile_size);
        }, state.ta_state);

    sum_up_all_label_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        number_of_labels,
        number_of_pos_neg_clauses_per_label,
        threshold);

    return Either<status_message_t, aligned_vector_int>::rightOf(state.cache.label_sum);
}


Either<status_message_t, std::vector<aligned_vector_int>>
predict_raw_impl(ClassifierState const & state, std::vector<aligned_vector_char> const & X)
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
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const n_jobs = Params::n_jobs(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);

    std::vector<aligned_vector_int> rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        std::visit([&](auto & ta_state)
            {
                calculate_clause_output_for_predict(
                    X[it],
                    state.cache.clause_output,
                    number_of_clauses,
                    number_of_features,
                    ta_state,
                    n_jobs,
                    clause_output_tile_size);
            }, state.ta_state);

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


template<typename state_type, typename row_type>
status_message_t
fit_online_impl(
    ClassifierState & state,
    numeric_matrix<state_type> & ta_state,
    std::vector<row_type> const & X,
    label_vector_type const & y,
    unsigned int epochs)
{
    if (auto sm = check_X_y(X, y);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    auto labels = unique_labels(y);

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const number_of_states = Params::number_of_states(params);
    auto const s = Params::s(params);
    auto const boost_true_positive_feedback = Params::boost_true_positive_feedback(params);
    auto const clause_output_tile_size = Params::clause_output_tile_size(params);
    auto const n_jobs = Params::n_jobs(params);

    if (auto sm = check_labels(labels, number_of_labels);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

    auto const number_of_examples = X.size();

    std::vector<int> ix(number_of_examples);
    std::iota(ix.begin(), ix.end(), 0);

    std::mt19937 gen(state.igen());

    label_vector_type opposite_y(y.size());

    for (unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        generate_opposite_y(y, opposite_y, number_of_labels, state.igen);
        std::shuffle(ix.begin(), ix.end(), gen);

        for (auto i = 0u; i < number_of_examples; ++i)
        {
            update_impl(
                X[ix[i]],
                y[ix[i]],
                opposite_y[ix[i]],

                number_of_pos_neg_clauses_per_label,
                threshold,
                number_of_clauses,
                number_of_features,
                number_of_states,
                s,
                boost_true_positive_feedback,
                n_jobs,

                state.fgen,
                ta_state,
                state.cache,

                clause_output_tile_size
            );
        }
    }

    return {S_OK, ""};
}


template<typename RowType>
status_message_t
fit_online_impl(
    ClassifierState & state,
    std::vector<RowType> const & X,
    label_vector_type const & y,
    unsigned int epochs)
{
    return std::visit([&](auto & ta_state)
        {
            return fit_online_impl(state, ta_state, X, y, epochs);
        }, state.ta_state);
}


template<typename RowType>
status_message_t
fit_impl_T(
    ClassifierState & state,
    std::vector<RowType> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    if (auto sm = check_X_y(X, y);
        sm.first != StatusCode::S_OK)
    {
        return sm;
    }

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

    return fit_online_impl(state, X, y, epochs);
}


status_message_t
partial_fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    if (is_fitted(state))
    {
        return fit_online_impl(state, X, y, epochs);
    }
    else
    {
        return fit_impl(state, X, y, max_number_of_labels, epochs);
    }
}




Classifier::Classifier(params_t const & params) :
    m_state(params)
{
}


Classifier::Classifier(params_t && params) :
    m_state(params)
{
}


Either<status_message_t, label_type>
Classifier::predict(aligned_vector_char const & sample) const
{
    return predict_impl(m_state, sample);
}


Either<status_message_t, label_vector_type>
Classifier::predict(std::vector<aligned_vector_char> const & X) const
{
    return predict_impl(m_state, X);
}


Either<status_message_t, aligned_vector_int>
Classifier::predict_raw(aligned_vector_char const & sample) const
{
    return predict_raw_impl(m_state, sample);
}


Either<status_message_t, std::vector<aligned_vector_int>>
Classifier::predict_raw(std::vector<aligned_vector_char> const & X) const
{
    return predict_raw_impl(m_state, X);
}


Either<status_message_t, real_type>
Classifier::evaluate(std::vector<aligned_vector_char> const & X, label_vector_type const & y) const
{
    return evaluate_impl(m_state, X, y);
}


status_message_t
Classifier::partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return partial_fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


status_message_t
Classifier::fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


status_message_t
fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    return fit_impl_T(state, X, y, max_number_of_labels, epochs);
}


params_t Classifier::read_params() const
{
    return m_state.m_params;
}


ClassifierState Classifier::read_state() const
{
    return m_state;
}


Either<status_message_t, Classifier>
make_classifier(std::string const & json_params)
{
    auto rv =
        make_params_from_json(json_params)
        .rightMap([](params_t && params){ return Classifier(params); })
        ;

    return rv;
}


} // namespace Tsetlini
