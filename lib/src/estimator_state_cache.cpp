#include "coin_tosser.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_params.hpp"
#include "params_companion.hpp"
#include "mt.hpp"

#include "strong_type/strong_type.hpp"


namespace Tsetlini
{


void ClassifierStateCache::reset(
    ClassifierStateCache::value_type & cache,
    params_t const & params)
{
    auto number_of_outputs = Params::number_of_classifier_clause_outputs(params);

    cache.clause_output.clear();
    cache.clause_output.resize(value_of(number_of_outputs));
    cache.label_sum.clear();
    cache.label_sum.resize(value_of(Params::number_of_labels(params)));
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(value_of(number_of_outputs));

    cache.ct = CoinTosserExact(
        real_type{1.} / value_of(Params::s(params)),
        value_of(Params::number_of_features(params)));
}


bool ClassifierStateCache::are_equal(value_type const & lhs, value_type const & rhs)
{
    return
        lhs.feedback_to_clauses.size() == rhs.feedback_to_clauses.size()
        and lhs.clause_output.size() == rhs.clause_output.size()
        and lhs.label_sum.size() == rhs.label_sum.size();
}


////////////////////////////////////////////////////////////////////////////////


void RegressorStateCache::reset(
    RegressorStateCache::value_type & cache,
    params_t const & params)
{
    auto const number_of_outputs = Params::number_of_regressor_clause_outputs(params);

    cache.clause_output.clear();
    cache.clause_output.resize(value_of(number_of_outputs));

    cache.ct = CoinTosserExact(
        real_type{1.} / value_of(Params::s(params)),
        value_of(Params::number_of_features(params)));
}


bool RegressorStateCache::are_equal(value_type const & lhs, value_type const & rhs)
{
    return lhs.clause_output.size() == rhs.clause_output.size();
}


////////////////////////////////////////////////////////////////////////////////


void ClassifierStateBitwiseCache::reset(
    ClassifierStateBitwiseCache::value_type & cache,
    params_t const & params)
{
    auto number_of_outputs = Params::number_of_classifier_clause_outputs(params);

    cache.clause_output.clear();
    cache.clause_output.resize(value_of(number_of_outputs));
    cache.label_sum.clear();
    cache.label_sum.resize(value_of(Params::number_of_labels(params)));
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(value_of(number_of_outputs));

    /*
     * While a factor of 24 (= 3 * 8) is arbitrary and seems to work, for smaller sample
     * sizes (size <= alignment)
     * one could improve it by using 24 * min(alignment, number_of_features))
     * instead, so that CoinTosserBitwise will have a chance to return pointer
     * to a position different than just the start.
     */
    auto const base_size = value_of(Params::number_of_features(params));
    cache.ct = CoinTosserBitwise(base_size, 3 * 8 * base_size);
}


bool ClassifierStateBitwiseCache::are_equal(value_type const & lhs, value_type const & rhs)
{
    return
        lhs.feedback_to_clauses.size() == rhs.feedback_to_clauses.size()
        and lhs.clause_output.size() == rhs.clause_output.size()
        and lhs.label_sum.size() == rhs.label_sum.size();
}


////////////////////////////////////////////////////////////////////////////////


void RegressorStateBitwiseCache::reset(
    RegressorStateBitwiseCache::value_type & cache,
    params_t const & params)
{
    auto const number_of_outputs = Params::number_of_regressor_clause_outputs(params);

    cache.clause_output.clear();
    cache.clause_output.resize(value_of(number_of_outputs));

    auto const base_size = value_of(Params::number_of_features(params));
    cache.ct = CoinTosserBitwise(base_size, 3 * 8 * base_size);
}


bool RegressorStateBitwiseCache::are_equal(value_type const & lhs, value_type const & rhs)
{
    return lhs.clause_output.size() == rhs.clause_output.size();
}


}  // namespace Tsetlini
