#include "coin_tosser.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_params.hpp"
#include "params_companion.hpp"
#include "mt.hpp"


namespace Tsetlini
{


void ClassifierStateCache::reset(
    ClassifierStateCache::value_type & cache,
    params_t const & params)
{
    cache.clause_output.clear();
    cache.clause_output.resize(Params::number_of_classifier_clauses(params) / 2);
    cache.label_sum.clear();
    cache.label_sum.resize(Params::number_of_labels(params));
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(Params::number_of_classifier_clauses(params) / 2);

    cache.ct = CoinTosserExact(
        1. / Params::s(params),
        Params::number_of_features(params));
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
    cache.clause_output.clear();
    cache.clause_output.resize(Params::number_of_regressor_clauses(params) / 2);
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(Params::number_of_regressor_clauses(params) / 2);

    cache.ct = CoinTosserExact(
        1. / Params::s(params),
        Params::number_of_features(params));
}


bool RegressorStateCache::are_equal(value_type const & lhs, value_type const & rhs)
{
    return
        lhs.feedback_to_clauses.size() == rhs.feedback_to_clauses.size()
        and lhs.clause_output.size() == rhs.clause_output.size();
}


////////////////////////////////////////////////////////////////////////////////


void ClassifierStateBitwiseCache::reset(
    ClassifierStateBitwiseCache::value_type & cache,
    params_t const & params)
{
    cache.clause_output.clear();
    cache.clause_output.resize(Params::number_of_classifier_clauses(params) / 2);
    cache.label_sum.clear();
    cache.label_sum.resize(Params::number_of_labels(params));
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(Params::number_of_classifier_clauses(params) / 2);

    /*
     * While a factor of 24 (= 3 * 8) is arbitrary and seems to work, for smaller sample
     * sizes (size <= alignment)
     * one could improve it by using 24 * min(alignment, number_of_features))
     * instead, so that CoinTosserBitwise will have a chance to return pointer
     * to a position different than just the start.
     */
    cache.ct = CoinTosserBitwise(Params::number_of_features(params),
        3 * 8 * Params::number_of_features(params));
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
    cache.clause_output.clear();
    cache.clause_output.resize(Params::number_of_regressor_clauses(params) / 2);
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(Params::number_of_regressor_clauses(params) / 2);

    cache.ct = CoinTosserBitwise(Params::number_of_features(params),
        3 * 8 * Params::number_of_features(params));
}


bool RegressorStateBitwiseCache::are_equal(value_type const & lhs, value_type const & rhs)
{
    return
        lhs.feedback_to_clauses.size() == rhs.feedback_to_clauses.size()
        and lhs.clause_output.size() == rhs.clause_output.size();
}


}  // namespace Tsetlini
