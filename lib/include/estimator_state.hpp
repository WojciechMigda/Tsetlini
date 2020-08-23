#pragma once

#ifndef LIB_INCLUDE_ESTIMATOR_STATE_HPP_
#define LIB_INCLUDE_ESTIMATOR_STATE_HPP_

#include "estimator_state_fwd.hpp"
#include "tsetlini_params.hpp"
#include "ta_state.hpp"
#include "estimator_state_cache.hpp"
#include "mt.hpp"

#include <type_traits>
#include <string>


namespace std
{
// this is needed for std::variant comparison, probably illegal. TODO
static constexpr
bool operator==(nullopt_t const & lhs, nullopt_t const & rhs){ return true; }

}


namespace Tsetlini
{


template<typename, typename = std::void_t<>>
struct is_estimator_state
    : std::false_type
{
};


template<typename T>
struct is_estimator_state<T, std::void_t<
    typename T::cache_type
    , typename T::ta_state_type
    , decltype(std::declval<T>().m_params)
    , decltype(std::declval<T>().ta_state)
    , decltype(std::declval<T>().cache)
    , decltype(std::declval<T>().igen)
    , decltype(std::declval<T>().fgen)
>>
    : std::true_type
{
};


template<typename TAStateType, typename EstimatorStateCacheType>
struct EstimatorState
{
    static_assert(is_TA_state<TAStateType>::value, "TAState requirement not met");
    static_assert(is_estimator_state_cache<EstimatorStateCacheType>::value, "EstimatorStateCache requirement not met");

    using ta_state_type = TAStateType;
    using cache_type = EstimatorStateCacheType;

    params_t m_params;
    typename ta_state_type::value_type ta_state;
    mutable typename cache_type::value_type cache;

    IRNG igen;
    FRNG fgen;

    explicit EstimatorState(params_t const & params);

    bool operator==(EstimatorState const & other) const;
};


template<typename TAStateType, typename EstimatorStateCacheType>
EstimatorState<TAStateType, EstimatorStateCacheType>::EstimatorState(params_t const & params)
    : m_params(params)
{
}


template<typename TAStateType, typename EstimatorStateCacheType>
bool EstimatorState<TAStateType, EstimatorStateCacheType>::operator==(EstimatorState const & other) const
{
    if (this == &other)
    {
        return true;
    }
    else
    {
        return
            ta_state == other.ta_state
            and igen == other.igen
            and fgen == other.fgen
            and m_params == other.m_params
            and cache_type::are_equal(cache, other.cache)
        ;
    }
}

using ClassifierStateClassic = EstimatorState<TAState, ClassifierStateCache>;
using RegressorStateClassic = EstimatorState<TAState, RegressorStateCache>;

using ClassifierStateBitwise = EstimatorState<TAStateWithSignum, ClassifierStateCache>;
using RegressorStateBitwise = EstimatorState<TAStateWithSignum, RegressorStateCache>;


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_ESTIMATOR_STATE_HPP_ */
