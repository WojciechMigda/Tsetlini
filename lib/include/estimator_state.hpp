#pragma once

#ifndef LIB_INCLUDE_ESTIMATOR_STATE_HPP_
#define LIB_INCLUDE_ESTIMATOR_STATE_HPP_

#include "tsetlini_params.hpp"
#include "ta_state.hpp"
#include "estimator_state_cache.hpp"
#include "mt.hpp"


namespace Tsetlini
{


template<typename TAState, typename StateCache>
struct EstimatorState
{
    params_t m_params;
    typename TAState::value_type ta_state;
    mutable typename StateCache::value_type cache;

    IRNG igen;
    FRNG fgen;

    explicit EstimatorState(params_t const & params) :
        m_params(params)
    {
    }

    bool operator==(EstimatorState const & other) const
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
                and StateCache::are_equal(cache, other.cache)
                ;
        }
    }
};


using ClassifierStateNew = EstimatorState<TAState, ClassifierStateCache>;
using RegressorStateNew = EstimatorState<TAState, RegressorStateCache>;


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_ESTIMATOR_STATE_HPP_ */
