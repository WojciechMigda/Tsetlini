#pragma once

#ifndef LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_
#define LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_


namespace Tsetlini
{


struct TAState;
struct TAStateWithSignum;
struct ClassifierStateCache;
struct RegressorStateCache;

template<typename TAStateType, typename EstimatorStateCacheType>
struct EstimatorState;

using ClassifierStateClassic = EstimatorState<TAState, ClassifierStateCache>;
using RegressorStateClassic = EstimatorState<TAState, RegressorStateCache>;

using ClassifierStateBitwise = EstimatorState<TAStateWithSignum, ClassifierStateCache>;
using RegressorStateBitwise = EstimatorState<TAStateWithSignum, RegressorStateCache>;


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_ */
