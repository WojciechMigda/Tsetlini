#pragma once

#ifndef LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_
#define LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_


namespace Tsetlini
{


struct TAState;
struct TAStateWithSignum;
struct ClassifierStateCache;
struct RegressorStateCache;
struct ClassifierStateBitwiseCache;
struct RegressorStateBitwiseCache;

template<typename TAStateType, typename EstimatorStateCacheType>
struct EstimatorState;

using ClassifierStateClassic = EstimatorState<TAState, ClassifierStateCache>;
using RegressorStateClassic = EstimatorState<TAState, RegressorStateCache>;

using ClassifierStateBitwise = EstimatorState<TAStateWithSignum, ClassifierStateBitwiseCache>;
using RegressorStateBitwise = EstimatorState<TAStateWithSignum, RegressorStateBitwiseCache>;


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_ */
