#pragma once

#ifndef LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_
#define LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_

#include <memory>


namespace Tsetlini
{


struct TAState;
struct TAStateWithSignum;
struct ClassifierStateCache;
struct RegressorStateCache;

template<typename TAStateType, typename EstimatorStateCacheType>
struct EstimatorState;

template<typename TAStateType, typename EstimatorStateCacheType>
void EstimatorStateDeleter(EstimatorState<TAStateType, EstimatorStateCacheType> *);

static constexpr
auto ClassifierStateClassicDeleter = &EstimatorStateDeleter<TAState, ClassifierStateCache>;
static constexpr
auto RegressorStateClassicDeleter = &EstimatorStateDeleter<TAState, RegressorStateCache>;
static constexpr
auto ClassifierStateBitwiseDeleter = &EstimatorStateDeleter<TAStateWithSignum, ClassifierStateCache>;
static constexpr
auto RegressorStateBitwiseDeleter = &EstimatorStateDeleter<TAStateWithSignum, RegressorStateCache>;


using ClassifierStateClassic = EstimatorState<TAState, ClassifierStateCache>;
using RegressorStateClassic = EstimatorState<TAState, RegressorStateCache>;

using ClassifierStateBitwise = EstimatorState<TAStateWithSignum, ClassifierStateCache>;
using RegressorStateBitwise = EstimatorState<TAStateWithSignum, RegressorStateCache>;


using ClassifierStateClassicDeleterType = void(*)(ClassifierStateClassic *);
using RegressorStateClassicDeleterType = void(*)(RegressorStateClassic *);
using ClassifierStateBitwiseDeleterType = void(*)(ClassifierStateBitwise *);
using RegressorStateBitwiseDeleterType = void(*)(RegressorStateBitwise *);


using ClassifierStateClassicPtr = std::unique_ptr<ClassifierStateClassic, ClassifierStateClassicDeleterType>;
using RegressorStateClassicPtr = std::unique_ptr<RegressorStateClassic, RegressorStateClassicDeleterType>;
using ClassifierStateBitwisePtr = std::unique_ptr<ClassifierStateBitwise, ClassifierStateBitwiseDeleterType>;
using RegressorStateBitwisePtr = std::unique_ptr<RegressorStateBitwise, RegressorStateBitwiseDeleterType>;


template<typename TAStateType, typename EstimatorStateCacheType>
bool operator==(EstimatorState<TAStateType, EstimatorStateCacheType> const &, EstimatorState<TAStateType, EstimatorStateCacheType> const &);

extern template bool operator==<TAState, ClassifierStateCache>(ClassifierStateClassic const &, ClassifierStateClassic const &);
extern template bool operator==<TAState, RegressorStateCache>(RegressorStateClassic const &, RegressorStateClassic const &);
extern template bool operator==<TAStateWithSignum, ClassifierStateCache>(ClassifierStateBitwise const &, ClassifierStateBitwise const &);
extern template bool operator==<TAStateWithSignum, RegressorStateCache>(RegressorStateBitwise const &, RegressorStateBitwise const &);


}  // namespace Tsetlini


template<typename TAStateType, typename EstimatorStateCacheType>
constexpr bool equal(
    Tsetlini::EstimatorState<TAStateType, EstimatorStateCacheType> const & lhs,
    Tsetlini::EstimatorState<TAStateType, EstimatorStateCacheType> const & rhs)
{
    return Tsetlini::operator==(lhs, rhs);
}


#endif /* LIB_INCLUDE_ESTIMATOR_STATE_FWD_HPP_ */
