#pragma once

#ifndef LIB_SRC_TSETLINI_ESTIMATOR_STATE_PRIVATE_HPP_
#define LIB_SRC_TSETLINI_ESTIMATOR_STATE_PRIVATE_HPP_

#include "estimator_state_fwd.hpp"

namespace Tsetlini
{


template<typename EstimatorStateType>
void initialize_state(EstimatorStateType & state);
template<typename EstimatorStateType>
void reset_state_cache(EstimatorStateType & state);


} // namespace Tsetlini

#endif /* LIB_SRC_TSETLINI_ESTIMATOR_STATE_PRIVATE_HPP_ */
