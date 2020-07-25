#pragma once

#include "estimator_state_fwd.hpp"

namespace Tsetlini
{


template<typename EstimatorStateType>
void initialize_state(EstimatorStateType & state);
template<typename EstimatorStateType>
void reset_state_cache(EstimatorStateType & state);


} // namespace Tsetlini
