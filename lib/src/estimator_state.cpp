#include "estimator_state.hpp"
#include "estimator_state_fwd.hpp"


namespace Tsetlini
{


/// Deleter
template<typename TAStateType, typename EstimatorStateCacheType>
void EstimatorStateDeleter(EstimatorState<TAStateType, EstimatorStateCacheType> *p)
{
    delete p;
}

// and its explicit template instantiations
template void EstimatorStateDeleter<TAState, ClassifierStateCache>(EstimatorState<TAState, ClassifierStateCache> *);
template void EstimatorStateDeleter<TAState, RegressorStateCache>(EstimatorState<TAState, RegressorStateCache> *);
template void EstimatorStateDeleter<TAStateWithSignum, ClassifierStateCache>(EstimatorState<TAStateWithSignum, ClassifierStateCache> *);
template void EstimatorStateDeleter<TAStateWithSignum, RegressorStateCache>(EstimatorState<TAStateWithSignum, RegressorStateCache> *);


/// equality operator
template<typename TAStateType, typename EstimatorStateCacheType>
bool operator==(EstimatorState<TAStateType, EstimatorStateCacheType> const & lhs, EstimatorState<TAStateType, EstimatorStateCacheType> const & rhs)
{
    if (&lhs == &rhs)
    {
        return true;
    }
    else
    {
        return
            lhs.ta_state == rhs.ta_state
            and lhs.igen == rhs.igen
            and lhs.fgen == rhs.fgen
            and lhs.m_params == rhs.m_params
            and EstimatorStateCacheType::are_equal(lhs.cache, rhs.cache)
        ;
    }
}

// and its explicit template instantiations
template bool operator==<TAState, ClassifierStateCache>(ClassifierStateClassic const &, ClassifierStateClassic const &);
template bool operator==<TAState, RegressorStateCache>(RegressorStateClassic const &, RegressorStateClassic const &);
template bool operator==<TAStateWithSignum, ClassifierStateCache>(ClassifierStateBitwise const &, ClassifierStateBitwise const &);
template bool operator==<TAStateWithSignum, RegressorStateCache>(RegressorStateBitwise const &, RegressorStateBitwise const &);


}  // namespace Tsetlini
