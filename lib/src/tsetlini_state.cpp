#define LOG_MODULE "tsetlini-core"
#include "logger.hpp"

#include "estimator_state.hpp"
#include "tsetlini_types.hpp"
#include "params_companion.hpp"
#include "tsetlini_params.hpp"
#include "tsetlini_estimator_state_private.hpp"

#include <algorithm>
#include <iterator>
#include <thread>
#include <random>
#include <limits>


namespace Tsetlini
{


static
void log_classifier_params(params_t const & params, bool verbose)
{
    LOG(info) << "number_of_labels: " << Params::number_of_labels(params) << '\n';
    LOG(info) << "number_of_clauses: " << Params::number_of_classifier_clauses(params) << '\n';
    LOG(info) << "number_of_features: " << Params::number_of_features(params) << '\n';
    LOG(info) << "s: " << Params::s(params) << '\n';
    LOG(info) << "number_of_states: " << Params::number_of_states(params) << '\n';
    LOG(info) << "threshold: " << Params::threshold(params) << '\n';
    LOG(info) << "weighted: " << Params::weighted(params) << '\n';
    LOG(info) << "counting_type: " << Params::counting_type(params) << '\n';
    LOG(info) << "n_jobs: " << Params::n_jobs(params) << '\n';
    LOG(info) << "random_state: " << Params::random_state(params) << '\n';
}


static
void log_estimator_params(ClassifierStateClassic const & state, bool verbose)
{
    log_classifier_params(state.m_params, verbose);
}


static
void log_estimator_params(ClassifierStateBitwise const & state, bool verbose)
{
    log_classifier_params(state.m_params, verbose);
}


static
void log_regressor_params(params_t const & params, bool verbose)
{
    LOG(info) << "number_of_clauses: " << Params::number_of_regressor_clauses(params) << '\n';
    LOG(info) << "number_of_features: " << Params::number_of_features(params) << '\n';
    LOG(info) << "s: " << Params::s(params) << '\n';
    LOG(info) << "number_of_states: " << Params::number_of_states(params) << '\n';
    LOG(info) << "threshold: " << Params::threshold(params) << '\n';
    LOG(info) << "weighted: " << Params::weighted(params) << '\n';
    LOG(info) << "counting_type: " << Params::counting_type(params) << '\n';
    LOG(info) << "n_jobs: " << Params::n_jobs(params) << '\n';
    LOG(info) << "random_state: " << Params::random_state(params) << '\n';
}


static
void log_estimator_params(RegressorStateClassic const & state, bool verbose)
{
    log_regressor_params(state.m_params, verbose);
}


static
void log_estimator_params(RegressorStateBitwise const & state, bool verbose)
{
    log_regressor_params(state.m_params, verbose);
}


static
std::string normalize_counting_type(
    std::string const & counting_type,
    int number_of_states,
    bool verbose)
{
    std::string rv;

    if ((number_of_states <= std::numeric_limits<std::int8_t>::max()) and
        ("auto" == counting_type or "int8" == counting_type))
    {
        LOG(trace) << "Selected int8 for ta_state\n";
        rv = "int8";
    }
    else if ((number_of_states <= std::numeric_limits<std::int16_t>::max()) and
        ("auto" == counting_type or "int8" == counting_type or "int16" == counting_type))
    {
        LOG(trace) << "Selected int16 for ta_state\n";
        rv = "int16";
    }
    else
    {
        LOG(trace) << "Selected int32 for ta_state\n";
        rv = "int32";
    }

    return rv;
}



static
int number_of_classifier_clauses(params_t const & params)
{
    return Params::number_of_classifier_clauses(params);
}


static
int number_of_estimator_clauses(ClassifierStateClassic const & est)
{
    return number_of_classifier_clauses(est.m_params);
}


static
int number_of_estimator_clauses(ClassifierStateBitwise const & est)
{
    return number_of_classifier_clauses(est.m_params);
}


static
int number_of_regressor_clauses(params_t const & params)
{
    return Params::number_of_regressor_clauses(params);
}


static
int number_of_estimator_clauses(RegressorStateClassic const & est)
{
    return number_of_regressor_clauses(est.m_params);
}


static
int number_of_estimator_clauses(RegressorStateBitwise const & est)
{
    return number_of_regressor_clauses(est.m_params);
}


template<typename EstimatorStateType>
void initialize_state(EstimatorStateType & state)
{
    static_assert(is_estimator_state<EstimatorStateType>::value, "EstimatorStateType requirement is not met");

    auto & params = state.m_params;
    auto const verbose = Params::verbose(params);

    state.igen.init(Params::random_state(params));
    state.fgen.init(Params::random_state(params));

    log_estimator_params(state, verbose);

    auto & ta_state = state.ta_state;

    auto const number_of_states = Params::number_of_states(params);
    auto const number_of_clauses = number_of_estimator_clauses(state);
    auto const number_of_features = Params::number_of_features(params);
    auto const counting_type =
        normalize_counting_type(Params::counting_type(params), number_of_states, verbose);
    auto const weighted = Params::weighted(params);

    using ta_state_type = typename EstimatorStateType::ta_state_type;
    ta_state_type::initialize(ta_state, counting_type, number_of_clauses, number_of_features, weighted, state.igen);

    using cache_type = typename EstimatorStateType::cache_type;
    cache_type::reset(state.cache, params);
}


// explicit template instantiations
template void initialize_state<ClassifierStateClassic>(ClassifierStateClassic & state);
template void initialize_state<RegressorStateClassic>(RegressorStateClassic & state);
template void initialize_state<ClassifierStateBitwise>(ClassifierStateBitwise & state);
template void initialize_state<RegressorStateBitwise>(RegressorStateBitwise & state);

template<typename EstimatorStateType>
void reset_state_cache(EstimatorStateType & state)
{
    EstimatorStateType::cache_type::reset(state.cache, state.m_params);
}

// explicit template instantiations
template void reset_state_cache<ClassifierStateClassic>(ClassifierStateClassic & state);
template void reset_state_cache<RegressorStateClassic>(RegressorStateClassic & state);
template void reset_state_cache<ClassifierStateBitwise>(ClassifierStateBitwise & state);
template void reset_state_cache<RegressorStateBitwise>(RegressorStateBitwise & state);


} // namespace Tsetlino
