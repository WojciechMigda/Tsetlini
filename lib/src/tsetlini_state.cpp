#define LOG_MODULE "tsetlini-core"
#include "logger.hpp"

#include "tsetlini_state.hpp"
#include "tsetlini_types.hpp"
#include "params_companion.hpp"
#include "tsetlini_params.hpp"
#include "tsetlini_classifier_state_private.hpp"

#include <algorithm>
#include <iterator>
#include <thread>
#include <random>
#include <limits>


namespace std
{
// this is needed for std::variant comparison, probably illegal. TODO
static constexpr
bool operator==(nullopt_t const & lhs, nullopt_t const & rhs){ return true; }

}


namespace Tsetlini
{


void initialize_state(ClassifierState & state)
{
    auto & params = state.m_params;

    auto const verbose = Params::verbose(params);

    state.igen.init(Params::random_state(params));
    state.fgen.init(Params::random_state(params));

    LOG(info) << "number_of_labels: " << Params::number_of_labels(params) << '\n';
    LOG(info) << "number_of_clauses: " << Params::number_of_classifier_clauses(params) << '\n';
    LOG(info) << "number_of_features: " << Params::number_of_features(params) << '\n';
    LOG(info) << "s: " << Params::s(params) << '\n';
    LOG(info) << "number_of_states: " << Params::number_of_states(params) << '\n';
    LOG(info) << "threshold: " << Params::threshold(params) << '\n';
    LOG(info) << "counting_type: " << Params::counting_type(params) << '\n';
    LOG(info) << "n_jobs: " << Params::n_jobs(params) << '\n';
    LOG(info) << "random_state: " << Params::random_state(params) << '\n';

    // convenience reference variables
    auto & ta_state_v = state.ta_state;
    auto & igen = state.igen;

    auto const number_of_states = Params::number_of_states(params);
    auto const & counting_type = Params::counting_type(params);
    auto const number_of_clauses = Params::number_of_classifier_clauses(params);
    auto const number_of_features = Params::number_of_features(params);

    if (number_of_states <= std::numeric_limits<std::int8_t>::max()
        and ("auto" == counting_type or "int8" == counting_type))
    {
        LOG(trace) << "Selected int8 for ta_state\n";
        ta_state_v = numeric_matrix_int8(2 * number_of_clauses, number_of_features);
    }
    else if (number_of_states <= std::numeric_limits<std::int16_t>::max()
        and ("auto" == counting_type or "int8" == counting_type or "int16" == counting_type))
    {
        LOG(trace) << "Selected int16 for ta_state\n";
        ta_state_v = numeric_matrix_int16(2 * number_of_clauses, number_of_features);
    }
    else
    {
        LOG(trace) << "Selected int32 for ta_state\n";
        ta_state_v = numeric_matrix_int32(2 * number_of_clauses, number_of_features);
    }

    auto ta_state_gen = [&params, &igen](auto & ta_state)
    {
        auto const number_of_clauses = Params::number_of_classifier_clauses(params);
        auto const number_of_features = Params::number_of_features(params);

        for (auto rit = 0; rit < 2 * number_of_clauses; ++rit)
        {
            auto row_data = ta_state.row_data(rit);

            for (auto cit = 0; cit < number_of_features; ++cit)
            {
                row_data[cit] = igen.next(-1, 0);
            }
        }
    };

    std::visit(ta_state_gen, ta_state_v);

    reset_state_cache(state);
}


void reset_state_cache(ClassifierState & state)
{
    auto & cache = state.cache;
    auto & params = state.m_params;

    cache.clause_output.clear();
    cache.clause_output.resize(Params::number_of_classifier_clauses(params));
    cache.label_sum.clear();
    cache.label_sum.resize(Params::number_of_labels(params));
    cache.feedback_to_clauses.clear();
    cache.feedback_to_clauses.resize(Params::number_of_classifier_clauses(params));

    // initialize frand cache
    cache.fcache = ClassifierState::frand_cache_type(state.fgen, 2 * Params::number_of_features(params), state.igen.peek());
}


ClassifierState::ClassifierState(params_t const & params) :
    m_params(params)
{
}


bool ClassifierState::operator==(ClassifierState const & other) const
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
            and cache.feedback_to_clauses.size() == other.cache.feedback_to_clauses.size()
            and cache.clause_output.size() == other.cache.clause_output.size()
            and cache.label_sum.size() == other.cache.label_sum.size()
            ;
    }
}

} // namespace Tsetlino
