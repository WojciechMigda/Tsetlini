#define LOG_MODULE "tsetlin-core"
#include "logger.hpp"

#include "tsetlin_state.hpp"
#include "tsetlin_types.hpp"
#include "params_companion.hpp"
#include "tsetlin_params.hpp"

#include <algorithm>
#include <iterator>
#include <thread>
#include <random>


namespace std
{
// this is needed for std::variant comparison, probably illegal. TODO
static constexpr
bool operator==(nullopt_t const & lhs, nullopt_t const & rhs){ return true; }

}


namespace Tsetlin
{


void initialize_state(ClassifierState & state)
{
    auto & params = state.m_params;

    auto const verbose = Params::verbose(params);

    state.igen.init(Params::random_state(params));
    state.fgen.init(Params::random_state(params));

    LOG(info) << "number_of_labels: " << Params::number_of_labels(params) << '\n';
    LOG(info) << "number_of_clauses: " << Params::number_of_clauses(params) << '\n';
    LOG(info) << "number_of_features: " << Params::number_of_features(params) << '\n';
    LOG(info) << "s: " << Params::s(params) << '\n';
    LOG(info) << "number_of_states: " << Params::number_of_states(params) << '\n';
    LOG(info) << "threshold: " << Params::threshold(params) << '\n';
    LOG(info) << "n_jobs: " << Params::n_jobs(params) << '\n';
    LOG(info) << "random_state: " << Params::random_state(params) << '\n';

    // convenience reference variables
    auto & ta_state = state.ta_state;
    auto & igen = state.igen;
    auto & cache = state.cache;

    std::generate_n(std::back_inserter(ta_state), Params::number_of_clauses(params),
        [&params, &igen]()
        {
            aligned_vector_int rv;

            std::generate_n(std::back_inserter(rv), Params::number_of_features(params) * 2,
                [&params, &igen]()
                {
                    return igen.next(Params::number_of_states(params), Params::number_of_states(params) + 1);
                }
            );

            return rv;
        }
    );

    cache.clause_output = aligned_vector_char(Params::number_of_clauses(params), 0);
    cache.label_sum = aligned_vector_int(Params::number_of_labels(params), 0);
    cache.feedback_to_clauses = feedback_vector_type(Params::number_of_clauses(params), 0);

    // initialize frand caches instances for use by all thread jobs
    cache.fcache.reserve(Params::n_jobs(params));
    for (auto it = 0; it < Params::n_jobs(params); ++it)
    {
        cache.fcache.emplace_back(2 * Params::number_of_features(params), igen.next());
    }
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
            ;
    }
}

} // namespace Tsetlin
