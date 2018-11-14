#define LOG_MODULE "tsetlin-core"
#include "logger.hpp"

#include "tsetlin_state.hpp"
#include "config_companion.hpp"

#include <any>
#include <algorithm>
#include <iterator>
#include <thread>

namespace Tsetlin
{

static const config_t default_config =
{
    {"number_of_classes", std::any(2)},
    {"number_of_pos_neg_clauses_per_class", std::any(5)},
    {"number_of_features", std::any(2)},
    {"number_of_states", std::any(100)},
    {"s", std::any(2.0f)},
    {"threshold", std::any(15)},
    {"boost_true_positive_feedback", std::any(0)},
    {"n_jobs", std::any(-1)},
    {"seed", std::any(0uL)},
    {"verbose", std::any(false)},
};


template<typename LHS, typename RHS>
config_t merge(LHS && lhs, RHS && rhs)
{
    config_patch_t rv(std::forward<RHS>(rhs));

    rv.merge(std::forward<LHS>(lhs));

    return rv;
}


ClassifierState make_classifier_state(config_patch_t const & patch)
{
    auto merged_config = merge(config_t{default_config}, patch);

    if (Config::n_jobs(merged_config) == -1)
    {
        merged_config.at("n_jobs") = std::max<int>(1, std::thread::hardware_concurrency());
    }

    ClassifierState state(merged_config);

    ////////////////////////////////////////////////////////////////////////////

    auto & config = state.config;

    auto const verbose = Config::verbose(config);

    state.gen.seed(Config::seed(config));
    state.igen.init(Config::seed(config));
    state.fgen.init(Config::seed(config));

    LOG(info) << "number_of_classes: " << Config::number_of_classes(config) << '\n';
    LOG(info) << "number_of_clauses: " << Config::number_of_clauses(config) << '\n';
    LOG(info) << "number_of_features: " << Config::number_of_features(config) << '\n';
    LOG(info) << "s: " << Config::s(config) << '\n';
    LOG(info) << "number_of_states: " << Config::number_of_states(config) << '\n';
    LOG(info) << "threshold: " << Config::threshold(config) << '\n';
    LOG(info) << "n_jobs: " << Config::n_jobs(config) << '\n';
    LOG(info) << "seed: " << Config::seed(config) << '\n';

    // convenience reference variables
    auto & ta_state = state.ta_state;
    auto & igen = state.igen;
    auto & cache = state.cache;

    std::generate_n(std::back_inserter(ta_state), Config::number_of_clauses(config),
        [&config, &igen]()
        {
            aligned_vector_int rv;

            std::generate_n(std::back_inserter(rv), Config::number_of_features(config) * 2,
                [&config, &igen]()
                {
                    return igen.next(Config::number_of_states(config), Config::number_of_states(config) + 1);
                }
            );

            return rv;
        }
    );

    cache.clause_output = aligned_vector_char(Config::number_of_clauses(config), 0);
    cache.class_sum = aligned_vector_int(Config::number_of_classes(config), 0);
    cache.feedback_to_clauses = feedback_vector_type(Config::number_of_clauses(config), 0);

    // initialize frand caches instances for use by all thread jobs
    cache.fcache.reserve(Config::n_jobs(config));
    for (auto it = 0; it < Config::n_jobs(config); ++it)
    {
        cache.fcache.emplace_back(2 * Config::number_of_features(config), igen.next());
    }

    return state;
}


ClassifierState::ClassifierState(config_patch_t const & config) :
    config(config)
{
}

} // namespace Tsetlin
