#include "tsetlin_state.hpp"
#include "config_companion.hpp"

#include <any>
#include <algorithm>
#include <iterator>

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
    ClassifierState state(merge(config_t{default_config}, patch));

    state.gen.seed(std::any_cast<seed_type>(state.config.at("seed")));

    auto & config = state.config;
    auto & ta_state = state.ta_state;

    std::generate_n(std::back_inserter(ta_state), Config::number_of_clauses(config),
        [&config]()
        {
            aligned_vector_int rv;

            std::generate_n(std::back_inserter(rv), Config::number_of_features(config) * 2,
                [&config]()
                {
                // TODO
                    return 1;
//                    return this->igen_.next(Config::number_of_states(config), Config::number_of_states(config) + 1);
                }
            );

            return rv;
        }
    );


    return state;
}


ClassifierState::ClassifierState(config_patch_t const & config) :
    config(config)
{
}

} // namespace Tsetlin
