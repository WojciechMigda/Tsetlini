#pragma once

#include "frand_cache.hpp"
#include "mt.hpp"

#include "tsetlin_config.hpp"
#include "tsetlin_types.hpp"

#include <random>
#include <vector>

namespace Tsetlin
{

struct ClassifierState;

ClassifierState make_classifier_state(config_patch_t const & config);




struct ClassifierState
{
    using frand_cache_type = frand_cache<FRNG, alignment>;

    struct Cache
    {
        feedback_vector_type feedback_to_clauses; // shape=(number of clauses)
        aligned_vector_char clause_output; // shape=(number of clauses)
        aligned_vector_int class_sum; // shape=(number of classes)

        frand_cache_type fcache;
    };

    config_patch_t const config;

    std::vector<aligned_vector_int> ta_state;

    mutable std::vector<Cache> vcache; // shape=(n_jobs)

    std::mt19937 gen;
    IRNG igen;
    FRNG fgen;

friend ClassifierState make_classifier_state(config_patch_t const & config);

private:
    explicit ClassifierState(config_patch_t const & config);
};




} // namespace Tsetlin
