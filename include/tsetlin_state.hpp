#pragma once

#include "tsetlin_config.hpp"
#include "tsetlin_types.hpp"

#include <random>

namespace Tsetlin
{

struct ClassifierState;

ClassifierState make_classifier_state(config_patch_t const & config);




struct ClassifierState
{
    config_patch_t const config;

    std::vector<aligned_vector_int> ta_state;
    std::mt19937 gen;

friend ClassifierState make_classifier_state(config_patch_t const & config);

private:
    explicit ClassifierState(config_patch_t const & config);
};




} // namespace Tsetlin
