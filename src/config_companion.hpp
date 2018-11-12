#pragma once

#include "tsetlin_config.hpp"

namespace Tsetlin
{

namespace Config
{

inline
int number_of_pos_neg_clauses_per_class(config_t const & config)
{
    return std::any_cast<int>(config.at("number_of_pos_neg_clauses_per_class"));
}


inline
int number_of_classes(config_t const & config)
{
    return std::any_cast<int>(config.at("number_of_classes"));
}


inline
int number_of_features(config_t const & config)
{
    return std::any_cast<int>(config.at("number_of_features"));
}


inline
int number_of_states(config_t const & config)
{
    return std::any_cast<int>(config.at("number_of_states"));
}


inline
int number_of_clauses_per_class(config_t const & config)
{
    return 2 * number_of_pos_neg_clauses_per_class(config);
}


inline
int number_of_clauses(config_t const & config)
{
    return number_of_clauses_per_class(config) * number_of_classes(config);
}

} // namespace Config

} // namespace Tsetlin
