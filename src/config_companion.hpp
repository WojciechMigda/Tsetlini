#pragma once

#include "tsetlin_config.hpp"

#include <variant>

namespace Tsetlin
{

namespace Config
{


inline
int n_jobs(config_t const & config)
{
    return std::get<int>(config.at("n_jobs"));
}


inline
seed_type seed(config_t const & config)
{
    return std::get<seed_type>(config.at("seed"));
}


inline
bool verbose(config_t const & config)
{
    return std::get<bool>(config.at("verbose"));
}


inline
int number_of_pos_neg_clauses_per_class(config_t const & config)
{
    return std::get<int>(config.at("number_of_pos_neg_clauses_per_class"));
}


inline
int number_of_classes(config_t const & config)
{
    return std::get<int>(config.at("number_of_classes"));
}


inline
int number_of_features(config_t const & config)
{
    return std::get<int>(config.at("number_of_features"));
}


inline
real_type s(config_t const & config)
{
    return std::get<real_type>(config.at("s"));
}


inline
int threshold(config_t const & config)
{
    return std::get<int>(config.at("threshold"));
}


inline
int boost_true_positive_feedback(config_t const & config)
{
    return std::get<int>(config.at("boost_true_positive_feedback"));
}


inline
int number_of_states(config_t const & config)
{
    return std::get<int>(config.at("number_of_states"));
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
