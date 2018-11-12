#pragma once

#include <string>
#include <unordered_map>
#include <any>

namespace Tsetlin
{

using config_patch_t = std::unordered_map<std::string, std::any>;
using config_t = std::unordered_map<std::string, std::any>;

//using model_config_t = struct model_config_s
//{
//    int number_of_classes;
//    int number_of_pos_neg_clauses_per_class;
//    int number_of_features;
//    int number_of_states;
//    real_type s;
//    int threshold;
//    int boost_true_positive_feedback;
//
//    int n_jobs;
//    seed_type seed;
//    bool verbose;
//    // learning rate
//    // missing value
//};

config_patch_t const config_patch_from_json(std::string const & json_params, bool verbose=false);
//model_config_t model_config_from_patch(config_patch_t const & patch, bool verbose=false, model_config_t const * config_p=nullptr);

} // namespace Tsetlin
