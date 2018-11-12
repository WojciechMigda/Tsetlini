#define LOG_MODULE "tsetlin"
#include "logger.hpp"

#include "tsetlin_config.hpp"
#include "tsetlin_types.hpp"
#include "json.hpp"

#include <string>

namespace Tsetlin
{

using json = nlohmann::json;

//static const model_config_t default_config = {2, 5, 2, 100, 4.0, 15, 0, -1, 1, false};


/**
 *******************************************************************************
 *   @brief json::parse wrapper which contains exceptions
 *******************************************************************************
 */
static
json json_parse(std::string const & json_params, bool verbose=false)
{
    json rv;

    try
    {
        rv = json::parse(json_params);
    }
    catch (const json::exception & ex)
    {
        LOG(warn) << ex.what() << '\n';
    }

    return rv;
}


config_patch_t const config_patch_from_json(std::string const & json_params, bool verbose)
{
    config_patch_t rv;

    auto const params = json_parse(json_params, verbose);

    for (auto const & kv : params.items())
    {
        auto const key = kv.key();
        auto const value = kv.value();

        if ((key == "number_of_classes") or
            (key == "number_of_pos_neg_clauses_per_class") or
            (key == "number_of_features") or
            (key == "number_of_states") or
            (key == "boost_true_positive_feedback") or
            (key == "threshold") or
            (key == "n_jobs")
            )
        {
            rv[key] = value.get<int>();
        }
        else if (key == "seed")
        {
            rv[key] = value.get<seed_type>();
        }
        else if (key == "s")
        {
            rv[key] = value.get<real_type>();
        }
        else if (key == "verbose")
        {
            rv[key] = value.get<bool>();
        }
        else
        {
            LOG(warn) << "Unknown key [" << key << "] in config\n";
        }
    }

    return rv;
}

//model_config_t model_config_from_patch(config_patch_t const & patch, bool verbose, model_config_t const * config_p)
//{
//    model_config_t rv = config_p ? *config_p : default_config;
//
//    for (auto const & [key, value]: patch)
//    {
//        if (key == "number_of_classes")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v > 0)
//            {
//                rv.number_of_classes = v;
//            }
//        }
//        else if (key == "number_of_pos_neg_clauses_per_class")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v > 0)
//            {
//                rv.number_of_pos_neg_clauses_per_class = v;
//            }
//        }
//        else if (key == "number_of_features")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v > 0)
//            {
//                rv.number_of_features = v;
//            }
//        }
//        else if (key == "number_of_states")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v > 0)
//            {
//                rv.number_of_states = v;
//            }
//        }
//        else if (key == "boost_true_positive_feedback")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v == 0 or v == 1)
//            {
//                rv.boost_true_positive_feedback = v;
//            }
//        }
//        else if (key == "threshold")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v > 0)
//            {
//                rv.threshold = v;
//            }
//        }
//        else if (key == "n_jobs")
//        {
//            auto const v = std::any_cast<int>(value);
//
//            if (v > 0 or v == -1)
//            {
//                rv.n_jobs = v;
//            }
//        }
//        else if (key == "s")
//        {
//            auto const v = std::any_cast<real_type>(value);
//
//            if (v >= 0.f)
//            {
//                rv.s = v;
//            }
//        }
//        else if (key == "verbose")
//        {
//            rv.verbose = std::any_cast<bool>(value);
//        }
//        else if (key == "seed")
//        {
//            rv.seed = std::any_cast<unsigned int>(value);
//        }
//    }
//
//    return rv;
//}

} // namespace Tsetlin
