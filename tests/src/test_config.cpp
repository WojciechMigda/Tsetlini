#include "tsetlin_config.hpp"
#include "tsetlin_types.hpp"

#include <gtest/gtest.h>
#include <any>

namespace
{


TEST(ConfigPatch, can_be_created_from_empty_string_json)
{
    auto const rv = Tsetlin::config_patch_from_json("");

    EXPECT_EQ(0u, rv.size());
}

TEST(ConfigPatch, can_be_created_from_empty_dict_json)
{
    auto const rv = Tsetlin::config_patch_from_json("{}");

    EXPECT_EQ(0u, rv.size());
}

TEST(ConfigPatch, can_be_created_from_invalid_json)
{
    auto const rv = Tsetlin::config_patch_from_json("[]");

    EXPECT_EQ(0u, rv.size());
}

TEST(ConfigPatch, can_be_created_from_malformed_json)
{
    auto const rv = Tsetlin::config_patch_from_json("5\"}");

    EXPECT_EQ(0u, rv.size());
}

TEST(ConfigPatch, can_be_created_from_json_with_one_integer_item)
{
    auto const rv = Tsetlin::config_patch_from_json(R"({"number_of_classes": 2})");

    EXPECT_EQ(1u, rv.size());
    EXPECT_EQ(2, std::any_cast<int>(rv.at("number_of_classes")));
}

TEST(ConfigPatch, can_be_created_from_json_with_one_float_item)
{
    auto const rv = Tsetlin::config_patch_from_json(R"({"s": 3.9})");

    EXPECT_EQ(1u, rv.size());
    EXPECT_FLOAT_EQ(3.9, std::any_cast<Tsetlin::real_type>(rv.at("s")));
}

TEST(ConfigPatch, can_be_created_from_json_with_one_boolean_item)
{
    auto const rv = Tsetlin::config_patch_from_json(R"({"verbose": true})");

    EXPECT_EQ(1u, rv.size());
    EXPECT_EQ(true, std::any_cast<bool>(rv.at("verbose")));
}

TEST(ConfigPatch, can_be_created_from_json_with_unknown_item)
{
    auto const rv = Tsetlin::config_patch_from_json(R"({"foobar": true})");

    EXPECT_EQ(0u, rv.size());
}


TEST(ConfigPatch, can_be_created_from_json_with_full_config)
{
    auto const rv = Tsetlin::config_patch_from_json(R"(
{
"verbose": true,
"number_of_classes": 3,
"number_of_pos_neg_clauses_per_class": 17,
"number_of_features": 9,
"number_of_states": 125,
"s": 6.3 ,
"threshold": 8,
"boost_true_positive_feedback": 1,
"n_jobs": -1,
"seed": 123
}
)");

    EXPECT_EQ(10u, rv.size());
    EXPECT_EQ(true, std::any_cast<bool>(rv.at("verbose")));
    EXPECT_EQ(3, std::any_cast<int>(rv.at("number_of_classes")));
    EXPECT_EQ(17, std::any_cast<int>(rv.at("number_of_pos_neg_clauses_per_class")));
    EXPECT_EQ(9, std::any_cast<int>(rv.at("number_of_features")));
    EXPECT_EQ(125, std::any_cast<int>(rv.at("number_of_states")));
    EXPECT_EQ(8, std::any_cast<int>(rv.at("threshold")));
    EXPECT_EQ(1, std::any_cast<int>(rv.at("boost_true_positive_feedback")));
    EXPECT_FLOAT_EQ(6.3, std::any_cast<Tsetlin::real_type>(rv.at("s")));
    EXPECT_EQ(123u, std::any_cast<Tsetlin::seed_type>(rv.at("seed")));
}


//static const Tsetlin::model_config_t default_config = {2, 5, 2, 100, 4.0, 15, 0, -1, 1, false};
//
//TEST(ModelConfig, is_not_patched_with_empty_patch)
//{
//    auto const rv = Tsetlin::model_config_from_patch({}, false, &default_config);
//
//    EXPECT_EQ(2, rv.number_of_classes);
//    EXPECT_EQ(5, rv.number_of_pos_neg_clauses_per_class);
//    EXPECT_EQ(2, rv.number_of_features);
//    EXPECT_EQ(100, rv.number_of_states);
//    EXPECT_FLOAT_EQ(4.f, rv.s);
//    EXPECT_EQ(15, rv.threshold);
//    EXPECT_EQ(0, rv.boost_true_positive_feedback);
//    EXPECT_EQ(-1, rv.n_jobs);
//    EXPECT_EQ(1, rv.seed);
//    EXPECT_EQ(false, rv.verbose);
//}
//
//TEST(ModelConfig, is_patched_with_full_patch)
//{
//    auto const rv = Tsetlin::model_config_from_patch(
//        {
//            {"number_of_classes", std::any(7)},
//            {"number_of_pos_neg_clauses_per_class", std::any(7)},
//            {"number_of_features", std::any(7)},
//            {"number_of_states", std::any(7)},
//            {"s", std::any(7.7f)},
//            {"threshold", std::any(7)},
//            {"boost_true_positive_feedback", std::any(1)},
//            {"n_jobs", std::any(7)},
//            {"seed", std::any(7u)},
//            {"verbose", std::any(true)},
//        },
//        false, &default_config);
//
//    EXPECT_EQ(7, rv.number_of_classes);
//    EXPECT_EQ(7, rv.number_of_pos_neg_clauses_per_class);
//    EXPECT_EQ(7, rv.number_of_features);
//    EXPECT_EQ(7, rv.number_of_states);
//    EXPECT_FLOAT_EQ(7.7f, rv.s);
//    EXPECT_EQ(7, rv.threshold);
//    EXPECT_EQ(1, rv.boost_true_positive_feedback);
//    EXPECT_EQ(7, rv.n_jobs);
//    EXPECT_EQ(7, rv.seed);
//    EXPECT_EQ(true, rv.verbose);
//}
//
//TEST(ModelConfig, is_not_patched_with_bad_domain_value__number_of_classes)
//{
//    auto const rv = Tsetlin::model_config_from_patch(
//        {
//            {"number_of_classes", std::any(0)}
//        },
//        false, &default_config);
//
//    EXPECT_EQ(2, rv.number_of_classes);
//}

}
