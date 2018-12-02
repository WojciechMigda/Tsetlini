#include "tsetlin.hpp"

#include <gtest/gtest.h>

namespace
{


TEST(TsetlinClassifier, can_be_created)
{
    auto const clf = Tsetlin::make_classifier();

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_empty_json)
{
    auto const clf = Tsetlin::make_classifier("");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_bad_json)
{
    auto const clf = Tsetlin::make_classifier("[]");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_empty_json_dict)
{
    auto const clf = Tsetlin::make_classifier("{}");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_json_with_unrecognized_param)
{
    auto const clf = Tsetlin::make_classifier(R"({"gotcha": 564})");

    EXPECT_FALSE(clf);
}


}
