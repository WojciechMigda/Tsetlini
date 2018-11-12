#include "tsetlin_state.hpp"

#include <gtest/gtest.h>

namespace
{


TEST(ClassifierState, can_be_created)
{
    ASSERT_NO_FATAL_FAILURE(
        auto const state = Tsetlin::make_classifier_state(Tsetlin::config_patch_from_json(""));
    );
}


} // namespace
