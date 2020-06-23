#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{


TEST(TsetlinRegressorClassic, can_be_created)
{
    auto const rgr = Tsetlini::make_regressor_classic();

    EXPECT_TRUE(rgr);
}


} // anonymous namespace
