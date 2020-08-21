#include "tsetlini.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_state_json.hpp"

#include <gtest/gtest.h>


namespace
{


TEST(RegressorStateClassic, can_be_serialized_and_deserialized_via_json)
{
    Tsetlini::make_regressor_classic()
        .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return sm; })
        .rightMap([](Tsetlini::RegressorClassic && reg1)
        {
            Tsetlini::make_regressor_classic()
                .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return sm; })
                .rightMap([&reg1](Tsetlini::RegressorClassic && reg2)
                {
                    auto _ = reg1.fit({{1, 0, 1, 0}, {1, 1, 1, 0}}, {0, 1}, 2);
                    auto s1 = reg1.read_state();

                    auto s2 = reg2.read_state();

                    auto const jss = to_json_string(s1);
                    from_json_string(s2, jss);

                    EXPECT_EQ(s1, s2);

                    return reg2;
                });

            return reg1;
        });
}


} // anonymous namespace
