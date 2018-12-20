#include "tsetlin.hpp"
#include "tsetlin_status_code.hpp"
#include "tsetlin_state_json.hpp"

#include <gtest/gtest.h>


namespace
{


TEST(ClassifierState, can_be_serialized_and_deserialized_via_json)
{
    Tsetlin::make_classifier()
        .leftMap([](Tsetlin::status_message_t && sm){ throw(sm.second); return sm; })
        .rightMap([](Tsetlin::Classifier && clf1)
        {
            Tsetlin::make_classifier()
                .leftMap([](Tsetlin::status_message_t && sm){ throw(sm.second); return sm; })
                .rightMap([&clf1](Tsetlin::Classifier && clf2)
                {
                    auto _ = clf1.fit({{1, 0, 1, 0}, {1, 1, 1, 0}}, {0, 1}, 2);
                    auto s1 = clf1.read_state();

                    auto s2 = clf2.read_state();

                    auto const jss = to_json_string(s1);
                    from_json_string(s2, jss);

                    EXPECT_EQ(s1, s2);

                    return clf2;
                });

            return clf1;
        });
}


} // anonymous namespace
