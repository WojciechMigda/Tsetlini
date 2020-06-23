#include "tsetlini.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_state_json.hpp"

#include <gtest/gtest.h>


namespace
{


TEST(ClassifierState, can_be_serialized_and_deserialized_via_json)
{
    Tsetlini::make_classifier_classic()
        .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return sm; })
        .rightMap([](Tsetlini::ClassifierClassic && clf1)
        {
            Tsetlini::make_classifier_classic()
                .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return sm; })
                .rightMap([&clf1](Tsetlini::ClassifierClassic && clf2)
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
