#include "tsetlini.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_state_json.hpp"
#include "basic_bit_vector_companion.hpp"

#include "gtest/gtest.h"

#include <memory>
#include <algorithm>
#include <vector>


namespace
{


TEST(ClassifierStateClassic, can_be_serialized_and_deserialized_via_json)
{
    Tsetlini::make_classifier_classic()
        .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
        .rightMap([](Tsetlini::ClassifierClassic && clf1)
        {
            Tsetlini::make_classifier_classic()
                .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
                .rightMap([&clf1](Tsetlini::ClassifierClassic && clf2)
                {
                    auto _ = clf1.fit({{1, 0, 1, 0}, {1, 1, 1, 0}}, {0, 1}, 2);
                    auto s1 = clf1.read_state();

                    auto s2 = clf2.read_state();

                    auto const jss = to_json_string(s1);
                    from_json_string(s2, jss);

                    EXPECT_EQ(s1, s2);

                    return std::move(clf2);
                });

            return std::move(clf1);
        });
}


auto to_bitvector = [](std::vector<Tsetlini::aligned_vector_char> const & X)
{
    std::vector<Tsetlini::bit_vector_uint64> rv;
    rv.reserve(X.size());

    std::transform(X.cbegin(), X.cend(), std::back_inserter(rv),
        [](auto const & sample){ return basic_bit_vectors::from_range<std::uint64_t>(sample.cbegin(), sample.cend()); }
    );

    return rv;
};


TEST(ClassifierStateBitwise, can_be_serialized_and_deserialized_via_json)
{
    Tsetlini::make_classifier_bitwise()
        .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
        .rightMap([](Tsetlini::ClassifierBitwise && clf1)
        {
            Tsetlini::make_classifier_bitwise()
                .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
                .rightMap([&clf1](Tsetlini::ClassifierBitwise && clf2)
                {
                    std::vector<Tsetlini::aligned_vector_char> const Xi{{1, 0, 1, 0}, {1, 1, 1, 0}};
                    auto const X = to_bitvector(Xi);

                    auto _ = clf1.fit(X, {0, 1}, 2);
                    auto s1 = clf1.read_state();

                    auto s2 = clf2.read_state();

                    auto const jss = to_json_string(s1);
                    from_json_string(s2, jss);

                    EXPECT_EQ(s1, s2);

                    return std::move(clf2);
                });

            return std::move(clf1);
        });
}


} // anonymous namespace
