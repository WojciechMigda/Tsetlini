#include "tsetlini.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_state_json.hpp"
#include "basic_bit_vector.hpp"
#include "basic_bit_vector_companion.hpp"
#include "estimator_state_fwd.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstdint>


using namespace boost::ut;

// helper
auto to_bitvector = [](std::vector<Tsetlini::aligned_vector_char> const & X)
{
    std::vector<Tsetlini::bit_vector_uint64> rv;
    rv.reserve(X.size());

    std::transform(X.cbegin(), X.cend(), std::back_inserter(rv),
        [](auto const & sample){ return basic_bit_vectors::from_range<std::uint64_t>(sample.cbegin(), sample.cend()); }
    );

    return rv;
};


suite TestRegressorStateJsonSerialization = []
{


"RegressorStateClassic can be serialized and deserialized via json"_test = []
{
    // create working regressor
    Tsetlini::make_regressor_classic()
        .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
        .rightMap([](Tsetlini::RegressorClassic && reg1)
        {
            // create another dummy regressor
            Tsetlini::make_regressor_classic()
                .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
                .rightMap([&reg1](Tsetlini::RegressorClassic && reg2)
                {
                    // initialize state by calling fit() on our working regressor
                    auto _ = reg1.fit({{1, 0, 1, 0}, {1, 1, 1, 0}}, {0, 1}, 2);
                    // and extract the state
                    auto s1 = reg1.clone_state();

                    // extract the state from the dummy one, it'll be overwritten
                    auto s2 = reg2.clone_state();

                    // serialize our working state
                    auto const jss = to_json_string(*s1);
                    // and de-serialize it onto the dummy one
                    from_json_string(*s2, jss);

                    expect(that % true == equal(*s1, *s2)) << "(De-)serialization failed";

                    return std::move(reg2);
                });

            return std::move(reg1);
        });
};


"RegressorStateBitwise can be serialized and deserialized via json"_test = []
{
    // create working regressor
    Tsetlini::make_regressor_bitwise()
        .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
        .rightMap([](Tsetlini::RegressorBitwise && reg1)
        {
        // create another dummy regressor
            Tsetlini::make_regressor_bitwise()
                .leftMap([](Tsetlini::status_message_t && sm){ throw(sm.second); return std::move(sm); })
                .rightMap([&reg1](Tsetlini::RegressorBitwise && reg2)
                {
                    std::vector<Tsetlini::aligned_vector_char> const Xi{{1, 0, 1, 0}, {1, 1, 1, 0}};
                    auto const X = to_bitvector(Xi);

                    // initialize state by calling fit() on our working regressor
                    auto _ = reg1.fit(X, {0, 1}, 2);
                    // and extract the state
                    auto s1 = reg1.clone_state();

                    // extract the state from the dummy one, it'll be overwritten
                    auto s2 = reg2.clone_state();

                    // serialize our working state
                    auto const jss = to_json_string(*s1);
                    // and de-serialize it onto the dummy one
                    from_json_string(*s2, jss);

                    expect(that % true == equal(*s1, *s2)) << "(De-)serialization failed";

                    return std::move(reg2);
                });

            return std::move(reg1);
        });
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
