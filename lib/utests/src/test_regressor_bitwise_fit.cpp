#include "tsetlini.hpp"
#include "tsetlini_types.hpp"
#include "basic_bit_vector.hpp"
#include "basic_bit_vector_companion.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <vector>


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


suite TestRegressorBitwiseFit = []
{


"RegressorBitwise::fit rejects empty input X"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X;
            Tsetlini::response_vector_type y{1, 0, 1, 0};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit rejects empty input y"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y;

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit rejects input X and y with unequal dimensions"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y{1, 0, 0, 1};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit rejects input y with negative response"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y{1, 0, -21};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit rejects input y with response over Threshold"_test = []
{
    Tsetlini::make_regressor_bitwise(R"({"threshold": 15})")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y{1, 15 + 1, 1};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit accepts valid input y with response just below Threshold"_test = []
{
    Tsetlini::make_regressor_bitwise(R"({"threshold": 15})")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y{1, 15, 1};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_OK == rv.first);

            return std::move(reg);
        });
};


"RegressorBitwise::fit accepts valid input"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::response_vector_type y{1, 0, 2};

            auto const rv = reg.fit(X, y);

            expect(that % Tsetlini::StatusCode::S_OK == rv.first);

            return std::move(reg);
        });
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
