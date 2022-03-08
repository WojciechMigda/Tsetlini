#include "tsetlini.hpp"
#include "tsetlini_types.hpp"
#include "basic_bit_vector.hpp"
#include "basic_bit_vector_companion.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <vector>
#include <cstdint>


using namespace boost::ut;


// helpers
Tsetlini::bit_vector_uint64 to_bitvector(Tsetlini::aligned_vector_char const & sample)
{
    return basic_bit_vectors::from_range<std::uint64_t>(sample.cbegin(), sample.cend());
}


std::vector<Tsetlini::bit_vector_uint64> to_bitvector(std::vector<Tsetlini::aligned_vector_char> const & X)
{
    std::vector<Tsetlini::bit_vector_uint64> rv;
    rv.reserve(X.size());

    std::transform(X.cbegin(), X.cend(), std::back_inserter(rv),
        [](auto const & sample){ return to_bitvector(sample); }
    );

    return rv;
}


void train_regressor(Tsetlini::RegressorBitwise & reg)
{
    std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
    Tsetlini::response_vector_type y{1, 0, 1};

    auto const _ = reg.partial_fit(X, y);
}


suite TestRegressorBitwisePredictMatrix = []
{


"RegressorBitwise::predict on matrix fails without prior train"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects empty input X"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X;

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0}, {0, 0, 0}});

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects input X with first padding bit set to 1"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[1].set(4);

            auto const rv = reg.predict(X);

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects input X with last padding bit set to 1"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[2].set(Tsetlini::bit_vector_uint64::block_bits - 1);

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects input X some padding bits set to 1"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[0].set(12);
            X[0].set(18);
            X[1].set(7);

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects input X with invalid number of features"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}});

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict accepts valid input X"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 1}, {0, 0, 1}});

            auto const either = reg.predict(X);

            !expect(that % true == either);

            either.rightMap([](auto && y){ expect(that % 3u == y.size()); return std::move(y); });

            return std::move(reg);
        });
};


};


suite TestRegressorBitwisePredictSample = []
{


"RegressorBitwise::predict on sample fails without prior train"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            Tsetlini::bit_vector_uint64 sample = to_bitvector({1, 0, 1});

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects empty input sample"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::bit_vector_uint64 sample;

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict rejects input sample with invalid number of features"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::bit_vector_uint64 sample = to_bitvector({1, 0, 1, 0});

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorBitwise::predict accepts valid input sample"_test = []
{
    Tsetlini::make_regressor_bitwise("{}")
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::bit_vector_uint64 sample = to_bitvector({1, 0, 1});

            auto const either = reg.predict(sample);

            !expect(that % true == either);

            either.rightMap([](auto && response){ expect(that % 0 <= response and response <= 15); return std::move(response); });

            return std::move(reg);
        });
};


};


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
