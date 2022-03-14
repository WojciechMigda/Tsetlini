#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <vector>


using namespace boost::ut;


void train_regressor(Tsetlini::RegressorClassic & reg)
{
    std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
    Tsetlini::response_vector_type y{1, 0, 1};

    auto const _ = reg.partial_fit(X, y);
}


suite TestRegressorClassicPredictMatrix = []
{


"RegressorClassic::predict on matrix fails without prior train"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects empty input X"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::aligned_vector_char> X;

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects input X with invalid number of features"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects input X with non-0/1 values"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const either = reg.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict accepts valid input X"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 0, 1}};

            auto const either = reg.predict(X);

            !expect(that % true == either);

            either.rightMap([](auto && y){ expect(that % 3u == y.size()); return std::move(y); });

            return std::move(reg);
        });
};


};


suite TestRegressorClassicPredictSample = []
{


"RegressorClassic::predict on sample fails without prior train"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            Tsetlini::aligned_vector_char sample{1, 0, 1};

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects empty input sample"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::aligned_vector_char sample;

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects input sample with invalid number of features"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::aligned_vector_char sample{1, 0, 1, 0};

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict rejects input sample with non-0/1 values"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::aligned_vector_char sample{1, -1, 2};

            auto const either = reg.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(reg);
        });
};


"RegressorClassic::predict accepts valid input sample"_test = []
{
    Tsetlini::make_regressor_classic()
        .rightMap(
        [](auto && reg)
        {
            train_regressor(reg);

            Tsetlini::aligned_vector_char sample{1, 0, 1};

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
