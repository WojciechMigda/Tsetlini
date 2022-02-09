#include "tsetlini.hpp"
#include "tsetlini_types.hpp"
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


void train_classifier(Tsetlini::ClassifierBitwise & clf)
{
    std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
    Tsetlini::label_vector_type y{1, 0, 1};

    auto const _ = clf.partial_fit(X, y, 3);
}


suite TestClassifierBitwiseDecisionFunctionMatrix = []
{


"ClassifierBitwise::decision_function on matrix fails without prior train"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});

            auto const either = clf.decision_function(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function rejects empty input X"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X;

            auto const either = clf.decision_function(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0}, {0, 0, 0}});

            auto const either = clf.decision_function(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function rejects input X with invalid number of features"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}});

            auto const either = clf.decision_function(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function accepts valid input X"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 1}, {0, 0, 1}});

            auto const either = clf.decision_function(X);

            !expect(that % true == either);

            either.rightMap([](auto && y){ expect(that % 3u == y.size()); return std::move(y); });
            either.rightMap([](auto && y){ expect(that % 3u == y.front().size()); return std::move(y); });

            return std::move(clf);
        });
};


};


suite TestClassifierBitwiseDecisionFunctionSample = []
{


"ClassifierBitwise::decision_function on sample fails without prior train"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            Tsetlini::bit_vector_uint64 sample = to_bitvector({1, 0, 1});

            auto const either = clf.decision_function(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function rejects empty input sample"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::bit_vector_uint64 sample;

            auto const either = clf.decision_function(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function rejects input sample with invalid number of features"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::bit_vector_uint64 sample = to_bitvector({1, 0, 1, 0});

            auto const either = clf.decision_function(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierBitwise::decision_function accepts valid input sample"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::bit_vector_uint64 sample = to_bitvector({1, 0, 1});

            auto const either = clf.decision_function(sample);

            !expect(that % true == either);

            either.rightMap([](auto && scores){ expect(that % 3u == scores.size()); return std::move(scores); });

            return std::move(clf);
        });
};


};


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
