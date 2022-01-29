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


suite TestClassifierBitwisePartialFit = []
{


"ClassifierBitwise::partial_fit on untrained classifier rejects empty input X"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X;
            Tsetlini::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects empty input y"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, 2);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X and y with unequal dimensions"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, 2);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input y with negative label"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, 2);

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier accepts valid input"_test = []
{
    Tsetlini::make_classifier_bitwise("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 2};

            auto const rv = clf.partial_fit(X, y, 2);

            expect(that % Tsetlini::StatusCode::S_OK == rv.first);

            return std::move(clf);
        });
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
