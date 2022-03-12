#include "tsetlini.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_strong_params.hpp"
#include "basic_bit_vector.hpp"
#include "basic_bit_vector_companion.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
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


suite TestClassifierBitwisePartialFit = []
{


"ClassifierBitwise::partial_fit on untrained classifier rejects empty input X"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X;
            Tsetlini::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects empty input y"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X with first padding bit set to 1"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[1].set(4);

            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X with last padding bit set to 1"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[2].set(Tsetlini::bit_vector_uint64::block_bits - 1);

            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X with some padding bits set to 1"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[0].set(12);
            X[0].set(18);
            X[1].set(7);

            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input X and y with unequal dimensions"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier rejects input y with negative label"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{3});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on untrained classifier accepts valid input"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 2};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{3});

            expect(that % Tsetlini::StatusCode::S_OK == rv.first);

            return std::move(clf);
        });
};


};


void train_classifier(Tsetlini::ClassifierBitwise & clf)
{
    std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
    Tsetlini::label_vector_type y{1, 0, 1};

    auto const _ = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{3});
}


suite TestClassifierBitwisePartialFitOnTrained = []
{


"ClassifierBitwise::partial_fit on trained classifier rejects empty input X"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X;
            Tsetlini::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects empty input y"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input X with first padding bit set to 1"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[1].set(4);

            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input X with last padding bit set to 1"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[2].set(Tsetlini::bit_vector_uint64::block_bits - 1);

            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input X with some padding bits set to 1"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            X[0].set(12);
            X[0].set(18);
            X[1].set(7);

            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input X with invalid number of features"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}});
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input X and y with unequal dimensions"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{2});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects input y with negative label"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{3});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier accepts valid input"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            Tsetlini::label_vector_type y{1, 0, 2};

            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{3});

            expect(that % Tsetlini::StatusCode::S_OK == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier rejects OOB labels"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            // train with 3 labels
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            // I pass labels from am invalid 4-label range
            Tsetlini::label_vector_type y{1, 0, 3};

            // and I pass invalid max_number_of_labels = 4
            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{4});

            expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);

            return std::move(clf);
        });
};


"ClassifierBitwise::partial_fit on trained classifier does not reject invalid max_number_of_labels when actual labels are in range"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            // train with 3 labels
            train_classifier(clf);

            std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
            // I pass labels from a valid 3-label range
            Tsetlini::label_vector_type y{1, 0, 2};

            // but I pass invalid max_number_of_labels = 4
            auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{4});

            expect(that % Tsetlini::StatusCode::S_OK == rv.first);

            return std::move(clf);
        });
};


"Passing too big max_number_of_labels to ClassifierBitwise::partial_fit"
" on trained classifier doesn't affect labels validation by subsequent partial_fit"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .rightMap(
        [](auto && clf)
        {
            // train with 3 labels
            train_classifier(clf);

            {
                std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
                // I pass labels from a valid 3-label range
                Tsetlini::label_vector_type y{1, 0, 2};

                // and I pass invalid max_number_of_labels = 4
                auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{4});

                expect(that % Tsetlini::StatusCode::S_OK == rv.first);
            }

            {
                std::vector<Tsetlini::bit_vector_uint64> X = to_bitvector({{1, 0, 1}, {1, 0, 0}, {0, 0, 0}});
                // then when I pass labels from an invalid 4-label range
                Tsetlini::label_vector_type y{1, 0, 3};

                // they are still rejected
                auto const rv = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{4});

                expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == rv.first);
            }

            return std::move(clf);
        });
};


};


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
