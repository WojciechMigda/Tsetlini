#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <vector>


using namespace boost::ut;


void train_classifier(Tsetlini::ClassifierClassic & clf)
{
    std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
    Tsetlini::label_vector_type y{1, 0, 1};

    auto const _ = clf.partial_fit(X, y, Tsetlini::max_number_of_labels_t{3});
}


suite TestClassifierClassicPredictMatrix = []
{


"ClassifierClassic::predict on matrix fails without prior train"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const either = clf.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects empty input X"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::aligned_vector_char> X;

            auto const either = clf.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects input X with rows of unequal length"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const either = clf.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects input X with invalid number of features"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const either = clf.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects input X with non-0/1 values"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const either = clf.predict(X);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict accepts valid input X"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 0, 1}};

            auto const either = clf.predict(X);

            !expect(that % true == either);

            either.rightMap([](auto && y){ expect(that % 3u == y.size()); return std::move(y); });

            return std::move(clf);
        });
};


};


suite TestClassifierClassicPredictSample = []
{


"ClassifierClassic::predict on sample fails without prior train"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            Tsetlini::aligned_vector_char sample{1, 0, 1};

            auto const either = clf.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_NOT_FITTED_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects empty input sample"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::aligned_vector_char sample;

            auto const either = clf.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects input sample with invalid number of features"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::aligned_vector_char sample{1, 0, 1, 0};

            auto const either = clf.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict rejects input sample with non-0/1 values"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::aligned_vector_char sample{1, -1, 2};

            auto const either = clf.predict(sample);

            !expect(that % false == either);

            either.leftMap([](auto && sm){ expect(that % Tsetlini::StatusCode::S_VALUE_ERROR == sm.first); return std::move(sm); });

            return std::move(clf);
        });
};


"ClassifierClassic::predict accepts valid input sample"_test = []
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            train_classifier(clf);

            Tsetlini::aligned_vector_char sample{1, 0, 1};

            auto const either = clf.predict(sample);

            !expect(that % true == either);

            either.rightMap([](auto && label){ expect(that % 0 <= label and label <= 2); return std::move(label); });

            return std::move(clf);
        });
};


};


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
