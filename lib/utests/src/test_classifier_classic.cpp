#include "tsetlini.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_strong_params.hpp"
#include "either.hpp"
#include "params_companion.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <limits>
#include <cmath>


using namespace boost::ut;
using namespace std::string_literals;


suite TestClassifierClassicArgs = []
{


"ClassifierClassic can be created from default arguments"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic();

    expect(that % true == !!clf);
};


"ClassifierClassic created from arguments can be move-assigned"_test = []
{
    std::optional<Tsetlini::ClassifierClassic> maybe_estimator;

    auto const rv = Tsetlini::make_classifier_classic()
        .rightFlatMap(
        [&](auto && est)
        {
            maybe_estimator = std::move(est);

            return Tsetlini::Either<Tsetlini::status_message_t, int>::rightOf(0);
        });

    expect(that % true == maybe_estimator.has_value());
};


"ClassifierClassic can be created with counting_type argument set to int8"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::counting_type_t{"int8"})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % "int8"s == value_of(Tsetlini::Params::counting_type(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with counting_type argument set to int16"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::counting_type_t{"int16"})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % "int16"s == value_of(Tsetlini::Params::counting_type(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with counting_type argument set to int32"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::counting_type_t{"int32"})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % "int32"s == value_of(Tsetlini::Params::counting_type(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with counting_type argument set to auto"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::counting_type_t{"auto"})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % "auto"s == value_of(Tsetlini::Params::counting_type(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with clause_output_tile_size argument set to 16"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::clause_output_tile_size_t{16})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 16 == value_of(Tsetlini::Params::clause_output_tile_size(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with clause_output_tile_size argument set to 32"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::clause_output_tile_size_t{32})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 32 == value_of(Tsetlini::Params::clause_output_tile_size(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with clause_output_tile_size argument set to 64"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::clause_output_tile_size_t{64})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 64 == value_of(Tsetlini::Params::clause_output_tile_size(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with clause_output_tile_size argument set to 128"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::clause_output_tile_size_t{128})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 128 == value_of(Tsetlini::Params::clause_output_tile_size(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with clause_output_tile_size argument set to 24"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::clause_output_tile_size_t{24});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with n_jobs argument set to 2"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::number_of_jobs_t{2})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 2 == value_of(Tsetlini::Params::n_jobs(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with number_of_jobs argument set to -1"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::number_of_jobs_t>;

    Tsetlini::make_classifier_classic(Tsetlini::number_of_jobs_t{-1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % underlying_type(std::thread::hardware_concurrency()) == value_of(Tsetlini::Params::n_jobs(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with number_of_jobs argument set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_jobs_t{0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_jobs argument set to -2"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_jobs_t{-2});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with number_of_physical_classifier_clauses_per_label argument set to 4"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{4})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 4 == value_of(Tsetlini::Params::number_of_physical_classifier_clauses_per_label(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with number_of_physical_classifier_clauses_per_label argument set to 3"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{3});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_physical_classifier_clauses_per_label argument set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{2});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_physical_classifier_clauses_per_label argument set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{1});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_physical_classifier_clauses_per_label argument set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_physical_classifier_clauses_per_label argument set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{-1});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_physical_classifier_clauses_per_label argument set to -4"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_physical_classifier_clauses_per_label_t{-4});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with number_of_states set argument to 1"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::number_of_states_t{1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 1 == value_of(Tsetlini::Params::number_of_states(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with number_of_states argument set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_states_t{0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with number_of_states argument set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::number_of_states_t{-1});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with boost_true_positive_feedback argument set to true"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::boost_tpf_t{true})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % true == value_of(Tsetlini::Params::boost_true_positive_feedback(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with boost_true_positive_feedback argument set to false"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::boost_tpf_t{false})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % false == value_of(Tsetlini::Params::boost_true_positive_feedback(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with threshold argument set to 1"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::threshold_t{1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 1 == value_of(Tsetlini::Params::threshold(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with threshold argument set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::threshold_t{0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with threshold argument set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::threshold_t{-1});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with max_weight argument set to 1"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::max_weight_t{1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 1 == value_of(Tsetlini::Params::max_weight(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with max_weight argument set to 10"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::max_weight_t{10})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 10 == value_of(Tsetlini::Params::max_weight(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with max_weight argument set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::max_weight_t{0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with max_weight argument set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::max_weight_t{-1});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with verbosity argument set to true"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::verbosity_t{true})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % true == value_of(Tsetlini::Params::verbose(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with verbosity argument set to false"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::verbosity_t{false})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % false == value_of(Tsetlini::Params::verbose(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with weighted argument set to true"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::weighted_flag_t{true})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % true == value_of(Tsetlini::Params::weighted(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with weighted argument set to false"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::weighted_flag_t{false})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % false == value_of(Tsetlini::Params::weighted(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic cannot be created with specificity argument set to -1.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::specificity_t{-1.0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with specificity argument set to 0.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::specificity_t{0.0});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with specificity argument set to 1.0 - epsilon"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::specificity_t>;

    auto const clf = Tsetlini::make_classifier_classic(
        Tsetlini::specificity_t{underlying_type{1} - std::numeric_limits<underlying_type>::epsilon()});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with specificity argument set to -inf"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::specificity_t>;

    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::specificity_t{-std::numeric_limits<underlying_type>::infinity()});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with specificity argument set to +inf"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::specificity_t>;

    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::specificity_t{std::numeric_limits<underlying_type>::infinity()});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with specificity argument set to +NaN"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::specificity_t{NAN});

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created with specificity argument set to -NaN"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(Tsetlini::specificity_t{-NAN});

    expect(that % false == !!clf);
};


"ClassifierClassic can be created with specificity argument set to 1.0"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::specificity_t>;

    Tsetlini::make_classifier_classic(Tsetlini::specificity_t{1.0})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % underlying_type{1} == value_of(Tsetlini::Params::s(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with specificity argument set to 3.14"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::specificity_t>;

    Tsetlini::make_classifier_classic(Tsetlini::specificity_t{3.14})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % underlying_type{3.14} == value_of(Tsetlini::Params::s(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with random_seed argument set to 1"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::random_seed_t{1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 1u == value_of(Tsetlini::Params::random_state(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with random_seed argument set to 1234"_test = []
{
    Tsetlini::make_classifier_classic(Tsetlini::random_seed_t{1234})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % 1234u == value_of(Tsetlini::Params::random_state(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created without random_seed argument"_test = []
{
    Tsetlini::make_classifier_classic()
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierClassic && clf)
            {
                expect(that % true == (clf.read_params().find("random_state") != clf.read_params().cend()));

                return std::move(clf);
            })
        ;
};


"ClassifierClassic can be created with arbitrarily ordered arguments"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic(
        Tsetlini::number_of_jobs_t{3},
        Tsetlini::verbosity_t{true},
        Tsetlini::number_of_physical_classifier_clauses_per_label_t{20},
        Tsetlini::number_of_states_t{125},
        Tsetlini::specificity_t{6.3},
        Tsetlini::threshold_t{8},
        Tsetlini::weighted_flag_t{true},
        Tsetlini::max_weight_t{7},
        Tsetlini::boost_tpf_t{1},
        Tsetlini::counting_type_t{"int32"},
        Tsetlini::clause_output_tile_size_t{32},
        Tsetlini::random_seed_t{123}
    );

    expect(that % true == !!clf);
};


}; // suite TestClassifierClassicArgs


////////////////////////////////////////////////////////////////////////////////


suite TestClassifierClassicJson = []
{


"ClassifierClassic can be created from default JSON"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json();

    expect(that % true == !!clf);
};


"ClassifierClassic can be move-assigned"_test = []
{
    std::optional<Tsetlini::ClassifierClassic> maybe_estimator;

    auto const rv = Tsetlini::make_classifier_classic_from_json()
        .rightFlatMap(
        [&](auto && est)
        {
            maybe_estimator = std::move(est);

            return Tsetlini::Either<Tsetlini::status_message_t, int>::rightOf(0);
        });

    expect(that % true == maybe_estimator.has_value());
};


"ClassifierClassic cannot be created from empty JSON"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json("");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from invalid JSON"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json("[]");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from empty JSON dict"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json("{}");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with unrecognized param"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"gotcha": 564})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with counting_type set to int8"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"counting_type": "int8"})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with counting_type set to int16"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"counting_type": "int16"})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with counting_type set to int32"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"counting_type": "int32"})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with counting_type set to auto"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"counting_type": "auto"})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with clause_output_tile_size set to 16"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"clause_output_tile_size": 16})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with clause_output_tile_size set to 32"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"clause_output_tile_size": 32})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with clause_output_tile_size set to 64"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"clause_output_tile_size": 64})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with clause_output_tile_size set to 128"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"clause_output_tile_size": 128})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with clause_output_tile_size set to 24"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"clause_output_tile_size": 24})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with n_jobs set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"n_jobs": 2})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with n_jobs set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"n_jobs": -1})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with n_jobs set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"n_jobs": 0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with n_jobs set to -2"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"n_jobs": -2})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with number_of_clauses_per_label set to 4"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": 4})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_clauses_per_label set to 3"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": 3})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_clauses_per_label set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": 2})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_clauses_per_label set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": 1})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_clauses_per_label set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": 0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_clauses_per_label set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": -1})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_clauses_per_label set to -4"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_clauses_per_label": -4})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with number_of_states set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_states": 1})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_states set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_states": 0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with number_of_states set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"number_of_states": -1})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with boost_true_positive_feedback set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"boost_true_positive_feedback": 1})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with boost_true_positive_feedback set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"boost_true_positive_feedback": 0})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with boost_true_positive_feedback set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"boost_true_positive_feedback": -1})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with boost_true_positive_feedback set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"boost_true_positive_feedback": 2})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with threshold set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"threshold": 1})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with threshold set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"threshold": 0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with threshold set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"threshold": -1})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with max_weight set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"max_weight": 1})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with max_weight set to 10"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"max_weight": 10})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with max_weight set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"max_weight": 0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with max_weight set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"max_weight": -1})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with verbose set to true"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"verbose": true})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with verbose set to false"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"verbose": false})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with weighted set to true"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"weighted": true})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with weighted set to false"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"weighted": false})");

    expect(that % true == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to -1.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": -1.0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to 0.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": 0.0})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to 0.999"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": 0.999})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to -inf"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": -inf})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to +inf"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": +inf})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to NaN"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": NaN})");

    expect(that % false == !!clf);
};


"ClassifierClassic cannot be created from JSON with specificity set to nan"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": nan})");

    expect(that % false == !!clf);
};


"ClassifierClassic can be created from JSON with specificity set to 1.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": 1.0})");

    expect(that % true == !!clf);
};


"ClassifierClassic can be created from JSON with specificity set to 3.14"_test = []
{
    auto const clf = Tsetlini::make_classifier_classic_from_json(R"({"s": 3.14})");

    expect(that % true == !!clf);
};


}; // suite TestClassifierClassicJson


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
