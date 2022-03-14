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


suite TestClassifierBitwiseArgs = []
{


"ClassifierBitwise can be created from default arguments"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise();

    expect(that % true == !!clf);
};


"ClassifierBitwise created from arguments can be move-assigned"_test = []
{
    std::optional<Tsetlini::ClassifierBitwise> maybe_estimator;

    auto const rv = Tsetlini::make_classifier_bitwise()
        .rightFlatMap(
        [&](auto && est)
        {
            maybe_estimator = std::move(est);

            return Tsetlini::Either<Tsetlini::status_message_t, int>::rightOf(0);
        });

    expect(that % true == maybe_estimator.has_value());
};


"ClassifierBitwise can be created with counting_type argument set to {int8, int16, int32, auto}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::counting_type_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::counting_type(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{"int8"s, "int16"s, "int32"s, "auto"s};


"ClassifierBitwise can be created with clause_output_tile_size argument set to {16, 32, 64, 128}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::clause_output_tile_size_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::clause_output_tile_size(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{16, 32, 64, 128};


"ClassifierBitwise cannot be created with clause_output_tile_size argument set to 24"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::clause_output_tile_size_t{24});

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created with n_jobs argument set to 2"_test = []
{
    Tsetlini::make_classifier_bitwise(Tsetlini::number_of_jobs_t{2})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % 2 == value_of(Tsetlini::Params::n_jobs(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierBitwise can be created with number_of_jobs argument set to -1"_test = []
{
    using underlying_type = strong::underlying_type_t<Tsetlini::number_of_jobs_t>;

    Tsetlini::make_classifier_bitwise(Tsetlini::number_of_jobs_t{-1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % underlying_type(std::thread::hardware_concurrency()) == value_of(Tsetlini::Params::n_jobs(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierBitwise cannot be created with number_of_jobs argument set to {0, -2}"_test = [](auto const & arg)
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::number_of_jobs_t{arg});

    expect(that % false == !!clf);
} | std::vector{0, -2};


"ClassifierBitwise can be created with number_of_physical_classifier_clauses_per_label argument set to 4"_test = []
{
    Tsetlini::make_classifier_bitwise(Tsetlini::number_of_physical_classifier_clauses_per_label_t{4})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % 4 == value_of(Tsetlini::Params::number_of_physical_classifier_clauses_per_label(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierBitwise cannot be created with number_of_physical_classifier_clauses_per_label argument set to {3, 2, 1, 0, -1, -4}"_test = [](auto const & arg)
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::number_of_physical_classifier_clauses_per_label_t{arg});

    expect(that % false == !!clf);
} | std::vector{3, 2, 1, 0, -1, -4};


"ClassifierBitwise can be created with number_of_states set argument to 1"_test = []
{
    Tsetlini::make_classifier_bitwise(Tsetlini::number_of_states_t{1})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % 1 == value_of(Tsetlini::Params::number_of_states(clf.read_params())));

                return std::move(clf);
            })
        ;
};


"ClassifierBitwise cannot be created with number_of_states argument set to {0, -1}"_test = [](auto const & arg)
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::number_of_states_t{arg});

    expect(that % false == !!clf);
} | std::vector{0, -1};


"ClassifierBitwise can be created with boost_true_positive_feedback argument set to {true, false}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::boost_tpf_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::boost_true_positive_feedback(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{true, false};


"ClassifierBitwise can be created with threshold argument set to {1, 2, 10}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::threshold_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::threshold(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{1, 2, 10};


"ClassifierBitwise cannot be created with threshold argument set to {0, -1}"_test = [](auto const & arg)
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::threshold_t{arg});

    expect(that % false == !!clf);
} | std::vector{0, -1};


"ClassifierBitwise can be created with max_weight argument set to {1, 2, 10}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::max_weight_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::max_weight(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{1, 2, 10};


"ClassifierBitwise cannot be created with max_weight argument set to {0, -1}"_test = [](auto const & arg)
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::max_weight_t{arg});

    expect(that % false == !!clf);
} | std::vector{0, -1};


"ClassifierBitwise can be created with verbosity argument set to {true, false}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::verbosity_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::verbose(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{true, false};


"ClassifierBitwise can be created with weighted argument set to {true, false}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::weighted_flag_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::weighted(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{true, false};


"ClassifierBitwise cannot be created with specificity argument set to {-1.0, 0.0, 1 - epsilon -inf, +inf, -NaN, +NaN}"_test = [](auto const & arg)
{
    auto const clf = Tsetlini::make_classifier_bitwise(Tsetlini::specificity_t{arg});

    expect(that % false == !!clf);
} | std::vector<strong::underlying_type_t<Tsetlini::specificity_t>>{
    -1.0, 0.0,
    strong::underlying_type_t<Tsetlini::specificity_t>{1} - std::numeric_limits<strong::underlying_type_t<Tsetlini::specificity_t>>::epsilon(),
    -std::numeric_limits<strong::underlying_type_t<Tsetlini::specificity_t>>::infinity(),
    std::numeric_limits<strong::underlying_type_t<Tsetlini::specificity_t>>::infinity(),
    NAN, -NAN
};


"ClassifierBitwise can be created with specificity argument set to {1.0, 3.14}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::specificity_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::s(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector<strong::underlying_type_t<Tsetlini::specificity_t>>{1.0, 3.14};


"ClassifierBitwise can be created with random_seed argument set to {1, 1234}"_test = [](auto const & arg)
{
    Tsetlini::make_classifier_bitwise(Tsetlini::random_seed_t{arg})
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([&arg](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % arg == value_of(Tsetlini::Params::random_state(clf.read_params())));

                return std::move(clf);
            })
        ;
} | std::vector{1u, 1234u};


"ClassifierBitwise can be created without random_seed argument"_test = []
{
    Tsetlini::make_classifier_bitwise()
        .leftMap([](Tsetlini::status_message_t && msg)
            {
                expect(false);

                return std::move(msg);
            })
        .rightMap([](Tsetlini::ClassifierBitwise && clf)
            {
                expect(that % true == (clf.read_params().find("random_state") != clf.read_params().cend()));

                return std::move(clf);
            })
        ;
};


"ClassifierBitwise can be created with arbitrarily ordered arguments"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise(
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


}; // suite TestClassifierBitwiseArgs


////////////////////////////////////////////////////////////////////////////////


suite TestClassifierBitwiseJson = []
{


"ClassifierBitwise can be created"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json();

    expect(that % true == !!clf);
};


"ClassifierBitwise can be move-assigned"_test = []
{
    std::optional<Tsetlini::ClassifierBitwise> maybe_estimator;

    auto const rv = Tsetlini::make_classifier_bitwise_from_json()
        .rightFlatMap(
        [&](auto && est)
        {
            maybe_estimator = std::move(est);

            return Tsetlini::Either<Tsetlini::status_message_t, int>::rightOf(0);
        });

    expect(that % true == maybe_estimator.has_value());
};


"ClassifierBitwise cannot be created from empty JSON"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json("");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from invalid JSON"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json("[]");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from empty JSON dict"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json("{}");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with unrecognized param"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"gotcha": 564})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with counting_type set to int8"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"counting_type": "int8"})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with counting_type set to int16"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"counting_type": "int16"})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with counting_type set to int32"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"counting_type": "int32"})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with counting_type set to auto"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"counting_type": "auto"})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with clause_output_tile_size set to 16"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"clause_output_tile_size": 16})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with clause_output_tile_size set to 32"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"clause_output_tile_size": 32})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with clause_output_tile_size set to 64"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"clause_output_tile_size": 64})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with clause_output_tile_size set to 128"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"clause_output_tile_size": 128})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with clause_output_tile_size set to 24"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"clause_output_tile_size": 24})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with n_jobs set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"n_jobs": 2})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with n_jobs set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"n_jobs": -1})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with n_jobs set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"n_jobs": 0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with n_jobs set to -2"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"n_jobs": -2})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with number_of_clauses_per_label set to 4"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": 4})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_clauses_per_label set to 3"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": 3})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_clauses_per_label set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": 2})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_clauses_per_label set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": 1})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_clauses_per_label set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": 0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_clauses_per_label set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": -1})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_clauses_per_label set to -4"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_clauses_per_label": -4})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with number_of_states set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_states": 1})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_states set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_states": 0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with number_of_states set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"number_of_states": -1})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with boost_true_positive_feedback set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"boost_true_positive_feedback": 1})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with boost_true_positive_feedback set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"boost_true_positive_feedback": 0})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with boost_true_positive_feedback set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"boost_true_positive_feedback": -1})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with boost_true_positive_feedback set to 2"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"boost_true_positive_feedback": 2})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with threshold set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"threshold": 1})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with threshold set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"threshold": 0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with threshold set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"threshold": -1})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with max_weight set to 1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"max_weight": 1})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with max_weight set to 10"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"max_weight": 10})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with max_weight set to 0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"max_weight": 0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with max_weight set to -1"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"max_weight": -1})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with verbose set to true"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"verbose": true})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with verbose set to false"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"verbose": false})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with weighted set to true"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"weighted": true})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with weighted set to false"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"weighted": false})");

    expect(that % true == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to -1.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": -1.0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to 0.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": 0.0})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to 0.999"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": 0.999})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to -inf"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": -inf})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to +inf"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": +inf})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to NaN"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": NaN})");

    expect(that % false == !!clf);
};


"ClassifierBitwise cannot be created from JSON with specificity set to nan"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": nan})");

    expect(that % false == !!clf);
};


"ClassifierBitwise can be created from JSON with specificity set to 1.0"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": 1.0})");

    expect(that % true == !!clf);
};


"ClassifierBitwise can be created from JSON with specificity set to 3.14"_test = []
{
    auto const clf = Tsetlini::make_classifier_bitwise_from_json(R"({"s": 3.14})");

    expect(that % true == !!clf);
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
