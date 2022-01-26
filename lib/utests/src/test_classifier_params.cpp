#include "tsetlini_params.hpp"
#include "tsetlini_types.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <variant>
#include <string>
#include <thread>
#include <algorithm>


using namespace boost::ut;
using namespace std::string_literals;


suite TestClassifierParams = []
{


"Can be created"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json();

    expect(that % true == either);
};


"Cannot be created from empty json"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json("");

    expect(that % false == either);
};


"Can be created from empty dict json"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json("{}");

    expect(that % true == either);
};


"Cannot be created from empty array json"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json("[]");

    expect(that % false == either);
};


"Cannot be created from malformed json"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json("5\"}");

    expect(that % false == either);
};


"Can be created from valid json with one integer param"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"number_of_states": 200})");

    expect(that % true == either);

    auto params = either.right().value;

    expect(that % 200 == std::get<int>(params.at("number_of_states")));
};


"Can be created from valid json with one floating point param"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"s": 3.9})");

    expect(that % true == either);

    auto params = either.right().value;

    expect(that % 3.9f == std::get<Tsetlini::real_type>(params.at("s")));
};


"Can be created from valid json with one boolean param"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"verbose": true})");

    expect(that % true == either);

    auto params = either.right().value;

    expect(that % true == std::get<bool>(params.at("verbose")));
};


"Can be created from valid json with one string param"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"counting_type": "int16"})");

    expect(that % true == either);

    auto params = either.right().value;

    expect(that % "int16"s == std::get<std::string>(params.at("counting_type")));
};


"Can be created from valid json with null random state param"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"random_state": null})");

    expect(that % true == either);

    auto params = either.right().value;

    expect(that % true == std::holds_alternative<Tsetlini::seed_type>(params.at("random_state")));
};


"Cannot be created from json with unrecognized param"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"zigzag": true})");

    expect(that % false == either);
};


"Can be created from valid json with full set of params"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"(
{
"verbose": true,
"number_of_pos_neg_clauses_per_label": 17,
"number_of_states": 125,
"s": 6.3 ,
"threshold": 8,
"weighted": true,
"max_weight": 7,
"boost_true_positive_feedback": 1,
"counting_type": "int32",
"clause_output_tile_size": 32,
"n_jobs": 3,
"random_state": 123
}
)");

    expect(that % true == either);

    auto params = either.right().value;

    expect(that % true == std::get<bool>(params.at("verbose")));
    expect(that % 17 == std::get<int>(params.at("number_of_pos_neg_clauses_per_label")));
    expect(that % 125 == std::get<int>(params.at("number_of_states")));
    expect(that % 8 == std::get<int>(params.at("threshold")));
    expect(that % true == std::get<bool>(params.at("weighted")));
    expect(that % 7 == std::get<int>(params.at("max_weight")));
    expect(that % 3 == std::get<int>(params.at("n_jobs")));
    expect(that % 1 == std::get<int>(params.at("boost_true_positive_feedback")));
    expect(that % "int32"s == std::get<std::string>(params.at("counting_type")));
    expect(that % 32 == std::get<int>(params.at("clause_output_tile_size")));
    expect(that % 6.3f == std::get<Tsetlini::real_type>(params.at("s")));
    expect(that % 123u == std::get<Tsetlini::seed_type>(params.at("random_state")));
};


"n_jobs param equal -1 is normalized with hardware concurrency"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json(R"({"n_jobs": -1})");

    expect(that % true == either);

    auto params = either.right().value;

    auto const hw_concurrency = std::max<int>(1, std::thread::hardware_concurrency());

    expect(that % hw_concurrency == std::get<int>(params.at("n_jobs")));

};


"Unspecified random_state param is initialized"_test = []
{
    auto const either = Tsetlini::make_classifier_params_from_json("{}");

    auto params = either.right().value;

    expect(that % true == std::holds_alternative<Tsetlini::seed_type>(params.at("random_state")));
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
