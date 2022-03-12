#include "tsetlini_algo_classic.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_types.hpp"

#include "strong_type/strong_type.hpp"
#include "rapidcheck.h"
#include "boost/ut.hpp"

#include <cstdlib>
#include <algorithm>


using namespace boost::ut;


long long constexpr MAX_THRESHOLD = std::numeric_limits<strong::underlying_type_t<Tsetlini::threshold_t>>::max();
auto constexpr MAX_NUM_OF_CLAUSE_OUTPUTS = 0x1000; // arbitrary

/*
 * `inRange` is exclusive on upper bound, so this is OK as max value for filling
 * weight test vectors.
 *
 *      weight = [0, MAX_WEIGHT)
 *
 * In real life scenario weight will never MAX_WEIGHT, because for
 * incrementation it is compared against `max_weight` after adding +1 to it.
 */
std::uint32_t constexpr MAX_WEIGHT = value_of(Tsetlini::MAX_WEIGHT_DEFAULT);

////////////////////////////////////////////////////////////////////////////////

/*
 * Reference:
 * "A Regression Tsetlin Machine with Integer Weighted Clauses for Compact Pattern Representation"
 * https://arxiv.org/abs/2002.01245
 * Eq 8.
 */
////////////////////////////////////////////////////////////////////////////////
///
///     Reference algorithm for weighted summation
///
////////////////////////////////////////////////////////////////////////////////
Tsetlini::response_type reference(
    Tsetlini::aligned_vector_char const & clause_outputs,
    Tsetlini::w_vector_type const & weights,
    Tsetlini::threshold_t const threshold
)
{
    auto sum = [](long long a, long long b){ return a + b; };
    auto product = [](long long c, long long w){ return c * (w + 1); };

    long long rv = std::inner_product(
        clause_outputs.cbegin(),
        clause_outputs.cend(),
        weights.cbegin(),
        0LL,
        sum, product
    );

    return std::clamp<long long>(rv, 0, value_of(threshold));
}


////////////////////////////////////////////////////////////////////////////////

suite SumUpRegressorVotes = []
{


"Weighted sum_up_regressor_votes replicates paper formula"_test = []
{
    auto ok = rc::check(
        []()
        {
            auto const threshold = Tsetlini::threshold_t{*rc::gen::inRange<long long>(1, MAX_THRESHOLD + 1)};
            auto const number_of_clause_outputs = *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS + 1);

            auto const clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(number_of_clause_outputs, rc::gen::arbitrary<bool>());
            auto const weights = *rc::gen::container<Tsetlini::w_vector_type>(number_of_clause_outputs, rc::gen::inRange(0u, MAX_WEIGHT));

            auto sum = Tsetlini::sum_up_regressor_votes(
                clause_output,
                threshold,
                weights);

            auto ground_truth = reference(clause_output, weights, threshold);

            RC_ASSERT(sum == ground_truth);
        }
    );

    expect(that % true == ok);
};


"Non-weighted sum_up_regressor_votes replicates paper formula"_test = []
{
    auto ok = rc::check(
        []()
        {
            auto const threshold = Tsetlini::threshold_t{*rc::gen::inRange<long long>(1, MAX_THRESHOLD + 1)};
            auto const number_of_clause_outputs = *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS + 1);

            auto const clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(number_of_clause_outputs, rc::gen::arbitrary<bool>());
            Tsetlini::w_vector_type const weights(number_of_clause_outputs, 0);
            Tsetlini::w_vector_type const empty_weights;

            auto sum = Tsetlini::sum_up_regressor_votes(
                clause_output,
                threshold,
                empty_weights);

            auto ground_truth = reference(clause_output, weights, threshold);

            RC_ASSERT(sum == ground_truth);
        }
    );

    expect(that % true == ok);
};


};


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
