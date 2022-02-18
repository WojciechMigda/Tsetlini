#include "tsetlini_algo_classic.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_types.hpp"

#include "strong_type/strong_type.hpp"
#include "rapidcheck.h"
#include "boost/ut.hpp"

#include <cstdlib>
#include <algorithm>


using namespace boost::ut;


long long constexpr MAX_THRESHOLD = std::numeric_limits<strong::underlying_type<Tsetlini::threshold_t>::type>::max();
auto constexpr MAX_NUM_OF_LABELS = 10; // let's be reasonable
auto constexpr MAX_NUM_OF_POLARIZED_CLAUSE_OUTPUTS_PER_LABEL = 0x100; // arbitrary

/*
 * `inRange` is exclusive on upper bound, so this is OK as max value for filling
 * weight test vectors.
 *
 *      weight = [0, MAX_WEIGHT)
 *
 * In real life scenario weight will never MAX_WEIGHT, because for
 * incrementation it is compared against `max_weight` after adding +1 to it.
 */
std::uint64_t constexpr MAX_WEIGHT = std::numeric_limits<Tsetlini::w_vector_type::value_type>::max();

////////////////////////////////////////////////////////////////////////////////

template<typename Seq>
auto flat_zip(Seq const & a, Seq const & b) -> Seq
{
    auto const N = std::min(a.size(), b.size());
    Seq rv(2 * N);

    for (auto ix = 0u; ix < N; ++ix)
    {
        rv[2 * ix + 0] = a[ix];
        rv[2 * ix + 1] = b[ix];
    }

    return rv;
}

/*
 * Reference:
 * "Extending the Tsetlin Machine With Integer-Weighted Clauses for Increased Interpretability"
 * https://arxiv.org/abs/2005.05131
 * Eq 9.
 */
////////////////////////////////////////////////////////////////////////////////
///
///     Reference algorithm for weighted summation
///
////////////////////////////////////////////////////////////////////////////////
void reference(
    Tsetlini::aligned_vector_char const & positive_clause_outputs,
    Tsetlini::aligned_vector_char const & negative_clause_outputs,
    Tsetlini::w_vector_type const & positive_weights,
    Tsetlini::w_vector_type const & negative_weights,
    Tsetlini::number_of_labels_t const number_of_labels,
    int const number_of_polarized_clause_outputs_per_label,
    Tsetlini::threshold_t const threshold,
    Tsetlini::aligned_vector_int & sums
)
{
    auto sum = [](long long a, long long b){ return a + b; };
    auto product = [](long long c, long long w){ return c * (w + 1); };

    for (auto label = 0; label < number_of_labels; ++label)
    {
        auto const start = (label + 0) * number_of_polarized_clause_outputs_per_label;
        auto const end = (label + 1) * number_of_polarized_clause_outputs_per_label;

        long long const positive_sum = std::inner_product(
            positive_clause_outputs.cbegin() + start,
            positive_clause_outputs.cbegin() + end,
            positive_weights.cbegin() + start,
            0LL,
            sum, product
        );

        long long const negative_sum = std::inner_product(
            negative_clause_outputs.cbegin() + start,
            negative_clause_outputs.cbegin() + end,
            negative_weights.cbegin() + start,
            0LL,
            sum, product
        );

        sums[label] = std::clamp<long long>(positive_sum - negative_sum, -value_of(threshold), +value_of(threshold));
    }
}


////////////////////////////////////////////////////////////////////////////////

suite SumUpAllLabelVotes = []
{


"Weighted sum_up_all_label_votes replicates paper formula"_test = []
{
    auto ok = rc::check(
        []()
        {
            auto const threshold = Tsetlini::threshold_t{*rc::gen::inRange<long long>(1, MAX_THRESHOLD + 1)};
            auto const number_of_labels = Tsetlini::number_of_labels_t{*rc::gen::inRange(2, MAX_NUM_OF_LABELS + 1)};
            auto const number_of_polarized_clause_outputs_per_label = *rc::gen::inRange(1, MAX_NUM_OF_POLARIZED_CLAUSE_OUTPUTS_PER_LABEL + 1);
            auto const number_of_polarized_clause_outputs = value_of(number_of_labels) * number_of_polarized_clause_outputs_per_label;
            auto const number_of_clause_outputs_per_label = Tsetlini::number_of_classifier_clause_outputs_per_label_t{2 * number_of_polarized_clause_outputs_per_label};

            auto const positive_clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(number_of_polarized_clause_outputs, rc::gen::arbitrary<bool>());
            auto const negative_clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(number_of_polarized_clause_outputs, rc::gen::arbitrary<bool>());
            auto const positive_weights = *rc::gen::container<Tsetlini::w_vector_type>(number_of_polarized_clause_outputs, rc::gen::inRange<std::uint64_t>(0, MAX_WEIGHT));
            auto const negative_weights = *rc::gen::container<Tsetlini::w_vector_type>(number_of_polarized_clause_outputs, rc::gen::inRange<std::uint64_t>(0, MAX_WEIGHT));

            auto const clause_output = flat_zip(positive_clause_output, negative_clause_output);
            auto const weights = flat_zip(positive_weights, negative_weights);

            Tsetlini::aligned_vector_int label_sums(value_of(number_of_labels));

            Tsetlini::sum_up_all_label_votes(
                clause_output,
                weights,
                label_sums,
                number_of_labels,
                number_of_clause_outputs_per_label,
                threshold);

            Tsetlini::aligned_vector_int ground_truth(value_of(number_of_labels));

            reference(positive_clause_output, negative_clause_output, positive_weights, negative_weights,
                number_of_labels, number_of_polarized_clause_outputs_per_label, threshold, ground_truth);

            RC_ASSERT(label_sums == ground_truth);
        }
    );

    expect(that % true == ok);
};


"Non-weighted sum_up_all_label_votes replicates paper formula"_test = []
{
    auto ok = rc::check(
        []()
        {
            auto const threshold = Tsetlini::threshold_t{*rc::gen::inRange<long long>(1, MAX_THRESHOLD + 1)};
            auto const number_of_labels = Tsetlini::number_of_labels_t{*rc::gen::inRange(2, MAX_NUM_OF_LABELS + 1)};
            auto const number_of_polarized_clause_outputs_per_label = *rc::gen::inRange(1, MAX_NUM_OF_POLARIZED_CLAUSE_OUTPUTS_PER_LABEL + 1);
            auto const number_of_polarized_clause_outputs = value_of(number_of_labels) * number_of_polarized_clause_outputs_per_label;
            auto const number_of_clause_outputs_per_label = Tsetlini::number_of_classifier_clause_outputs_per_label_t{2 * number_of_polarized_clause_outputs_per_label};

            auto const positive_clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(number_of_polarized_clause_outputs, rc::gen::arbitrary<bool>());
            auto const negative_clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(number_of_polarized_clause_outputs, rc::gen::arbitrary<bool>());
            Tsetlini::w_vector_type const positive_weights(number_of_polarized_clause_outputs, 0);
            Tsetlini::w_vector_type const negative_weights(number_of_polarized_clause_outputs, 0);

            auto const clause_output = flat_zip(positive_clause_output, negative_clause_output);
            Tsetlini::w_vector_type const empty_weights;

            Tsetlini::aligned_vector_int label_sums(value_of(number_of_labels));

            Tsetlini::sum_up_all_label_votes(
                clause_output,
                empty_weights,
                label_sums,
                number_of_labels,
                number_of_clause_outputs_per_label,
                threshold);

            Tsetlini::aligned_vector_int ground_truth(value_of(number_of_labels));

            reference(positive_clause_output, negative_clause_output, positive_weights, negative_weights,
                number_of_labels, number_of_polarized_clause_outputs_per_label, threshold, ground_truth);

            RC_ASSERT(label_sums == ground_truth);
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
