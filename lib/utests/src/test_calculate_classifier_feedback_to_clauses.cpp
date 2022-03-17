#include "tsetlini_algo_classic.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_strong_params_private.hpp"
#include "tsetlini_types.hpp"

#include "strong_type/strong_type.hpp"
#include "boost/ut.hpp"

#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <random>
#include <cmath>


using namespace boost::ut;


auto constexpr MAX_NUM_OF_CLAUSE_OUTPUTS_PER_LABEL = 4;
auto constexpr MAX_NUM_OF_LABELS = 5;
auto constexpr MAX_THRESHOLD = 1024;


/*
 * generate random integer in closed range [lo, hi]
 */
auto random_int = [](auto & gen, auto lo, auto hi)
{
    return std::uniform_int_distribution<>(lo, hi)(gen);
};


/*
 * Extract range for a label from a sequence
 */
template<typename T>
auto range_for_label(
    std::vector<T> const & seq,
    Tsetlini::number_of_classifier_clause_outputs_per_label_t number_of_outputs_per_label,
    Tsetlini::label_type label)
{
    return std::vector<T>(seq.cbegin() + label * value_of(number_of_outputs_per_label), seq.cbegin() + (label + 1) * value_of(number_of_outputs_per_label));
};


/*
 * Extract even-indexed elements
 */
template<typename T>
auto extract_even_positions(std::vector<T> const & seq)
{
    std::vector<T> rv;

    for (auto ix = 0u; ix < seq.size(); ++ix)
    {
        if (ix % 2 == 0)
        {
            rv.push_back(seq[ix]);
        }
    }

    return rv;
}


/*
 * Extract odd-indexed elements
 */
template<typename T>
auto extract_odd_positions(std::vector<T> const & seq)
{
    std::vector<T> rv;

    for (auto ix = 0u; ix < seq.size(); ++ix)
    {
        if (ix % 2 == 1)
        {
            rv.push_back(seq[ix]);
        }
    }

    return rv;
}


/*
 * In-range functor
 */
auto in_range = [](auto target, auto margin)
{
    return [target, margin](auto x)
        {
            return
                ((target - margin) <= x) and (x <= (target + margin));
        };
};


////////////////////////////////////////////////////////////////////////////////


suite CalculateClassifierFeedbackToClauses = []
{


"calculate_classifier_feedback_to_clauses yields statistically correct results"_test = []
{
    /*
     * Begin with a PRNG section
     */
    std::random_device rd;
    std::mt19937 gen(rd());

    auto const seed = rd();
    FRNG fgen(seed);

    /*
     * Initialize few random constants for the algorithm
     */
    Tsetlini::number_of_classifier_clause_outputs_per_label_t number_of_clause_outputs_per_label{4 * random_int(gen, 1, MAX_NUM_OF_CLAUSE_OUTPUTS_PER_LABEL)};
    Tsetlini::number_of_labels_t number_of_labels{random_int(gen, 2, MAX_NUM_OF_LABELS)};
    Tsetlini::feedback_vector_type feedback_to_clauses(value_of(number_of_clause_outputs_per_label) * value_of(number_of_labels));

    Tsetlini::threshold_t threshold{random_int(gen, 1, MAX_THRESHOLD)};
    int const target_label_votes = random_int(gen, 0, value_of(threshold));
    int const opposite_label_votes = random_int(gen, 0, value_of(threshold));

    Tsetlini::label_type target_label{random_int(gen, 0, value_of(number_of_labels) - 1)};
    Tsetlini::label_type opposite_label{(target_label + 1 + random_int(gen, 0, value_of(number_of_labels) - 2)) % value_of(number_of_labels)};

    /*
     * Prepare containers for aggregate statistics
     */
    std::vector<std::size_t> type_1_counts(feedback_to_clauses.size(), 0);
    std::vector<std::size_t> type_2_counts(feedback_to_clauses.size(), 0);

    /*
     * Repeatedly call the algorithm and collect feedback counts
     */
    auto N_REPEAT = 10'000u * feedback_to_clauses.size();

    for (auto it = 0u; it < N_REPEAT; ++it)
    {
        Tsetlini::calculate_classifier_feedback_to_clauses(
            feedback_to_clauses,
            target_label,
            opposite_label,
            target_label_votes,
            opposite_label_votes,
            Tsetlini::number_of_classifier_clause_outputs_per_label_t{number_of_clause_outputs_per_label},
            threshold,
            fgen);

        for (auto ix = 0u; ix < feedback_to_clauses.size(); ++ix)
        {
            type_1_counts[ix] += feedback_to_clauses[ix] == Tsetlini::Type_I_Feedback;
            type_2_counts[ix] += feedback_to_clauses[ix] == Tsetlini::Type_II_Feedback;
        }
    }


    /*
     * Aggregates will contain non-zero numbers only for ranges which correspond
     * to labels which were subject to the algorithm, and then within these
     * ranges the numbers follow a pattern.
     *
     * First, extract the ranges:
     */
    auto const sparse_type_1_counts_for_target_label = range_for_label(type_1_counts, number_of_clause_outputs_per_label, target_label);
    auto const sparse_type_2_counts_for_target_label = range_for_label(type_2_counts, number_of_clause_outputs_per_label, target_label);
    auto const sparse_type_1_counts_for_opposite_label = range_for_label(type_1_counts, number_of_clause_outputs_per_label, opposite_label);
    auto const sparse_type_2_counts_for_opposite_label = range_for_label(type_2_counts, number_of_clause_outputs_per_label, opposite_label);

    /*
     * Now, for the target label the Type I feedback is assigned only to the 'direct
     * literal' clauses (even indices), while the Type II feedback is assigned only
     * to the 'negated literal' clauses (odd indices).
     */
    auto const type_1_counts_for_target_label = extract_even_positions(sparse_type_1_counts_for_target_label);
    auto const type_2_counts_for_target_label = extract_odd_positions(sparse_type_2_counts_for_target_label);

    /*
     * In contrast, for the opposite label the Type I feedback is assigned only to
     * the 'negated literal' clauses (odd indices), while the Type II feedback
     * is assigned only to the 'direct literal' clauses (even indices).
     */
    auto const type_1_counts_for_opposite_label = extract_odd_positions(sparse_type_1_counts_for_opposite_label);
    auto const type_2_counts_for_opposite_label = extract_even_positions(sparse_type_2_counts_for_opposite_label);

    /*
     * With that we can verify Type I and Type II aggregate counts for both labels.
     * Feedback counts, for each label respectively, should be on average the same
     * irrespective of feedback type.
     * For the target label it should be:
     *
     *      Sigma(target_label) = N_REPEAT * (T - target_label_votes) / 2T
     *
     * For the opposite label it should be:
     *
     *      Sigma(opposite_label) = N_REPEAT * (T + opposite_label_votes) / 2T
     *
     * From the above Sigma(target_label) will always fall into [0, N_REPEAT / 2] range,
     * while Sigma(opposite_label) will fall into [N_REPEAT / 2, N_REPEAT] range.
     */
    const auto THR2_inv = (1.f / (value_of(threshold) * 2));

    const auto expected_target_label_count = std::round(N_REPEAT * THR2_inv * (value_of(threshold) - target_label_votes));
    const auto expected_opposite_label_count = std::round(N_REPEAT * THR2_inv * (value_of(threshold) + opposite_label_votes));

    /*
     * Now we can verify that each count falls within a margin of expected value.
     * For the target label I will use margin of (N_REPEAT / 2) / 100,
     * for the opposite label the margin will be N_REPEAT / 100.
     */
    unsigned int const target_label_margin = std::round((N_REPEAT / 2) / 100.);
    unsigned int const opposite_label_margin = std::round(N_REPEAT / 100.);

    expect(that % true == std::all_of(type_1_counts_for_target_label.cbegin(), type_1_counts_for_target_label.cend(),
        in_range(expected_target_label_count, target_label_margin))) << "type_1_counts_for_target_label outside expected margin";
    expect(that % true == std::all_of(type_2_counts_for_target_label.cbegin(), type_2_counts_for_target_label.cend(),
        in_range(expected_target_label_count, target_label_margin))) << "type_2_counts_for_target_label outside expected margin";

    expect(that % true == std::all_of(type_1_counts_for_opposite_label.cbegin(), type_1_counts_for_opposite_label.cend(),
        in_range(expected_opposite_label_count, opposite_label_margin))) << "type_1_counts_for_opposite_label outside expected margin";
    expect(that % true == std::all_of(type_2_counts_for_opposite_label.cbegin(), type_2_counts_for_opposite_label.cend(),
        in_range(expected_opposite_label_count, opposite_label_margin))) << "type_2_counts_for_opposite_label outside expected margin";
};


}; // suite CalculateClassifierFeedbackToClauses


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
