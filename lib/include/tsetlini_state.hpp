#pragma once

#include "frand_cache.hpp"
#include "mt.hpp"

#include "tsetlini_params.hpp"
#include "tsetlini_types.hpp"

#include <vector>

namespace Tsetlini
{


struct ClassifierState
{
    using frand_cache_type = frand_cache<alignment>;
    using ta_state_v_type = std::variant<
        numeric_matrix_int32
        , numeric_matrix_int16
        , numeric_matrix_int8
    >;

    struct Cache
    {
        feedback_vector_type feedback_to_clauses; // shape=(number of clauses)
        aligned_vector_char clause_output; // shape=(number of clauses)
        aligned_vector_int label_sum; // shape=(number of labels)

        frand_cache_type fcache;
    };

    params_t m_params;

    ta_state_v_type ta_state;

    mutable Cache cache;

    IRNG igen;
    FRNG fgen;

    explicit ClassifierState(params_t const & params);

    bool operator==(ClassifierState const & other) const;
};


struct RegressorState
{
    using frand_cache_type = frand_cache<alignment>;
    using ta_state_v_type = std::variant<
        numeric_matrix_int32
        , numeric_matrix_int16
        , numeric_matrix_int8
    >;

    struct Cache
    {
        feedback_vector_type feedback_to_clauses; // shape=(number of clauses)
        aligned_vector_char clause_output; // shape=(number of clauses)

        frand_cache_type fcache;
    };

    params_t m_params;

    ta_state_v_type ta_state;

    mutable Cache cache;

    IRNG igen;
    FRNG fgen;

    explicit RegressorState(params_t const & params);

    bool operator==(RegressorState const & other) const;
};


} // namespace Tsetlini
