#pragma once

#ifndef LIB_INCLUDE_ESTIMATOR_STATE_CACHE_HPP_
#define LIB_INCLUDE_ESTIMATOR_STATE_CACHE_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_params.hpp"
#include "frand_cache.hpp"
#include "mt.hpp"

#include <type_traits>
#include <utility>


namespace Tsetlini
{


template<typename, typename = std::void_t<>>
struct is_estimator_state_cache
    : std::false_type
{
};


template<typename T>
struct is_estimator_state_cache<T, std::void_t<
        typename T::value_type,
        decltype(T::are_equal(std::declval<typename T::value_type const &>(), std::declval<typename T::value_type const &>())),
        decltype(T::reset(
            std::declval<typename T::value_type &>(),
            std::declval<params_t const &>()))
    >>
    : std::true_type
{
};


struct EstimatorStateCacheBase
{
    using frand_cache_type = frand_cache<alignment, FRNG::result_type>;
};


struct ClassifierStateCache : public EstimatorStateCacheBase
{
    struct value_type
    {
        feedback_vector_type feedback_to_clauses; // shape=(number of clauses)
        aligned_vector_char clause_output; // shape=(number of clauses)
        aligned_vector_int label_sum; // shape=(number of labels)

        frand_cache_type fcache;
    };

    static bool are_equal(value_type const & lhs, value_type const & rhs);
    static void reset(value_type & cache, params_t const & params);
};


struct RegressorStateCache : public EstimatorStateCacheBase
{
    struct value_type
    {
        feedback_vector_type feedback_to_clauses; // shape=(number of clauses)
        aligned_vector_char clause_output; // shape=(number of clauses)

        frand_cache_type fcache;
    };

    static bool are_equal(value_type const & lhs, value_type const & rhs);
    static void reset(value_type & cache, params_t const & params);
};


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_ESTIMATOR_STATE_CACHE_HPP_ */
