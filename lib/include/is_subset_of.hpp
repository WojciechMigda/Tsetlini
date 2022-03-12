#pragma once

#ifndef LIB_INCLUDE_IS_SUBSET_OF_HPP_
#define LIB_INCLUDE_IS_SUBSET_OF_HPP_


#include <tuple>
#include <type_traits>


namespace Tsetlini::meta
{


template <typename T, typename... Ts>
constexpr bool contains = (std::is_same<T, Ts>{} || ...);

template <typename Subset, typename... Set>
constexpr bool is_subset_of = false;

/*
 * Check if argument pack Us is a subset of Ts, passed using a tuple
 *
 * Inspired by https://stackoverflow.com/questions/42580997/check-if-one-set-of-types-is-a-subset-of-the-other/42581655#42581655
 */
template <typename... Ts, typename... Us>
constexpr bool is_subset_of<std::tuple<Ts...>, Us...> = (contains<Us, Ts...> && ...);


}  // namespace Tsetlini::meta


#endif /* LIB_INCLUDE_IS_SUBSET_OF_HPP_ */
