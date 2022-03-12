#pragma once

#ifndef LIB_INCLUDE_ARG_EXTRACT_HPP_
#define LIB_INCLUDE_ARG_EXTRACT_HPP_

#include <tuple>
#include <type_traits>
#include <optional>


namespace Tsetlini::arg
{


template<typename T, typename ...Args>
auto extract(Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<T, std::remove_reference_t<Args>>...>, T &&>
{
    return std::get<T &&>(std::forward_as_tuple(std::move(args)...));
}

template<typename T, typename ...Args>
auto extract(Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<T const, std::remove_reference_t<Args>>...>, T const &>
{
    return std::get<T const &&>(std::forward_as_tuple(std::move(args)...));
}


////////////////////////////////////////////////////////////////////////////////


template<typename T, typename ...Args>
auto extract_or(T &&, Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<std::remove_reference_t<T>, std::remove_reference_t<Args>>...>, T &&>
{
    return std::get<T &&>(std::forward_as_tuple(std::move(args)...));
}


template<typename T, typename ...Args>
auto extract_or(T &&, Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<std::remove_reference_t<T> const, std::remove_reference_t<Args>>...>, T const &>
{
    return std::get<T const &&>(std::forward_as_tuple(std::move(args)...));
}


template<typename T, typename ...Args>
auto extract_or(T const &, Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<std::remove_reference_t<T>, std::remove_reference_t<Args>>...>, T &&>
{
    return std::get<T &&>(std::forward_as_tuple(std::move(args)...));
}


template<typename T, typename ...Args>
auto extract_or(T const &, Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<std::remove_reference_t<T> const, std::remove_reference_t<Args>>...>, T const &>
{
    return std::get<T const &&>(std::forward_as_tuple(std::move(args)...));
}


////////////////////////////////////////////////////////////////////////////////


template<typename T, typename ...Args>
auto extract_or(T && dval, Args && ...) -> std::enable_if_t<not std::disjunction_v<std::is_same<T, std::decay_t<Args>>...>, T &&>
{
    return std::move(dval);
}


template<typename T, typename ...Args>
auto extract_or(T const & dval, Args && ...) -> std::enable_if_t<not std::disjunction_v<std::is_same<T, std::decay_t<Args>>...>, T const &>
{
    return dval;
}


////////////////////////////////////////////////////////////////////////////////


template<typename T, typename ...Args>
auto maybe_extract(Args && ...) -> std::enable_if_t<not std::disjunction_v<std::is_same<T, std::decay_t<Args>>...>, std::optional<T>>
{
    return std::nullopt;
}


template<typename T, typename ...Args>
auto maybe_extract(Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<T, std::remove_reference_t<Args>>...>, std::optional<T>>
{
    return std::get<T &&>(std::forward_as_tuple(std::move(args)...));
}

template<typename T, typename ...Args>
auto maybe_extract(Args && ...args) -> std::enable_if_t<std::disjunction_v<std::is_same<T const, std::remove_reference_t<Args>>...>, std::optional<T>>
{
    return std::get<T const &&>(std::forward_as_tuple(std::move(args)...));
}


}  // namespace Tsetlini::arg


#endif /* LIB_INCLUDE_ARG_EXTRACT_HPP_ */
