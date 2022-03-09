#pragma once

#ifndef LIB_INCLUDE_EITHER_HPP_
#define LIB_INCLUDE_EITHER_HPP_

#include "neither/either.hpp"


namespace Tsetlini
{


template<class L, class R>
using Either = neither::Either<L, R>;


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_EITHER_HPP_ */
