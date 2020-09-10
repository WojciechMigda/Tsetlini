#pragma once

#ifndef LIB_SRC_LOSS_FN_HPP_
#define LIB_SRC_LOSS_FN_HPP_


#include <string>
#include <functional>


namespace Tsetlini
{

using loss_fn_type = std::function<float(float)>;

loss_fn_type make_loss_fn(std::string const & name, float const C1);


}  // namespace Tsetlini


#endif /* LIB_SRC_LOSS_FN_HPP_ */
