#include "loss_fn.hpp"
#include "tsetlini_strong_params.hpp"

#include "strong_type/strong_type.hpp"

#include <functional>
#include <cstdlib>
#include <cstdio>
#include <cmath>


namespace Tsetlini
{


std::function<float(float)> make_loss_fn(loss_fn_name_t const & name, loss_fn_C1_t const C1)
{
    if ((name == "MAE") or
        (name == "L1"))
    {
        return [](float x){ return std::abs(x); };
    }
    else if ((name == "MSE") or
        (name == "L2"))
    {
        return [](float x){ return x * x; };
    }
    else if (name == "berHu")
    {
        return [C1](float const x)
            {
                if (std::abs(x) <= C1)
                {
                    return std::abs(x);
                }
                else
                {
                    return x * x - value_of(C1 * C1 + C1);
                }
            };
    }
    else if (name == "L1+2")
    {
        return [C1](float x)
            {
                return value_of(C1) * std::abs(x) + (1.f - value_of(C1)) * x * x;
            };
    }
    else
    {
        printf("loss_fn.cpp - unreachable code condition. Aborting\n");
        std::exit(1);
        return [](float x){ return 0; };
    }
}


}  // namespace Tsetlini
