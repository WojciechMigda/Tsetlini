#include "loss_fn.hpp"


#include <functional>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>


namespace Tsetlini
{


std::function<float(float)> make_loss_fn(std::string const & name)
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
    else
    {
        printf("loss_fn.cpp - unreachable code condition. Aborting\n");
        std::exit(1);
        return [](float x){ return 0; };
    }
}


}  // namespace Tsetlini
