#include "arg_extract.hpp"
#include "boost/ut.hpp"

#include <cstdlib>


using namespace boost::ut;


static constexpr int MEANING_OF_LIFE = 42;
static constexpr int NOT_FOUND = 13;


struct Noncopyable
{
    const int v;

    Noncopyable(int num): v(num) {}
    Noncopyable(Noncopyable const &) = delete;

private:
    Noncopyable & operator=(Noncopyable const &);
};


template<typename ...Args>
int wrapped_extract(long, void *, Args && ...args)
{
    return Tsetlini::arg::extract<Noncopyable>(args...).v;
}


template<typename ...Args>
int wrapped_extract_or_lv(int num, void *, Args && ...args)
{
    Noncopyable dval(num);

    return Tsetlini::arg::extract_or<Noncopyable>(dval, args...).v;
}


template<typename ...Args>
int wrapped_extract_or_clv(int num, void *, Args && ...args)
{
    Noncopyable const dval(num);

    return Tsetlini::arg::extract_or<Noncopyable>(dval, args...).v;
}


template<typename ...Args>
int wrapped_extract_or_rv(int num, void *, Args && ...args)
{
    return Tsetlini::arg::extract_or<Noncopyable>(Noncopyable(num), args...).v;
}


suite TestParamExtract = []
{


"extract works with an r-value"_test = []
{
    int v = wrapped_extract(0L, nullptr, Noncopyable(MEANING_OF_LIFE), 1);

    expect(that % v == MEANING_OF_LIFE);
};


"extract works with an l-value"_test = []
{
    Noncopyable lval(MEANING_OF_LIFE);

    int v = wrapped_extract(0L, nullptr, lval, 1);

    expect(that % v == MEANING_OF_LIFE);
};


"extract works with a const l-value"_test = []
{
    Noncopyable const clval(MEANING_OF_LIFE);

    int v = wrapped_extract(0L, nullptr, clval, 1);

    expect(that % v == MEANING_OF_LIFE);
};


};


suite TestParamExtractOr = []
{


"extract_or works with r-value default and missing parameter"_test = []
{
    int v = wrapped_extract_or_rv(NOT_FOUND, nullptr, nullptr);

    expect(that % v == NOT_FOUND);
};


"extract_or works with r-value default and r-value parameter"_test = []
{
    int v = wrapped_extract_or_rv(NOT_FOUND, nullptr, Noncopyable(MEANING_OF_LIFE));

    expect(that % v == MEANING_OF_LIFE);
};


"extract_or works with r-value default and l-value parameter"_test = []
{
    Noncopyable lval(MEANING_OF_LIFE);

    int v = wrapped_extract_or_rv(NOT_FOUND, nullptr, lval);

    expect(that % v == MEANING_OF_LIFE);
};


"extract_or works with r-value default and const l-value parameter"_test = []
{
    Noncopyable const lval(MEANING_OF_LIFE);

    int v = wrapped_extract_or_rv(NOT_FOUND, nullptr, lval);

    expect(that % v == MEANING_OF_LIFE);
};


////////////////////////////////////////////////////////////////////////////////


"extract_or works with l-value default and missing parameter"_test = []
{
    int v = wrapped_extract_or_lv(NOT_FOUND, nullptr, nullptr);

    expect(that % v == NOT_FOUND);
};


"extract_or works with l-value default and r-value parameter"_test = []
{
    int v = wrapped_extract_or_lv(NOT_FOUND, nullptr, Noncopyable(MEANING_OF_LIFE));

    expect(that % v == MEANING_OF_LIFE);
};


"extract_or works with l-value default and l-value parameter"_test = []
{
    Noncopyable lval(MEANING_OF_LIFE);

    int v = wrapped_extract_or_lv(NOT_FOUND, nullptr, lval);

    expect(that % v == MEANING_OF_LIFE);
};


"extract_or works with l-value default and const l-value parameter"_test = []
{
    Noncopyable const clval(MEANING_OF_LIFE);

    int v = wrapped_extract_or_lv(NOT_FOUND, nullptr, clval);

    expect(that % v == MEANING_OF_LIFE);
};


////////////////////////////////////////////////////////////////////////////////


"extract_or works with const l-value default and missing parameter"_test = []
{
    int v = wrapped_extract_or_clv(NOT_FOUND, nullptr, nullptr);

    expect(that % v == NOT_FOUND);
};


"extract_or works with const l-value default and r-value parameter"_test = []
{
    int v = wrapped_extract_or_clv(NOT_FOUND, nullptr, Noncopyable(MEANING_OF_LIFE));

    expect(that % v == MEANING_OF_LIFE);
};


"extract_or works with const l-value default and l-value parameter"_test = []
{
    Noncopyable lval(MEANING_OF_LIFE);

    int v = wrapped_extract_or_clv(NOT_FOUND, nullptr, lval);

    expect(that % v == MEANING_OF_LIFE);
};


"extract_or works with const l-value default and const l-value parameter"_test = []
{
    Noncopyable const clval(MEANING_OF_LIFE);

    int v = wrapped_extract_or_clv(NOT_FOUND, nullptr, clval);

    expect(that % v == MEANING_OF_LIFE);
};


};


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
