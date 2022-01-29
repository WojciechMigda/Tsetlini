#include "basic_numeric_matrix.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <cstdint>


using namespace boost::ut;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;


suite TestNumericMatrix = []
{


"Can be created"_test = []
{
    basic_numeric_matrix<unsigned int> nm(5, 17);
};


"NumericMatrix shape() returns dimensions passed to the constructor"_test = []
{
    basic_numeric_matrix<unsigned int> nm(5, 17);

    auto const [nrows, ncols] = nm.shape();
    expect(that % 5u == nrows);
    expect(that % 17u == ncols);
};


"Number of elements per row is zero for NumericMatrix with zero columns"_test = []
{
    basic_numeric_matrix<u32, 64> nm(5, 0);

    auto const row_items = nm.row_items();

    expect(that % 0u == row_items);
};


"Number of elements per row is correct for NumericMatrix with less columns than single alignment"_test = []
{
    /*
     * 3 columns (12 bytes) is less than one 64-byte alignment (16 x 4-byte words).
     */
    basic_numeric_matrix<u32, 64> nm(5, 3);

    auto const row_items = nm.row_items();

    expect(that % 16u == row_items);
};


"Number of elements per row is correct for NumericMatrix with number of columns which occupies entire single alignment"_test = []
{
    /*
     * 16 columns (64 bytes) takes the same amount of memory as exactly one
     * 64-byte alignment (16 x 4-byte words).
     */
    basic_numeric_matrix<u32, 64> nm(5, 16);

    auto const row_items = nm.row_items();

    expect(that % 16u == row_items);
};


"Number of elements per row is correct for NumericMatrix with more columns than single alignment"_test = []
{
    /*
     * 9 columns (36 bytes) is more than one 32-byte alignment (8 x 4-byte words).
     */
    basic_numeric_matrix<u32, 32> nm(5, 9);

    auto const row_items = nm.row_items();

    expect(that % 16u == row_items);
};


"Created NumericMatrix is initialized with zeros"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_numeric_matrix<u32, 32> nm(NR, NC);

    u32 sum = 0;

    for (auto rit = 0u; rit < NR; ++rit)
    {
        for (auto cit = 0u; cit < NC; ++cit)
        {
            sum |= nm[{rit, cit}];
        }
    }

    expect(that % 0u == sum);
};


"First element 0,0 can be set"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_numeric_matrix<u32, 32> nm(NR, NC);

    nm.row_data(0)[0] = 1;

    expect(that % 1u == nm[{0, 0}]);
};


"Last element i,j can be set"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_numeric_matrix<u32, 32> nm(NR, NC);

    nm.row_data(NR - 1)[NC - 1] = 1;

    expect(that % 1u == nm[{NR - 1, NC - 1}]);
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
