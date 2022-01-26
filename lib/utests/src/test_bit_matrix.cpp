#include "basic_bit_matrix.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <cstdint>
#include <sstream>


using namespace boost::ut;


using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;


suite TestBitMatrix = []
{


"BitMatrix can be created"_test = []
{
    std::stringstream side_effect;

    expect(nothrow([&]
        {
            basic_bit_matrix<unsigned int> bm(/* rows= */ 5, /* cols= */ 17);

            side_effect << &bm;
        }));
};


"BitMatrix shape() returns dimensions passed to the constructor"_test = []
{
    basic_bit_matrix<unsigned int> bm(5, 17);

    auto const [nrows, ncols] = bm.shape();

    expect(that % 5u == nrows);
    expect(that % 17u == ncols);
};


"Blocks per row is zero for BitMatrix with zero columns"_test = []
{
    basic_bit_matrix<u32, 64> bm(5, 0);

    auto const row_blocks = bm.row_blocks();

    expect(that % 0u == row_blocks);
};


"Blocks per row is correct for BitMatrix with less columns than either alignment or block capacity"_test = []
{
    /*
     * 17 columns (3 bytes) is less that single 4-byte block and less than 64-byte alignment.
     *
     * Hence, row_blocks() equals number of blocks required implied by single alignment:
     * 32 bits * 16 == 8 bits * 64 > 17 bits
     */
    basic_bit_matrix<u32, 64> bm(5, 17);

    auto const row_blocks = bm.row_blocks();

    expect(that % 16u == row_blocks);
};


"Blocks per row is correct for BitMatrix with more columns than single block capacity, but less than alignment"_test = []
{
    /*
     * 65 columns (9 bytes) is more that single 4-byte block and less than 64-byte alignment.
     *
     * Hence, row_blocks() equals number of blocks required for single alignment:
     * 32 bits * 16 == 8 bits * 64 > 65 bits
     */
    basic_bit_matrix<u32, 64> bm(5, 65);

    auto const row_blocks = bm.row_blocks();

    expect(that % 16u == row_blocks);
};


"Blocks per row is correct for BitMatrix with more columns than single block capacity, that also equals alignment"_test = []
{
    /*
     * 65 columns (9 bytes) is more that single 4-byte block and 4-byte alignment.
     *
     * Hence, row_blocks() equals number of blocks required for all columns:
     * 32 bits * 3 > 65 bits
     */
    basic_bit_matrix<u32, 4> bm(5, 65);

    auto const row_blocks = bm.row_blocks();

    expect(that % 3u == row_blocks);
};


"Blocks per row is correct for BitMatrix with more columns than single block capacity, that also exceeds alignment"_test = []
{
    /*
     * 65 columns (9 bytes) is more that single 4-byte block.
     * Bit matrix is constructed with a 2-byte alignment, but this is less than
     * alignment of 32-bit word, hence, 4-byte alignment is assumed.
     *
     * Hence, row_blocks() equals number of blocks required for all columns:
     * 32 bits * 3 > 65 bits
     */
    basic_bit_matrix<u32, 2> bm(5, 65);

    auto const row_blocks = bm.row_blocks();

    expect(that % 3u == row_blocks);
};


"Blocks per row is correct for BitMatrix with more columns than either single block capacity and alignment"_test = []
{
    /*
     * 257 columns (33 bytes) is more that single 4-byte block and 32-byte alignment.
     *
     * Hence, row_blocks() equals number of blocks required for all columns:
     * 32 bits * 16 > 257 bits
     */
    basic_bit_matrix<u32, 32> bm(5, 257);

    auto const row_blocks = bm.row_blocks();

    expect(that % 16u == row_blocks);
};


"Created BitMatrix is initialized with zeros"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    unsigned int sum = 0;

    for (auto rit = 0u; rit < NR; ++rit)
    {
        for (auto cit = 0u; cit < NC; ++cit)
        {
            sum |= bm[{rit, cit}];
        }
    }

    expect(that % 0u == sum);
};


"First bit 0,0 can be set"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 0);

    expect(that % 1u == bm.test(0, 0));
};


"First bit 0,0 can be cleared"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 0);

    bm.clear(0, 0);

    expect(that % 0u == bm.test(0, 0));
};


"First bit 0,0 can be flipped from 1 to 0"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 0);

    bm.flip(0, 0);

    expect(that % 0u == bm.test(0, 0));
};


"First bit 0,0 can be flipped from 0 to 1"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.flip(0, 0);

    expect(that % 1u == bm.test(0, 0));
};


"Last bit i,j can be set"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(NR - 1, NC - 1);

    expect(that % 1u == bm.test(NR - 1, NC - 1));
};


"Last bit i,j can be cleared"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(NR - 1, NC - 1);

    bm.clear(NR - 1, NC - 1);

    expect(that % 0u == bm.test(NR - 1, NC - 1));
};


"Last bit i,j can be flipped from 1 to 0"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(NR - 1, NC - 1);

    bm.flip(NR - 1, NC - 1);

    expect(that % 0u == bm.test(NR - 1, NC - 1));
};


"Last bit i,j can be flipped from 0 to 1"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.flip(NR - 1, NC - 1);

    expect(that % 1u == bm.test(NR - 1, NC - 1));
};


"Last bit i,j can be flipped from 0 to 1"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.flip(NR - 1, NC - 1);

    expect(that % 1u == bm.test(NR - 1, NC - 1));
};


"Two bits in a single block can be set"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 1);
    bm.set(0, 4);

    expect(that % 1u == bm.test(0, 4));
    expect(that % 1u == bm.test(0, 1));
};


"Single bits in a single block with two bits set can be cleared"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 1);
    bm.set(0, 4);

    bm.clear(0, 1);

    expect(that % 1u == bm.test(0, 4));
    expect(that % 0u == bm.test(0, 1));
};


"Set bit in first block of the next row can be flipped"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, NC - 1);
    bm.set(NR - 1, 0);

    bm.flip(NR - 1, 0);

    expect(that % 1u == bm.test(0, NC - 1));
    expect(that % 0u == bm.test(NR - 1, 0));
};


"Clear bit in first block of the next row can be flipped"_test = []
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, NC - 1);

    bm.flip(NR - 1, 0);

    expect(that % 1u == bm.test(0, NC - 1));
    expect(that % 1u == bm.test(NR - 1, 0));
};


"Correct block is modified when single bit is set"_test = []
{
    {
        auto constexpr NR = 2u;
        auto constexpr NC = 257u;

        basic_bit_matrix<u32, 32> bm(NR, NC);

        bm.set(NR - 1, 0);

        auto data = bm.data();
        auto block = data[bm.row_offset(NR - 1)];

        expect(that % 0u != block);
    }
    {
        auto constexpr NR = 257u;
        auto constexpr NC = 2u;

        basic_bit_matrix<u32, 32> bm(NR, NC);

        bm.set(NR - 1, 0);

        auto data = bm.data();
        auto block = data[bm.row_offset(NR - 1)];

        expect(that % 0u != block);
    }
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
