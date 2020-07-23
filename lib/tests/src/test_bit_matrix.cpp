#include "basic_bit_matrix.hpp"

#include <gtest/gtest.h>
#include <cstdint>


namespace
{

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;


TEST(BitMatrix, can_be_created)
{
    basic_bit_matrix<unsigned int> bm(5, 17);
}


TEST(BitMatrix, shape_matches_ctor)
{
    basic_bit_matrix<unsigned int> bm(5, 17);

    auto const [nrows, ncols] = bm.shape();
    ASSERT_EQ(5, nrows);
    ASSERT_EQ(17, ncols);
}


TEST(BitMatrix, row_blocks_is_correct_for_zero_columns)
{
    basic_bit_matrix<u32, 64> bm(5, 0);

    auto const row_blocks = bm.row_blocks();

    EXPECT_EQ(0, row_blocks);
}


TEST(BitMatrix, row_blocks_is_correct_for_data_smaller_than_alignment_and_block)
{
    // data < block < alignment

    basic_bit_matrix<u32, 64> bm(5, 17);

    auto const row_blocks = bm.row_blocks();

    EXPECT_EQ(16, row_blocks);
}


TEST(BitMatrix, row_blocks_is_correct_for_data_smaller_than_alignment_larger_than_block)
{
    // block < data < alignment

    basic_bit_matrix<u32, 64> bm(5, 65);

    // 65 bits, 3 blocks for data, 16 blocks for alignment

    auto const row_blocks = bm.row_blocks();

    EXPECT_EQ(16, row_blocks);
}


TEST(BitMatrix, row_blocks_is_correct_for_data_larger_than_block_with_equal_alignment)
{
    // (alignment == block) < data

    basic_bit_matrix<u32, 2> bm(5, 65);

    // 65 bits, 3 blocks for data, 3 blocks for alignment

    auto const row_blocks = bm.row_blocks();

    EXPECT_EQ(3, row_blocks);
}


TEST(BitMatrix, row_blocks_is_correct_for_data_larger_than_alignment_and_block)
{
    // block < alignment < data

    basic_bit_matrix<u32, 32> bm(5, 257);

    // 257 bits, 9 blocks for data, 16 blocks for alignment

    auto const row_blocks = bm.row_blocks();

    EXPECT_EQ(16, row_blocks);
}


TEST(BitMatrix, new_matrix_is_initialized_with_zeros)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    unsigned int rv = 0;

    for (auto rit = 0u; rit < NR; ++rit)
    {
        for (auto cit = 0u; cit < NC; ++cit)
        {
            rv |= bm[{rit, cit}];
        }
    }

    ASSERT_EQ(0, rv);
}


TEST(BitMatrix, first_bit_can_be_set)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 0);

    ASSERT_EQ(1, bm.test(0, 0));
}


TEST(BitMatrix, first_bit_can_be_cleared)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 0);

    bm.clear(0, 0);

    ASSERT_EQ(0, bm.test(0, 0));
}


TEST(BitMatrix, first_bit_set_can_be_flipped)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 0);

    bm.flip(0, 0);

    ASSERT_EQ(0, bm.test(0, 0));
}


TEST(BitMatrix, first_bit_clear_can_be_flipped)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.flip(0, 0);

    ASSERT_EQ(1, bm.test(0, 0));
}


TEST(BitMatrix, last_bit_can_be_set)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(1, 256);

    ASSERT_EQ(1, bm.test(1, 256));
}


TEST(BitMatrix, last_bit_can_be_cleared)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(1, 256);

    bm.clear(1, 256);

    ASSERT_EQ(0, bm.test(1, 256));
}


TEST(BitMatrix, last_bit_set_can_be_flipped)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(1, 256);

    bm.flip(1, 256);

    ASSERT_EQ(0, bm.test(1, 256));
}


TEST(BitMatrix, last_bit_clear_can_be_flipped)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.flip(1, 256);

    ASSERT_EQ(1, bm.test(1, 256));
}


TEST(BitMatrix, two_bits_in_block_are_set)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 1);
    bm.set(0, 4);

    ASSERT_EQ(1, bm.test(0, 4));
    ASSERT_EQ(1, bm.test(0, 1));
}


TEST(BitMatrix, one_bit_in_block_is_cleared)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 1);
    bm.set(0, 4);

    bm.clear(0, 1);

    ASSERT_EQ(1, bm.test(0, 4));
    ASSERT_EQ(0, bm.test(0, 1));
}


TEST(BitMatrix, one_bit_set_in_next_row_is_flipped)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 256);
    bm.set(1, 0);

    bm.flip(1, 0);

    ASSERT_EQ(1, bm.test(0, 256));
    ASSERT_EQ(0, bm.test(1, 0));
}


TEST(BitMatrix, one_bit_clear_in_next_row_is_flipped)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(0, 256);

    bm.flip(1, 0);

    ASSERT_EQ(1, bm.test(0, 256));
    ASSERT_EQ(1, bm.test(1, 0));
}


TEST(BitMatrix, set_bit_modifies_correct_block)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(NR - 1, 0);

    auto data = bm.data();
    auto block = data[bm.row_offset(NR - 1)];

    ASSERT_NE(0, block);
}


TEST(BitMatrix, set_bit_modifies_correct_block2)
{
    auto constexpr NR = 257u;
    auto constexpr NC = 2u;

    basic_bit_matrix<u32, 32> bm(NR, NC);

    bm.set(NR - 1, 0);

    auto data = bm.data();
    auto block = data[bm.row_offset(NR - 1)];

    ASSERT_NE(0, block);
}


} // anonymous namespace
