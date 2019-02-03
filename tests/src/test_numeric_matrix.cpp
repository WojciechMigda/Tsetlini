#include "basic_numeric_matrix.hpp"

#include <gtest/gtest.h>
#include <cstdint>


namespace
{

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;


TEST(BasicNumericMatrix, can_be_created)
{
    basic_numeric_matrix<unsigned int> nm(5, 17);
}


TEST(BasicNumericMatrix, shape_matches_ctor)
{
    basic_numeric_matrix<unsigned int> nm(5, 17);

    auto const [nrows, ncols] = nm.shape();
    ASSERT_EQ(5, nrows);
    ASSERT_EQ(17, ncols);
}


TEST(BasicNumericMatrix, row_items_is_correct_for_zero_columns)
{
    basic_numeric_matrix<u32, 64> nm(5, 0);

    auto const row_items = nm.row_items();

    EXPECT_EQ(0, row_items);
}


TEST(BasicNumericMatrix, row_items_is_correct_for_data_smaller_than_alignment)
{
    // data < alignment

    basic_numeric_matrix<u32, 64> nm(5, 3);

    auto const row_items = nm.row_items();

    EXPECT_EQ(16, row_items);
}


TEST(BasicNumericMatrix, row_items_is_correct_for_data_equal_to_alignment)
{
    // data == alignment

    basic_numeric_matrix<u32, 64> nm(5, 16);

    auto const row_items = nm.row_items();

    EXPECT_EQ(16, row_items);
}


TEST(BasicNumericMatrix, row_items_is_correct_for_data_larger_than_alignment)
{
    // alignment < data

    basic_numeric_matrix<u32, 32> nm(5, 9);

    auto const row_items = nm.row_items();

    EXPECT_EQ(16, row_items);
}


TEST(BasicNumericMatrix, new_matrix_is_initialized_with_zeros)
{
    auto constexpr NR = 2u;
    auto constexpr NC = 257u;

    basic_numeric_matrix<u32, 32> nm(NR, NC);

    u32 rv = 0;

    for (auto rit = 0u; rit < NR; ++rit)
    {
        for (auto cit = 0u; cit < NC; ++cit)
        {
            rv |= nm[{rit, cit}];
        }
    }

    ASSERT_EQ(0, rv);
}


} // anonymous namespace
