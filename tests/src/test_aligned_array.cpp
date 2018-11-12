#include "aligned_array.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <memory>

namespace
{


TEST(AlignedArray, can_be_created)
{
    using array_type = AlignedArray<int, 8, 64>;

    ASSERT_NO_FATAL_FAILURE(
        array_type const arr;
    );
}


TEST(AlignedArray, data_is_aligned_on_stack)
{
    AlignedArray<int, 8, 64> const arr;

    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(arr.data()) % decltype(arr)::alignment);
}


TEST(AlignedArray, data_is_aligned_as_packed_struct_on_stack)
{
    #pragma pack(1)
    struct
    {
        char dummy;
        AlignedArray<int, 8, 64> arr;
    } s;

    EXPECT_EQ(1u, reinterpret_cast<std::size_t>(&s.arr) % 2);
    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(s.arr.data()) % decltype(s.arr)::alignment);
}


TEST(AlignedArray, data_is_aligned_as_packed_struct_on_heap)
{
    #pragma pack(1)
    struct S
    {
        char dummy;
        AlignedArray<int, 8, 64> arr;
    };

    auto sp = std::make_unique<S>();

    EXPECT_EQ(1u, reinterpret_cast<std::size_t>(&sp->arr) % 2);
    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(sp->arr.data()) % decltype(sp->arr)::alignment);
}


TEST(AlignedArray, data_keeps_alignment_after_being_copied_onto)
{
    using array_type = AlignedArray<int, 8, 64>;
    array_type arr;

    #pragma pack(1)
    struct
    {
        char dummy;
        array_type arr;
    } s;

    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(arr.data()) % array_type::alignment);

    EXPECT_EQ(1u, reinterpret_cast<std::size_t>(&s.arr) % 2);
    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(s.arr.data()) % array_type::alignment);

    s.arr = arr;

    EXPECT_EQ(1u, reinterpret_cast<std::size_t>(&s.arr) % 2);
    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(s.arr.data()) % array_type::alignment);
}


TEST(AlignedArray, elements_can_be_read_and_written)
{
    constexpr auto nelem = 8u;

    #pragma pack(1)
    struct
    {
        char dummy;
        AlignedArray<int, nelem, 64> arr;
    } s;

    s.arr[0] = 5;
    EXPECT_EQ(5, s.arr[0]);
    s.arr[0] = 7;
    EXPECT_EQ(7, s.arr[0]);

    s.arr[nelem - 1] = 15;
    EXPECT_EQ(15, s.arr[nelem - 1]);
    s.arr[nelem - 1] = 17;
    EXPECT_EQ(17, s.arr[nelem - 1]);
}

TEST(AlignedArray, elements_are_copied_during_copy)
{
    constexpr auto nelem = 8u;

    #pragma pack(1)
    struct
    {
        char dummy;
        AlignedArray<int, nelem, 64> arr;
    } s, d;

    std::fill_n(s.arr.data(), nelem, 5);
    std::fill_n(d.arr.data(), nelem, 15);

    d.arr = s.arr;

    EXPECT_TRUE(std::equal(s.arr.data(), s.arr.data() + nelem, d.arr.data()));
}


TEST(AlignedArray, can_be_copy_constructed)
{
    constexpr auto nelem = 8u;
    using array_type = AlignedArray<int, nelem, 64>;

    #pragma pack(1)
    struct
    {
        char dummy;
        array_type arr;
    } s;

    std::fill_n(s.arr.data(), nelem, 5);

    array_type d(s.arr);

    EXPECT_EQ(0u, reinterpret_cast<std::size_t>(d.data()) % array_type::alignment);
    EXPECT_TRUE(std::equal(s.arr.data(), s.arr.data() + nelem, d.data()));
}


} // namespace
