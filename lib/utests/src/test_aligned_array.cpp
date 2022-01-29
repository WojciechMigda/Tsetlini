#include "aligned_array.hpp"

#include "boost/ut.hpp"

#include <cstdlib>
#include <memory>
#include <algorithm>
#include <sstream>


using namespace boost::ut;


/*
 * As of now, AlignedArray is not being used in Tsetlini.
 */
suite TestAlignedArray = []
{


"AlignedArray can be created"_test = []
{
    using array_type = AlignedArray<int, 8, 64>;
    std::stringstream side_effect;

    expect(nothrow([&]
        {
            array_type arr;

            side_effect << &arr;
        }));
};


"Data is aligned for AlignedArray declared on the stack"_test = []
{
    AlignedArray<int, 8, 64> const arr;

    auto const address = reinterpret_cast<std::size_t>(arr.data());
    auto const alignment = decltype(arr)::alignment;

    expect(that % 0u == address % alignment) << "AlignedArray data is misaligned";
};


"Data is aligned for AlignedArray declared as member of packed struct on the stack"_test = []
{
    #pragma pack(1)
    struct
    {
        char dummy;
        AlignedArray<int, 8, 64> arr;
    } s;

    auto const arr_address = reinterpret_cast<std::size_t>(&s.arr);

    expect(that % 1u == arr_address % 2) << "AlignedArray address is not odd";

    auto const data_address = reinterpret_cast<std::size_t>(s.arr.data());
    auto const alignment = decltype(s.arr)::alignment;

    expect(that % 0u == data_address % alignment) << "AlignedArray data is misaligned";
};


"Data is aligned for AlignedArray declared as member of packed struct on the heap"_test = []
{
    #pragma pack(1)
    struct S
    {
        char dummy;
        AlignedArray<int, 8, 64> arr;
    };

    auto sptr = std::make_unique<S>();

    auto const arr_address = reinterpret_cast<std::size_t>(&sptr->arr);

    expect(that % 1u == arr_address % 2) << "AlignedArray address is not odd";

    auto const data_address = reinterpret_cast<std::size_t>(sptr->arr.data());
    auto const alignment = decltype(sptr->arr)::alignment;

    expect(that % 0u == data_address % alignment) << "AlignedArray data is misaligned";
};


"Data alignment is not changed after AlignedArray is copied onto"_test = []
{
    using array_type = AlignedArray<int, 8, 64>;
    array_type arr;

    #pragma pack(1)
    struct
    {
        char dummy;
        array_type arr;
    } s;

    auto const src_data_address = reinterpret_cast<std::size_t>(arr.data());
    auto const src_alignment = decltype(arr)::alignment;

    expect(that % 0u == src_data_address % src_alignment) << "Source AlignedArray data is misaligned";

    auto const dst_arr_address = reinterpret_cast<std::size_t>(&s.arr);

    expect(that % 1u == dst_arr_address % 2) << "Destination AlignedArray address is not odd";

    auto const dst_data_address = reinterpret_cast<std::size_t>(s.arr.data());
    auto const dst_alignment = decltype(s.arr)::alignment;

    expect(that % 0u == dst_data_address % dst_alignment) << "Destination AlignedArray data is misaligned";

    s.arr = arr;

    auto const dst_data_address_after = reinterpret_cast<std::size_t>(s.arr.data());
    auto const dst_alignment_after = decltype(s.arr)::alignment;

    expect(that % 0u == dst_data_address_after % dst_alignment_after) << "Destination AlignedArray data became misaligned";
};


"AlignedArray elements can be read and written"_test = []
{
    auto constexpr NELEM = 8u;

    #pragma pack(1)
    struct
    {
        char dummy;
        AlignedArray<int, NELEM, 64> arr;
    } s;

    s.arr[0] = 5;
    expect(that % 5 == s.arr[0]);
    s.arr[0] = 7;
    expect(that % 7 == s.arr[0]);

    s.arr[NELEM - 1] = 15;
    expect(that % 15 == s.arr[NELEM - 1]);
    s.arr[NELEM - 1] = 17;
    expect(that % 17 == s.arr[NELEM - 1]);
};


"AlignedArray elements are copied during copy operation"_test = []
{
    auto constexpr NELEM = 8u;

    #pragma pack(1)
    struct
    {
        char dummy;
        AlignedArray<int, NELEM, 64> arr;
    } src, dst;

    std::fill_n(src.arr.data(), NELEM, 5);
    std::fill_n(dst.arr.data(), NELEM, 15);

    dst.arr = src.arr;

    expect(that % src.arr == dst.arr);
};


"AlignedArray can be copy constructed"_test = []
{
    auto constexpr NELEM = 8u;
    using array_type = AlignedArray<int, NELEM, 64>;

    #pragma pack(1)
    struct
    {
        char dummy;
        array_type arr;
    } s;

    std::fill_n(s.arr.data(), NELEM, 5);

    array_type d(s.arr);

    auto const data_address = reinterpret_cast<std::size_t>(s.arr.data());
    auto const alignment = decltype(s.arr)::alignment;

    expect(that % 0u == data_address % alignment) << "AlignedArray data is misaligned";
    expect(that % d == s.arr);
};


};

int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
