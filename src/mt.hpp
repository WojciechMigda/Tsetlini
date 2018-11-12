/*******************************************************************************
 * Copyright (c) 2018 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the MIT License
 *******************************************************************************
 *
 * Filename: likely.h
 *
 * Description:
 *      SIMD-friendly Mersenne Twister PRNG
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2018-10-13   wm              Initial version
 *
 ******************************************************************************/

#pragma once

#include "aligned_array.hpp"
#include "likely.h"

#include <type_traits>

template<typename ValueType, unsigned int Alignment, unsigned int NumberOfStreams, typename DerivedT>
struct BasePRNG
{
    using value_type = ValueType;
    using derived_type = DerivedT;

    static constexpr unsigned int MTSZ = 624;
    static constexpr unsigned int alignment = Alignment;
    static constexpr unsigned int NS = NumberOfStreams;


    unsigned int index;

    AlignedArray<value_type, MTSZ * NS, alignment> aRES;
    AlignedArray<unsigned int, MTSZ * NS, alignment> aMT;

    inline
    BasePRNG(int seed=1)
    {
        init(seed);
    }

    inline
    void init(int seed=1)
    {
        auto MT = aMT.data();

        for (auto it = 0u; it < NS; ++it)
        {
            MT[it] = it + seed;
        }

        for (unsigned int i = 1 * NS; i < MTSZ * NS; ++i)
        {
            MT[i] = (1812433253UL * (MT[i - NS] ^ (MT[i - NS] >> 30)) + i / NS);
        }
        index = 0;
    }

    void generate()
    {
        auto MULT1 = 2567483615UL;
        auto MT = aMT.data();
        auto RES = aRES.data();

        for (unsigned int i = 0; i < 227 * NS; ++i)
        {
            auto y = (MT[i] & 0x8000000UL) + (MT[i + NS] & 0x7FFFFFFFUL);

            MT[i] = MT[i + 397 * NS] ^ (y >> 1) ^ (y & 1 ? MULT1 : 0);
        }
        for (unsigned int i = 227 * NS; i < (MTSZ - 1) * NS; ++i)
        {
            auto y = (MT[i] & 0x8000000UL) + (MT[i + NS] & 0x7FFFFFFFUL);

            MT[i] = MT[i - 227 * NS] ^ (y >> 1) ^ (y & 1 ? MULT1 : 0);
        }

        for (auto it = 0u; it < NS; ++it)
        {
            auto y = (MT[(MTSZ - 1) * NS + it] & 0x8000000UL) + (MT[0 + it] & 0x7FFFFFFFUL);
            MT[(MTSZ - 1) * NS + it] = MT[(MTSZ - 1 - 227) * NS + it] ^ (y >> 1) ^ (y & 1 ? MULT1 : 0);
        }

        for (auto it = 0u; it < MTSZ * NS; ++it)
        {
            auto y = MT[it];
            y ^= y >> 11;
            y ^= y << 7  & 2636928640UL;
            y ^= y << 15 & 4022730752UL;
            y ^= y >> 18;
            RES[it] = static_cast<derived_type *>(this)->post_process(y);
        }
    }

    inline
    value_type rand()
    {
        if (UNLIKELY(index == 0))
        {
            generate();
        }

        value_type y = aRES[index];

        if (UNLIKELY(index == MTSZ * NS - 1))
        {
            index = 0;
        }
        else
        {
            ++index;
        }

        return y;
    }

    inline
    value_type next()
    {
        return rand();
    }
};


template<unsigned int Alignment=64, unsigned int NumberOfStreams=8>
struct basic_IRNG : public BasePRNG<unsigned int, Alignment, NumberOfStreams, basic_IRNG<Alignment, NumberOfStreams>>
{
    static constexpr unsigned int alignment = Alignment;
    static constexpr unsigned int number_of_streams = NumberOfStreams;

    using base_type = BasePRNG<unsigned int, Alignment, NumberOfStreams, basic_IRNG<Alignment, NumberOfStreams>>;
    using value_type = typename base_type::value_type;

    basic_IRNG(int seed=1) : base_type(seed)
    {
    }

    inline
    value_type next()
    {
        return base_type::next();
    }

    inline
    value_type next(unsigned int x)
    {
        return base_type::rand() % x;
    }

    inline
    value_type next(int a, int b)
    {
        return a + (base_type::rand() % (b + 1 - a));
    }

    inline
    value_type post_process(unsigned int y)
    {
        return y;
    }
};

using IRNG = basic_IRNG<128, 8>;


template<unsigned int Alignment=64, unsigned int NumberOfStreams=8>
struct basic_FRNG : public BasePRNG<float, Alignment, NumberOfStreams, basic_FRNG<Alignment, NumberOfStreams>>
{
    static constexpr unsigned int alignment = Alignment;
    static constexpr unsigned int number_of_streams = NumberOfStreams;

    using base_type = BasePRNG<float, Alignment, NumberOfStreams, basic_FRNG<Alignment, NumberOfStreams>>;
    using value_type = typename base_type::value_type;

    basic_FRNG(int seed=1) : base_type(seed)
    {
    }

    inline
    value_type post_process(unsigned int y)
    {
        return (y + 0.5f) * (1.0f / 4294967296.0f);
    }
};

using FRNG = basic_FRNG<128, 8>;
