/*******************************************************************************
 * Copyright (c) 2016 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the MIT License
 *******************************************************************************
 *
 * Filename: likely.h
 *
 * Description:
 *      LIKELY and LIKELY macros which utilize GCC extension to hint
 *      branch probability
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2016-05-13   wm              Initial version
 *
 ******************************************************************************/


#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#if (defined __GNUC__ && __GNUC__ >= 3)
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif


#ifdef __cplusplus
}
#endif
