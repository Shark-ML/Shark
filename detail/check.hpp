#ifndef REMORA_DETAIL_CHECK_HPP
#define REMORA_DETAIL_CHECK_HPP

#include <cassert>

#ifndef NDEBUG
#define RANGE_CHECK(cond) assert(cond)
#define SIZE_CHECK(cond) assert(cond)
#else
#define RANGE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define SIZE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#endif

#endif