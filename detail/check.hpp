#ifndef REMORA_DETAIL_CHECK_HPP
#define REMORA_DETAIL_CHECK_HPP

#include <cassert>

#ifndef NDEBUG
#define REMORA_RANGE_CHECK(cond) assert(cond)
#define REMORA_SIZE_CHECK(cond) assert(cond)
#else
#define REMORA_RANGE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define REMORA_SIZE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#endif

#endif