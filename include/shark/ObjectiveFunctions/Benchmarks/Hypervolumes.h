#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_HYPERVOLUMES_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_HYPERVOLUMES_H

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

namespace shark {namespace benchmarks{

// The optimal hypervolumes for various population sizes.  All with a reference
// point of (11, 11).

#define HV(type, vol10, vol100)                                         \
	double optimal_hyper_volume(type const &, std::size_t mu) { \
	switch(mu){ \
	case 10: return vol10; \
	case 100: return vol100; \
	} \
	throw SHARKEXCEPTION("No known hypervolume for " #type \
	                     " for mu = " + std::to_string(mu)); \
	}

HV(DTLZ1, 120.86111, 120.873737)
HV(DTLZ2, 120.178966, 120.210644)
HV(DTLZ3, 120.178966, 120.210643)
HV(DTLZ4, 120.178966, 120.210634)
HV(DTLZ7, 115.964708, 116.101551)
HV(ZDT1, 120.613761, 120.662137)
HV(ZDT2, 120.286820, 120.328881)
HV(ZDT3, 128.748470, 128.775955)
HV(ZDT4, 120.613761, 120.662137)
HV(ZDT6, 117.483246, 117.514950)

#undef HV

}} // namespace shark

#endif
