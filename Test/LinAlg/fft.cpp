#include "shark/LinAlg/fft.h"

#define BOOST_TEST_MODULE LinAlg_fft
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

// Simple test function which is sampled:
double func(double t)
{
	return t * t;
}


BOOST_AUTO_TEST_CASE( LinAlg_fft )
{
	ComplexMatrix data(16,1);  // f( t ) for all sample points t.


	// Create and save sample data (as real value < ft, 0 >):
	for (int t = -8; t < 8; t++)
	{
		std::complex< double > ft = func((double)t);
		data(t+8,0)=ft;
	}

	// Map function from time domain to frequency domain:
	fft(data);

	// Map function from frequency domain to time domain
	// (inverse fourier transformation):
	ifft(data);

	BOOST_CHECK_SMALL(fabs(data(0,0).real() - 64.0) + fabs(data(0,0).imag()),1.e-15);
}
