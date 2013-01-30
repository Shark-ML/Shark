#include "shark/LinAlg/eigenvalues.h"
#include <cmath>

#define BOOST_TEST_MODULE LinAlg_eigen
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;

const size_t Dimensions=2;

BOOST_AUTO_TEST_CASE( LinAlg_eigen )
{
	//we construct a rotation matrix for a given angle phi
	const double phi=M_PI/3;//60 degree

	RealMatrix R(Dimensions, Dimensions);   // rotation matrix
	//eigenvalues
	RealVector real(Dimensions, Dimensions);
	RealVector imag(Dimensions, Dimensions);

	double sinPhi=sin(phi);
	double cosPhi=cos(phi);

	R(0,0)=cosPhi;
	R(1,1)=cosPhi;
	R(0,1)=-sinPhi;
	R(1,0)=sinPhi;

	double resultReal[Dimensions]={cosPhi,cosPhi};
	double resultImag[Dimensions]={sinPhi,-sinPhi};

	//calculate complex eigenvalues
	eigen(R,real,imag);

	//test for equality
	for (size_t row = 0; row < Dimensions; row++)
	{
		BOOST_CHECK_SMALL(real(row)-resultReal[row],1.e-14);
		BOOST_CHECK_SMALL(imag(row)-resultImag[row],1.e-14);
	}
}
