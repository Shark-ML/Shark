#include "shark/LinAlg/eigenvalues.h"

#define BOOST_TEST_MODULE LinAlg_eigensort
#include <boost/test/unit_test.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_eigensort )
{
	const size_t NumVecs=3;
	const size_t NumDim=3;

	RealMatrix x(NumVecs, NumDim);   // matrix of eigenvectors
	RealVector lambda(NumVecs); // vector of unsorted eigenvalues

	// Initialization values for matrix of eigenvectors:
	double eigenvectors[NumVecs][NumDim] =
	{
		{ 0.333333, 0.666667, 0.666667},
		{ 0.666667,-0.666667, 0.333333},
		{-0.666667,-0.333333, 0.666667}
	};
	// Initialization values for vector of eigenvalues:
	double eigenvalues[NumVecs] ={3., 9., 6.};

	//desired result after sorting
	double eigenvectorsSorted[NumVecs][NumDim] =
	{
		{ 0.666667,-0.666667, 0.333333},
		{-0.666667,-0.333333, 0.666667},
		{ 0.333333, 0.666667, 0.666667}
	};
	double eigenvaluesSorted[NumVecs] ={9., 6.,3.};

	// Initializing eigenvector matrix and eigenvalue vector:
	for (size_t vec = 0; vec < NumVecs; vec++)
	{
		for (size_t value = 0; value < NumDim; value++)
		{
			x(value, vec) = eigenvectors[vec][value];
		}
		lambda(vec) = eigenvalues[vec];
	}


	// Sorting eigenvectors and eigenvalues:
	eigensort(x, lambda);

	//test the sorting
	for (size_t vec = 0; vec < NumVecs; vec++)
	{
		for (size_t value = 0; value < NumDim; value++)
		{
			BOOST_CHECK_EQUAL(x(value, vec),eigenvectorsSorted[vec][value]);
		}
		BOOST_CHECK_EQUAL(lambda(vec),eigenvaluesSorted[vec]);
	}
}
