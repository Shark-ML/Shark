//===========================================================================
/*!
 *
 *
 * \brief       Test for the PartlyPrecomputedMatrix class
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================


#define BOOST_TEST_MODULE LINALG_PARTLYPRECOMPUTEDMATRIX

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Timer.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Rng/GlobalRng.h>

#include <algorithm>

using namespace shark;


template <class InputType, class CacheType>
class DummyKernelMatrix
{
public:
	typedef CacheType QpFloatType;

	DummyKernelMatrix(size_t _size)
	{
		m_size = _size;
	}

	/// return a single matrix entry
	QpFloatType operator()(std::size_t i, std::size_t j) const
	{ return entry(i, j); }

	/// return a single matrix entry
	QpFloatType entry(std::size_t i, std::size_t j) const
	{
		return (QpFloatType)(i + 1) / (j + 1);
	}

	/// \brief Computes the i-th row of the kernel matrix.
	///
	///The entries start,...,end of the i-th row are computed and stored in storage.
	///There must be enough room for this operation preallocated.
	void row(std::size_t i, std::size_t start, std::size_t end, QpFloatType* storage) const
	{
		m_accessCounter += end - start;

		SHARK_PARALLEL_FOR(int j = start; j < (int) end; j++)
		{
			storage[j - start] = entry(i, j);
		}
	}


	/// return the size of the quadratic matrix
	std::size_t size() const
	{ return m_size; }

	/// query the kernel access counter
	unsigned long long getAccessCount() const
	{ return m_accessCounter; }

	/// reset the kernel access counter
	void resetAccessCount()
	{ m_accessCounter = 0; }

protected:
	// fake size of matrix
	size_t m_size;

	/// counter for the kernel accesses
	mutable unsigned long long m_accessCounter;
};


// FIXME: count accesses to GaussianRbfKernel to verify that the cache works.


///\brief test if the whole matrix fits, if we make the cache large enough
BOOST_AUTO_TEST_SUITE (LinAlg_PartlyPrecomputedMatrix)

BOOST_AUTO_TEST_CASE(LinAlg_PartlyPrecomputedMatrix_MediumCache)
{

	shark::Timer timer;
	Rng::seed(floor(timer.now()));
	// FIXME: need a test for 1x1?? ;)

	size_t maxDimension = 120;
	size_t maxGamma = 100;
	size_t repeats = 16;
	bool verbose = false;
	for(int i = 0; i < repeats; i++)
	{
		// some random dimension
		size_t currentDimension = Rng::discrete(2, maxDimension);

		if(verbose) std::cout << i << std::endl;
		if(verbose) std::cout << currentDimension << std::endl;

		// create unionjack matrix
		std::vector<RealVector> unionJack(currentDimension);
		RealVector unionJackVector(currentDimension);
		for(int r = 0; r < currentDimension; r++)
		{
			unionJackVector.clear();
			unionJackVector[currentDimension - r - 1] += 1;
			unionJackVector[r] += 1;
			unionJack[r] = unionJackVector;
		}

		Data<RealVector> unionJackData = createDataFromRange(unionJack);
		if(verbose) std::cout << ".." << std::endl;

		for(int r = 0; r < unionJackData.numberOfElements(); r++)
		{
			for(int c = 0; c < unionJackData.element(r).size(); c++)
			{
				if(verbose) std::cout << " " << unionJackData.element(r)[c];
			}
			if(verbose) std::cout << std::endl;
		}

		// compute a much bigger cachesize
		typedef KernelMatrix<RealVector, double > KernelMatrixType;
		size_t cacheSize = sizeof(double) * currentDimension * currentDimension / 2;
		if(verbose)  std::cout << "cache size: " << cacheSize << std::endl;

		// now cache this with a kernel with random gamma
		double gamma = Rng::discrete(1, maxGamma);
		if(verbose) std::cout << "g:" << gamma << std::endl;

		GaussianRbfKernel<> kernel(log(gamma));
		KernelMatrixType  km(kernel, unionJackData);
		PartlyPrecomputedMatrix<KernelMatrixType> K(&km, cacheSize);

		// now test for any row--
		// the kernel matrix should have the following form:
		// on the two crossing diagonals we have ones.
		// if its even: the rest of the entries are 1/gamma^4
		// if its odd: the row in the middle and the column in
		// the middle have 1/gamma^6
		size_t error = 0;
		for(int r = 0; r < currentDimension; r++)
		{
			for(int c = 0; c < currentDimension; c++)
			{
				size_t expectedEntry =  gamma * gamma * gamma * gamma;
				if((r == c) || (c == currentDimension - r - 1))
					expectedEntry = 1;
				if((currentDimension % 2 == 1) && ((r == currentDimension / 2) || (c == currentDimension / 2)))
					expectedEntry *= gamma * gamma;
				if(verbose) std::cout << " " << 1.0 / K.entry(r, c) << "/" << expectedEntry;
				error += 1.0 / K.entry(r, c) - expectedEntry;
			}
			if(verbose) std::cout << std::endl;
		}
		if(verbose) std::cout << "error: " << error << std::endl;;


		// get a whole row
		if(verbose) std::cout << " --- " << std::endl;
		int r = 1;
		blas::vector<double> kernelRow(currentDimension, 0);
		K.row(r, kernelRow);

		for(int c = 0; c < currentDimension; c++)
		{
			size_t expectedEntry =  gamma * gamma * gamma * gamma;
			if((r == c) || (c == currentDimension - r - 1))
				expectedEntry = 1;
			if((currentDimension % 2 == 1) && ((r == currentDimension / 2) || (c == currentDimension / 2)))
				expectedEntry *= gamma * gamma;
			if(verbose) std::cout << " " << 1.0 / K.entry(r, c) << "/" << expectedEntry;
			error += 1.0 / K.entry(r, c) - expectedEntry;
		}
		if(verbose) std::cout << std::endl;
		if(verbose) std::cout << " --- 8< ---" << std::endl;

		// this test might be not exact, and might seem nonsentical.
		// we still hope it should give reasonable timings.

		// find a cached and an uncached row

		// test if isCached is what we expected
		for(size_t r = 0; r < currentDimension; r++)
		{
			// FIXME: everything below X should not be cached

			if(K.isCached(r))
            {
				if(verbose) { std::cout << "+";}
            } else { 
                if(verbose) {std::cout << "-";}
            }
		}

		//FIXME: do this cache-rows counting

		if(verbose) std::cout << std::endl;

		size_t cachedRowIndex =  0;
		size_t uncachedRowIndex =  currentDimension - 1;

		// first test how many uncached rows we roughly can obtain in a certain given timespan.
		double timespan = 0.01;

		size_t uncachedRowEvalCount = 0;
		double uncachedStartTime = timer.now();
		do
		{
			// evaluate a whole uncached row
			uncachedRowEvalCount++;
			K.row(uncachedRowIndex, kernelRow);
		}
		while(timer.now() - uncachedStartTime < timespan);
		if(verbose) std::cout << "uncached: " << uncachedRowEvalCount;

		size_t cachedRowEvalCount = 0;
		double cachedStartTime = timer.now();
		do
		{
			// evaluate a whole uncached row
			cachedRowEvalCount++;
			K.row(cachedRowIndex, kernelRow);
		}
		while(timer.now() - cachedStartTime < timespan);
		if(verbose) std::cout << "cached: " << cachedRowEvalCount;
		if(verbose) std::cout << std::endl;




		// now that we know how many evaluations we can roughly do in a given timespan
		// we do this 10 timespans, but alternating

		double totalTime = timespan * 59;
		double cachedTotalTime = 0;
		double uncachedTotalTime = 0;
		double cachedTotalEvalCount = 0;
		double uncachedTotalEvalCount = 0;
		double startTime = timer.now();

		do
		{
			// evaluate N cached rows accesses
			cachedStartTime = timer.now();
			for(size_t p = 0; p < cachedRowEvalCount; p++)
				K.row(cachedRowIndex, kernelRow);
			cachedTotalTime += timer.now() - cachedStartTime ;
			cachedTotalEvalCount += cachedRowEvalCount;

			// evaluate N cached rows accesses
			uncachedStartTime = timer.now();
			for(size_t p = 0; p < uncachedRowEvalCount; p++)
				K.row(uncachedRowIndex, kernelRow);
			uncachedTotalTime += timer.now() - uncachedStartTime ;
			uncachedTotalEvalCount += uncachedRowEvalCount;
		}
		while(timer.now() - startTime < timespan);

		// now we hope to have some kind of balanced, unbiased
		// time and number of row evals in that time
		// we believe that cached access must be at least twice as fast
		// than uncached, given the dimension
		double uncachedSpeed = (double)uncachedTotalEvalCount / (double)uncachedTotalTime;
		double cachedSpeed = (double)cachedTotalEvalCount / (double)cachedTotalTime;
		if(verbose) std::cout << "uncached vs cached speed: " << uncachedSpeed << "/ " << cachedSpeed << " --> " << (double)cachedSpeed / (double)uncachedSpeed << std::endl;

		// this might be a 'stupid' test, but for all values i ever saw this was true:
		// the speed of the cache is always at least 10x faster than uncached
		// for the gamma and dimensions we choose here. so expect at least 3x speed
		BOOST_CHECK(cachedSpeed / uncachedSpeed >= 3.0);
	}
}



///\brief test if we can cache a gigantic matrix and still can access all rows
BOOST_AUTO_TEST_CASE(LinAlg_PartlyPrecomputedMatrix_GiganticKernel)
{

	shark::Timer timer;
	Rng::seed(floor(timer.now()));
	// FIXME: need a test for 1x1?? ;)

	size_t maxDimension = 120;
	size_t maxGamma = 100;
	size_t repeats = 2000;
	bool verbose = false;

	// create that 'gigantic' matrix
	size_t currentDimension = Rng::discrete(2, maxDimension) + 1000000;

	// compute a cache that should hold a few rows at most
	typedef DummyKernelMatrix<RealVector, double > KernelMatrixType;

	// allocate enough space for exactly 10 rows
	{
		size_t cacheSize = sizeof(double) * currentDimension * 10;
		KernelMatrixType kernel(currentDimension);
		PartlyPrecomputedMatrix<KernelMatrixType> K(&kernel, cacheSize);

		// see how many rows are actually cached
		size_t nCachedRows = 0;
		for(size_t r = 0; r < currentDimension; r++)
		{
			if(K.isCached(r))
				nCachedRows++;
		}

		// as we have allocated enough space for 10 rows, we expect 10 rows.
		BOOST_CHECK(nCachedRows == 10);
	}

	// allocate enough space just not enough 11 rows
	size_t cacheSize = sizeof(double) * currentDimension * 11 - 1;
	KernelMatrixType kernel(currentDimension);
	PartlyPrecomputedMatrix<KernelMatrixType> K(&kernel, cacheSize);

	// see how many rows are actually cached
	size_t nCachedRows = 0;
	for(size_t r = 0; r < currentDimension; r++)
	{
		if(K.isCached(r))
			nCachedRows++;
	}

	// as we have allocated enough space for  only 10 rows, we expect 10 rows.
	BOOST_CHECK(nCachedRows == 10);


	// do some random access
	for(int i = 0; i < repeats; i++)
	{
		// find a random entry
		size_t r = Rng::discrete(0, currentDimension - 1);
		size_t c = Rng::discrete(0, currentDimension - 1);

		BOOST_CHECK((K.entry(r, c) == (double)(r + 1) / (double)(c + 1)));
	}

}



BOOST_AUTO_TEST_SUITE_END()
