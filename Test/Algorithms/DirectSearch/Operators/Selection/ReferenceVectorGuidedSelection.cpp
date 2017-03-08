#define BOOST_TEST_MODULE DirectSearch_Operators_ReferenceVectorGuidedSelection

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>

#include <iostream>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_ReferenceVectorGuidedSelection)

BOOST_AUTO_TEST_CASE(populationPartition_correct)
{
	typedef ReferenceVectorGuidedSelection<shark::Individual<RealVector, RealVector>> rv;
	// Three unit vectors
	const double sq = std::sqrt(2);
    const RealMatrix refVecs{
	    {0, 1},
	    {1 / sq, 1/ sq},
	    {1, 0}
    };
	
	const RealMatrix fitness{
		// Group 1
		{1 / sq, 1 / sq},
		{1 / sq - 0.1, 1 / sq + 0.1},
		// Group 0
		{0, 1},
		{0.1, 0.9},
		// Group 2
		{1, 0},
		{0.9, 0.1}
	};

	const std::vector<std::set<std::size_t>> groups = rv::populationPartition(rv::cosAngles(fitness, refVecs));

	std::set<std::size_t> exp0{2, 3};
	std::set<std::size_t> exp1{0, 1};
	std::set<std::size_t> exp2{4, 5};
	BOOST_CHECK(groups[0] == exp0);
	BOOST_CHECK(groups[1] == exp1);
	BOOST_CHECK(groups[2] == exp2);
}

BOOST_AUTO_TEST_CASE(objectiveValueTranslation_correct)
{
	typedef ReferenceVectorGuidedSelection<shark::Individual<RealVector, RealVector>> rv;
	const RealMatrix fitness{
		{12, 45}, 
		{87, 2}, 
		{23, 1}};
	const RealVector mins = rv::minCol(fitness);

	BOOST_CHECK_EQUAL(mins.size(), fitness.size2());
	BOOST_CHECK_EQUAL(mins[0], 12);
	BOOST_CHECK_EQUAL(mins[1], 1);
	
	const RealMatrix trans = fitness - repeat(mins, fitness.size1());
	BOOST_CHECK_EQUAL(trans.size1(), fitness.size1());
	BOOST_CHECK_EQUAL(trans.size2(), fitness.size2());
	const RealMatrix exp{{0, 44}, {75, 1}, {11, 0}};
	for(std::size_t i = 0; i < trans.size1(); ++i)
	{
		for(std::size_t j = 0; j < trans.size2(); ++j)
		{
			BOOST_CHECK_EQUAL(trans(i, j), exp(i, j));
		}
	}	
}

BOOST_AUTO_TEST_CASE(cosAngles_correct)
{
	typedef ReferenceVectorGuidedSelection<shark::Individual<RealVector, RealVector>> rv;
	const RealMatrix vecs{{0,1}, {1,0}};
	const RealMatrix f{{12, 12}};
	const RealMatrix angles = acos(rv::cosAngles(f, vecs));
	
	BOOST_CHECK_EQUAL(angles.size1(), f.size1());
	BOOST_CHECK_EQUAL(angles.size2(), vecs.size1());

	BOOST_CHECK_EQUAL(angles(0, 0), angles(0, 1));
	BOOST_CHECK_CLOSE(angles(0,0), M_PI / 4, 1e-9);
}


BOOST_AUTO_TEST_SUITE_END()

