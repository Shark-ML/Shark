#define BOOST_TEST_MODULE DirectSearch_Operators_ReferenceVectorAdaptation

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/ReferenceVectorAdaptation.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

#include <iostream>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_ReferenceVectorAdaptation)

BOOST_AUTO_TEST_CASE(updateAngles_correct)
{
	typedef Individual<RealVector, RealVector> I;
	const RealMatrix vecs{
		{0, 1}, 
		{std::cos(3*M_PI / 8), std::sin(3*M_PI / 8)},
		{1 / std::sqrt(2), 1 / std::sqrt(2)},
		{1, 0} };
	RealVector angles(4);
	ReferenceVectorAdaptation<I>::updateAngles(vecs, angles);
	BOOST_CHECK_EQUAL(angles.size(), 4);
	// The three upper vectors are all 22.5 degress from each other.
	BOOST_CHECK_EQUAL(angles[0], angles[1]);
	BOOST_CHECK_EQUAL(angles[0], angles[2]);
	BOOST_CHECK_CLOSE(angles[0], M_PI / 8, 1e-10);
	// The last vector is 45 degrees from the middle one.
	BOOST_CHECK_CLOSE(angles[3], M_PI / 4, 1e-10);
}

BOOST_AUTO_TEST_CASE(adaptVectors_correct)
{
	ReferenceVectorAdaptation<Individual<RealVector, RealVector>> adapt;
	const RealMatrix initVecs{
		{0, 1},
		{std::cos(3*M_PI / 8), std::sin(3*M_PI / 8)},
		{1 / std::sqrt(2), 1 / std::sqrt(2)},
		{std::cos(M_PI / 8), std::sin(M_PI / 8)},
		{1, 0} };
	RealMatrix vecs = initVecs;
	RealVector angles(vecs.size1());
	adapt.m_initVecs = initVecs;
	std::vector<Individual<RealVector, RealVector>> population(4);
	population[0].unpenalizedFitness() = RealVector{0, 0};
	population[1].unpenalizedFitness() = RealVector{1, 10};
	population[2].unpenalizedFitness() = RealVector{2, 20};
	population[3].unpenalizedFitness() = RealVector{3, 30};
	for(auto & p : population)
	{
		p.penalizedFitness() = p.unpenalizedFitness();
	}
	adapt(population, vecs, angles);
	
	const RealMatrix exp {
		{0, 1},
		{initVecs(1,0) * 3 / std::sqrt(std::pow(initVecs(1,0)*3, 2) +
		                               std::pow(initVecs(1,1)*30, 2)), 
		 initVecs(1, 1) * 30 / std::sqrt(std::pow(initVecs(1,0)*3, 2) +
		                                 std::pow(initVecs(1,1)*30, 2))},
		{initVecs(2,0) * 3 / std::sqrt(std::pow(initVecs(2,0)*3, 2) +
		                               std::pow(initVecs(2,1)*30, 2)), 
		 initVecs(2, 1) * 30 / std::sqrt(std::pow(initVecs(2,0)*3, 2) +
		                                 std::pow(initVecs(2,1)*30, 2))},
		{initVecs(3,0) * 3 / std::sqrt(std::pow(initVecs(3,0)*3, 2) +
		                               std::pow(initVecs(3,1)*30, 2)), 
		 initVecs(3, 1) * 30 / std::sqrt(std::pow(initVecs(3,0)*3, 2) +
		                                 std::pow(initVecs(3,1)*30, 2))},
		{1, 0} };
	BOOST_CHECK_EQUAL(vecs.size1(), exp.size1());
	BOOST_CHECK_EQUAL(vecs.size2(), exp.size2());
	for(std::size_t i = 0; i < vecs.size1(); ++i)
	{
		for(std::size_t j = 0; j < vecs.size2(); ++j)
		{
			BOOST_CHECK_CLOSE(vecs(i, j), exp(i, j), 1e-10);
		}
	}
}

BOOST_AUTO_TEST_CASE(adaptVectors_zeros_correct)
{
	ReferenceVectorAdaptation<Individual<RealVector, RealVector>> adapt;
	const RealMatrix initVecs{
		{0, 1},
		{std::cos(3*M_PI / 8), std::sin(3*M_PI / 8)},
		{1 / std::sqrt(2), 1 / std::sqrt(2)},
		{std::cos(M_PI / 8), std::sin(M_PI / 8)},
		{1, 0} };
	RealMatrix vecs = initVecs;
	RealVector angles(vecs.size1());
	adapt.m_initVecs = initVecs;
	std::vector<Individual<RealVector, RealVector>> population(4);
	population[0].unpenalizedFitness() = RealVector{1, 0};
	population[1].unpenalizedFitness() = RealVector{1, 10};
	population[2].unpenalizedFitness() = RealVector{1, 20};
	population[3].unpenalizedFitness() = RealVector{1, 30};
	for(auto & p : population)
	{
		p.penalizedFitness() = p.unpenalizedFitness();
	}
	adapt(population, vecs, angles);
	
	const RealMatrix exp {
		{0, 1},
		{initVecs(1, 0) / std::sqrt(std::pow(initVecs(1,0), 2) +
		                            std::pow(initVecs(1,1)*30, 2)),
		 initVecs(1, 1) * 30 / std::sqrt(std::pow(initVecs(1,0), 2) +
		                                 std::pow(initVecs(1,1)*30, 2))},
		{initVecs(2, 0) / std::sqrt(std::pow(initVecs(2,0), 2) +
		                                 std::pow(initVecs(2,1)*30, 2)),
		 initVecs(2, 1) * 30 / std::sqrt(std::pow(initVecs(2,0), 2) +
		                                 std::pow(initVecs(2,1)*30, 2))},
		{initVecs(3, 0) / std::sqrt(std::pow(initVecs(3,0), 2) +
		                                 std::pow(initVecs(3,1)*30, 2)),
		 initVecs(3, 1) * 30 / std::sqrt(std::pow(initVecs(3,0), 2) +
		                                 std::pow(initVecs(3,1)*30, 2))},
		{1, 0} };
	BOOST_CHECK_EQUAL(vecs.size1(), exp.size1());
	BOOST_CHECK_EQUAL(vecs.size2(), exp.size2());
	for(std::size_t i = 0; i < vecs.size1(); ++i)
	{
		for(std::size_t j = 0; j < vecs.size2(); ++j)
		{
			BOOST_CHECK_CLOSE(vecs(i, j), exp(i, j), 1e-10);
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()

