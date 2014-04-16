#define BOOST_TEST_MODULE DirectSearch_FastNonDominatedSort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

//check that the relation is correct on a number of selected points
BOOST_AUTO_TEST_CASE( FastNonDominatedSort_Selected_Points_Test ) {
	RealVector p0(3);
	p0[0] = -1; p0[1] = 0; p0[2] = 1;
	RealVector p1(3);//strictly dominated by p0
	p1[0] = 1; p1[1] = 2; p1[2] = 3;
	RealVector p2(3);//weakly dominated by p0
	p2[0] = -1; p2[1] = 2; p2[2] = 3;
	RealVector p3(3);//trade off with p0
	p3[0] = -2; p3[1] = 0; p3[2] = 2;
	
	typedef ParetoDominanceComparator<IdentityFitnessExtractor> Dominance;
	Dominance pdc;
	
	BOOST_CHECK_EQUAL(pdc(p0,p0), Dominance::A_EQUALS_B);
	BOOST_CHECK_EQUAL(pdc(p0,p3), Dominance::TRADE_OFF);
	BOOST_CHECK_EQUAL(pdc(p0,p1), Dominance::A_STRICTLY_DOMINATES_B);
	BOOST_CHECK_EQUAL(pdc(p1,p0), Dominance::B_STRICTLY_DOMINATES_A);
	BOOST_CHECK_EQUAL(pdc(p0,p2), Dominance::A_WEAKLY_DOMINATES_B);
	BOOST_CHECK_EQUAL(pdc(p2,p0), Dominance::B_WEAKLY_DOMINATES_A);
}

//randomly creates populations of individuals, sorts them and checks that the domination relation is 
//correct
BOOST_AUTO_TEST_CASE( FastNonDominatedSort_Random_Test ) {
	std::size_t numPoints = 20;
	std::size_t numTrials = 10;
	std::size_t numDims = 3;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		//create points
		std::vector<RealVector > population(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i][j]= Rng::uni(-1,2);
			}
		}
		
		typedef ParetoDominanceComparator<IdentityFitnessExtractor> Dominance;
		Dominance pdc;
		
		//check that ranks are okay
		for(std::size_t i = 0; i != numPoints; ++i){
			for(std::size_t j = 0; j != numPoints; ++j){
				//test all 6 results
				int comp = pdc(population[i],population[j]);
				if( comp == Dominance::A_STRICTLY_DOMINATES_B){
					for(std::size_t k = 0; k != numDims; ++k){
						BOOST_CHECK(population[i][k] < population[j][k]);
					}
				}
				if( comp == Dominance::B_STRICTLY_DOMINATES_A){
					for(std::size_t k = 0; k != numDims; ++k){
						BOOST_CHECK(population[j][k] < population[i][k]);
					}
				}
				if( comp == Dominance::A_WEAKLY_DOMINATES_B){
					bool equality = false;
					for(std::size_t k = 0; k != numDims; ++k){
						BOOST_CHECK(population[i][k] <= population[j][k]);
						if(population[i][k]==population[j][k])
							equality=true;
						
						BOOST_CHECK_EQUAL(equality,true);
					}
				}
				if( comp == Dominance::B_WEAKLY_DOMINATES_A){
					bool equality = false;
					for(std::size_t k = 0; k != numDims; ++k){
						BOOST_CHECK(population[j][k] >= population[i][k]);
						if(population[j][k]==population[i][k])
							equality=true;
						
						BOOST_CHECK_EQUAL(equality,true);
					}
				}
				
				if( comp == Dominance::A_EQUALS_B){
					for(std::size_t k = 0; k != numDims; ++k){
						BOOST_CHECK(population[i][k] == population[j][k]);
					}
				}
				if( comp == Dominance::TRADE_OFF){
					bool hasGreater = false;
					bool hasSmaller = false;
					for(std::size_t k = 0; k != numDims; ++k){
						if(population[j][k] > population[i][k])
							hasGreater = true;
						if(population[j][k] < population[i][k])
							hasSmaller = true;
					}
					BOOST_CHECK_EQUAL(hasGreater, true);
					BOOST_CHECK_EQUAL(hasSmaller, true);
				}
			}
		}
	}
}