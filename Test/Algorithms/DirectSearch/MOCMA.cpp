/*!
 * 
 *
 * \brief       Unit tests for class ()MOCMA.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#define BOOST_TEST_MODULE DirectSearch_CMA
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/MOCMA.h>

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;

struct PointExtractor{

template<class T>
RealVector const& operator()(T const& arg)const{
	return arg.value;
}
};


BOOST_AUTO_TEST_CASE( MOCMA_HYPERVOLUME_ZDT1 ) {
	Rng::seed(42);
	ZDT1 objective(5);
	MOCMA mocma;
	mocma.notionOfSuccess() = shark::MOCMA::IndividualBased;
	mocma.mu() = 10;
	mocma.init( objective);
	
	for(std::size_t i = 0; i != 10000; ++i){
		mocma.step(objective);
		std::clog<<"\r"<<i<<std::flush;
	}
	BOOST_REQUIRE_EQUAL(mocma.solution().size(), 10);
	for(std::size_t i = 0; i != 10; ++i){
		std::cout<<mocma.solution()[i].value(0)<<" "<<mocma.solution()[i].value(1)<<std::endl;
	}
	HypervolumeCalculator hyp;
	RealVector reference(2);
	reference(0) = 11;
	reference(1) = 11;
	double volume = hyp(PointExtractor(),mocma.solution(),reference);
	std::cout<<volume<<std::endl;
	BOOST_CHECK_SMALL(volume - 120.613761, 5.e-3);
	
}


BOOST_AUTO_TEST_CASE( MOCMA_SERIALIZATION ) {
	MOCMA mocma;

	DTLZ1 dtlz1;
	dtlz1.setNumberOfObjectives( 3 );
	dtlz1.setNumberOfVariables( 10 );
	BOOST_CHECK_NO_THROW( mocma.init( dtlz1 ) );
	BOOST_CHECK_NO_THROW( mocma.step( dtlz1 ) );
	
	{
		std::stringstream ss;
		boost::archive::text_oarchive oa( ss );

		BOOST_CHECK_NO_THROW( (oa << mocma) );

		MOCMA mocma2;

		boost::archive::text_iarchive ia( ss );
		BOOST_CHECK_NO_THROW( (ia >> mocma2) );

		Rng::seed( 1 );
		FastRng::seed( 1 );
		mocma.step( dtlz1 );
		MOCMA::SolutionType set1 = mocma.solution();
		Rng::seed( 1 );
		FastRng::seed( 1 );
		mocma2.step( dtlz1 );
		MOCMA::SolutionType set2 = mocma2.solution();

		for( unsigned int i = 0; i < set1.size(); i++ ) {
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).value - set2.at( i ).value ), 1E-20 );
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).point- set2.at( i ).point), 1E-20 );
		}

	}
}
