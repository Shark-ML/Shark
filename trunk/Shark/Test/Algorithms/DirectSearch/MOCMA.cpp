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


//~ BOOST_AUTO_TEST_CASE( MOCMA_CHROMOSOME_SERIALIZATION ) {

	//~ mocma::Chromosome chromosome1, chromosome2;
	//~ BOOST_CHECK(			chromosome1.m_mutationDistribution.covarianceMatrix().size1() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_mutationDistribution.covarianceMatrix().size2() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_mutationDistribution.eigenValues().size() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_mutationDistribution.eigenVectors().size1() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_mutationDistribution.eigenVectors().size2() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_evolutionPath.size() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_lastStep.size() == 0 );
	//~ BOOST_CHECK(			chromosome1.m_lambda == 0 );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_noSuccessfulOffspring, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_stepSize, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_stepSizeDampingFactor, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_stepSizeLearningRate, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_successProbability, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_targetSuccessProbability, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_evolutionPathLearningRate, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_covarianceMatrixLearningRate, 0. );
	//~ BOOST_CHECK_EQUAL(		chromosome1.m_needsCovarianceUpdate, false );

	//~ chromosome1.m_mutationDistribution.resize( 10 );
	//~ chromosome1.m_evolutionPath = blas::repeat(0.0,10);
	//~ chromosome1.m_lastStep = blas::repeat(0.0,10);
	//~ chromosome1.m_lambda = 5;
	//~ chromosome1.m_noSuccessfulOffspring = 5.;
	//~ chromosome1.m_stepSize = 5.;
	//~ chromosome1.m_stepSizeDampingFactor = 5.;
	//~ chromosome1.m_stepSizeLearningRate = 5.;
	//~ chromosome1.m_successProbability = 5.;
	//~ chromosome1.m_targetSuccessProbability = 5.;
	//~ chromosome1.m_evolutionPathLearningRate = 5.;
	//~ chromosome1.m_covarianceMatrixLearningRate = 5.;
	//~ chromosome1.m_needsCovarianceUpdate = true;
	
	//~ {
		//~ std::stringstream ss;
		//~ boost::archive::text_oarchive oa( ss );
		//~ BOOST_CHECK_NO_THROW( (oa << chromosome1) );

		//~ boost::archive::text_iarchive ia( ss );
		//~ BOOST_CHECK_NO_THROW( (ia >> chromosome2) );

		//~ BOOST_CHECK( chromosome1 == chromosome2 );
	//~ }
//~ }

//~ BOOST_AUTO_TEST_CASE( MOCMA_SERIALIZER_SERIALIZATION ) {
	//~ mocma::Chromosome chromosome1, chromosome2;
	//~ mocma::Initializer initializer1, initializer2;
	//~ BOOST_CHECK_EQUAL( initializer1.m_searchSpaceDimension, 0 );
	//~ BOOST_CHECK_EQUAL( initializer1.m_noObjectives, 0 );
	//~ BOOST_CHECK_CLOSE( initializer1.m_initialSigma, 0., 1E-10 );
	//~ BOOST_CHECK_EQUAL( initializer1.m_useNewUpdate, false );
	//~ BOOST_CHECK_EQUAL( initializer1.m_constrainedFitnessFunction, false );

	//~ BOOST_CHECK_THROW( initializer1( chromosome1 ), Exception );
	//~ initializer1.m_searchSpaceDimension = 5;
	//~ BOOST_CHECK_THROW( initializer1( chromosome1 ), Exception );
	//~ initializer1.m_noObjectives = 5;
	//~ BOOST_CHECK_NO_THROW( initializer1( chromosome1 ) );
	//~ initializer1.m_initialSigma = 5;
	//~ initializer1.m_useNewUpdate = true;
	//~ initializer1.m_constrainedFitnessFunction = true;
	//~ {
		//~ std::stringstream ss;
		//~ boost::archive::text_oarchive oa( ss );
		//~ BOOST_CHECK_NO_THROW( (oa << initializer1) );

		//~ boost::archive::text_iarchive ia( ss );
		//~ BOOST_CHECK_NO_THROW( (ia >> initializer2) );

		//~ BOOST_CHECK( initializer1 == initializer2 );
	//~ }
//~ }


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
		MOCMA::SolutionSetType set1 = mocma.solution();
		Rng::seed( 1 );
		FastRng::seed( 1 );
		mocma2.step( dtlz1 );
		MOCMA::SolutionSetType set2 = mocma2.solution();

		for( unsigned int i = 0; i < set1.size(); i++ ) {
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).value - set2.at( i ).value ), 1E-20 );
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).point- set2.at( i ).point), 1E-20 );
		}

	}
}
