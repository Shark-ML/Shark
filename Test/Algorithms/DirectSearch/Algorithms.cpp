/**
 *
 *  \brief Generic test for single- and multi-objective optimizers.
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <shark/Core/Shark.h>

#define BOOST_TEST_MODULE ObjectiveFunctions_Benchmarks
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>

#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/progress.hpp>
#include <boost/serialization/vector.hpp>

#include <shark/Algorithms/DirectSearch/Experiments/Experiment.h>
#include <shark/Algorithms/DirectSearch/InterruptibleAlgorithmRunner.h>
#include <shark/Algorithms/DirectSearch/Experiments/FrontStore.h>

#include <shark/Algorithms/DirectSearch/AGE.h>
#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
#include <shark/Algorithms/DirectSearch/SMS-EMOA.h>
#include <shark/Algorithms/DirectSearch/RealCodedNSGAII.h>


#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/HypervolumeApproximator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>

#include <shark/Core/Chart.h>
#include <shark/Core/Renderers/HighchartRenderer.h>

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <shark/Statistics/Statistics.h>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <fstream>

namespace shark {

	template<
		typename IndicatorType,
		typename FunctionType,
		typename ResultType,
		typename MetaDataType
	> struct PerformanceIndicatorStore {
	public:
		typedef IndicatorType indicator_type;
		typedef FunctionType function_type;
		typedef MultiObjectiveFunctionTraits< function_type > function_traits;
		typedef typename function_traits::ParetoFrontType front_type;
		typedef ResultType result_type;
		typedef MetaDataType result_meta_data_type;

		indicator_type m_indicator;
		front_type m_referenceFront;

		//std::map< 

		PerformanceIndicatorStore( 
			std::size_t searchSpaceDimension, 
			std::size_t objectiveSpaceDimension ) : m_referenceFront( function_traits::referenceFront( 100, searchSpaceDimension, objectiveSpaceDimension ) ) {
		}

		void onNewResult ( 
			SHARK_ARGUMENT( const result_type & front, "Actual result announced to the outside world" ),
			SHARK_ARGUMENT( const result_meta_data_type & meta, "Meta data describing the results" )
			) {
				static shark::IdentityFitnessExtractor ife;
				std::cout << "Indicator: " << m_indicator( front, m_referenceFront, ife ) << std::endl;
		}
	};

}

BOOST_AUTO_TEST_CASE( MultiObjective_DirectSearch_Algorithms ) {

	shark::Shark::init( 
		boost::unit_test::framework::master_test_suite().argc,
		boost::unit_test::framework::master_test_suite().argv 
	);
	shark::Shark::info( std::cout );

	shark::moo::Experiment::Options options;
	options.addDefaultOptions();
	options.parse( shark::Shark::argc(), shark::Shark::argv() );

	shark::moo::RealValuedMultiObjectiveOptimizerFactory::instance().print( std::cout );

	typedef shark::moo::RealValuedMultiObjectiveOptimizerFactory::class_type optimizer_type;

	shark::moo::RealValuedMultiObjectiveOptimizerFactory::const_iterator it;
	for( it = shark::moo::RealValuedMultiObjectiveOptimizerFactory::instance().begin();
		it != shark::moo::RealValuedMultiObjectiveOptimizerFactory::instance().end();
		++it 
	) {
		BOOST_TEST_MESSAGE( "Considering function: " << it->first );
		BOOST_CHECK( it->second != NULL );
		boost::shared_ptr< optimizer_type > optimizer( it->second->create() );

		BOOST_CHECK( optimizer );

		typedef shark::moo::InterruptibleAlgorithmRunner<
			optimizer_type,
			shark::ELLI1
		> runner_type;
		runner_type runner (
			optimizer,
			boost::shared_ptr<
				shark::ELLI1
			>( new shark::ELLI1() )
		);

		typedef shark::FrontStore< 
			runner_type::result_type,
			runner_type::ResultMetaData
		> front_store_type;
		front_store_type frontStore( it->first );
		runner.signalResultsAvailable().connect( 
			boost::bind( 
				&front_store_type::onNewResult, 
				boost::ref( frontStore ),
				_1,
				_2
			)
		);
		
		BOOST_CHECK_NO_THROW( 
			runner.run( 
				options.value( shark::moo::Experiment::Options::SeedTag() ),
				options.value( shark::moo::Experiment::Options::StorageIntervalTag() ),
				options.value( shark::moo::Experiment::Options::SearchSpaceDimensionTag() ),
				options.value( shark::moo::Experiment::Options::ObjectiveSpaceDimensionTag() ),
				10000,//options.value( shark::moo::Experiment::Options::MaxNoEvaluationsTag() ),
				1/30//options.value( shark::moo::Experiment::Options::TimeLimitTag() )
			)
		);

	}

}