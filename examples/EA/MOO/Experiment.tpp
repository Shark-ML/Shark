/**
 *
 *  \brief Example setups for conducting EA experiments with the Shark library.
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
#include <shark/Algorithms/DirectSearch/InterruptibleAlgorithmRunner.h>
#include <shark/Algorithms/DirectSearch/Experiments/Experiment.h>
#include <shark/Algorithms/DirectSearch/Experiments/FrontStore.h>

#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
#include <shark/Algorithms/DirectSearch/SMS-EMOA.h>

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/units/systems/si.hpp>

#include <limits>

DEFINE_OPTION( OptimizerOptionTag, optimizer, std::string, "" );
DEFINE_SIMPLE_OPTION( ListOptimizersTag, listOptimizers );
DEFINE_SIMPLE_OPTION( ListObjectiveFunctionsTag, listObjectiveFunctions );

int main( int argc, char ** argv ) {

	shark::moo::Experiment::Options options;
	options.addDefaultOptions();
	
	options.addOption( OptimizerOptionTag() );
	options.addOption( ListOptimizersTag() );
	options.addOption( ListObjectiveFunctionsTag() );

	if( !options.parse( argc, argv ) ) {
		std::cerr << "Problem parsing command line." << std::endl;
		return( EXIT_FAILURE );
	}

	if( options.hasValue( ListOptimizersTag() ) ) {
		shark::moo::RealValuedMultiObjectiveOptimizerFactory::instance().print( std::cout );
		return( EXIT_SUCCESS );
	}

	if( options.hasValue( ListObjectiveFunctionsTag() ) ) {
		shark::moo::RealValuedObjectiveFunctionFactory::instance().print( std::cout );
		return( EXIT_SUCCESS );
	}

	if( options.value( OptimizerOptionTag() ).size() == 0 ) {
		// std::cerr << options << std::endl;
		return( EXIT_FAILURE );
	}	

	if( options.value( shark::moo::Experiment::Options::ObjectiveFunctionTag() ).size() == 0 ) {
		//std::cerr << options << std::endl;
		return( EXIT_FAILURE );
	}	

	boost::optional< boost::property_tree::ptree > configurationTree;

	if( !options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ).empty() ) {
		try {
			boost::property_tree::ptree pt;
			boost::property_tree::read_json( options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ), pt );
			configurationTree = pt;
		} catch( ... ) {
			std::cerr << "Problem reading algorithm configuration file: " << options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ) << std::endl;
			return( EXIT_FAILURE );
		}
	}
	
	std::string optimizerName  = options.value( OptimizerOptionTag() );
	std::string objectiveFunctionName = options.value( shark::moo::Experiment::Options::ObjectiveFunctionTag() );

	boost::shared_ptr< 
		shark::AbstractMultiObjectiveOptimizer< shark::VectorSpace< double > >
	> optimizer( shark::moo::RealValuedMultiObjectiveOptimizerFactory::instance()[ optimizerName ] );

	boost::shared_ptr<
		shark::MultiObjectiveFunction
	> objectiveFunction( shark::moo::RealValuedObjectiveFunctionFactory::instance()[ objectiveFunctionName ] );

	typedef shark::moo::InterruptibleAlgorithmRunner< 
		shark::AbstractMultiObjectiveOptimizer< shark::VectorSpace< double > >, 
		shark::MultiObjectiveFunction 
	> runner_type;
	runner_type abstractRunner( 
		optimizer,
		objectiveFunction
	);

	typedef shark::FrontStore< 
		runner_type::result_type,
		runner_type::ResultMetaData
	> front_store_type;
	front_store_type frontStore( options.value( shark::moo::Experiment::Options::ResultDirTag() ) );

	abstractRunner.signalResultsAvailable().connect( 
		boost::bind( 
			&front_store_type::onNewResult, 
			boost::ref( frontStore ),
			_1,
			_2
		)
	);

	try {
		abstractRunner.run( 
			options.value( shark::moo::Experiment::Options::SeedTag() ),
			options.value( shark::moo::Experiment::Options::StorageIntervalTag() ),
			options.value( shark::moo::Experiment::Options::SearchSpaceDimensionTag() ),
			options.value( shark::moo::Experiment::Options::ObjectiveSpaceDimensionTag() ),
			options.value( shark::moo::Experiment::Options::MaxNoEvaluationsTag() ),
			options.value( shark::moo::Experiment::Options::TimeLimitTag() ),
			configurationTree
			); 
	} catch( const shark::Exception & e ) {
		std::cerr << "Exception while running: " << e.what() << ": " << e.file() << "@" << e.line() << std::endl;
	}

	return( EXIT_SUCCESS );
}
