#include <shark/Algorithms/DirectSearch/InterruptibleAlgorithmRunner.h>
#include <shark/Algorithms/DirectSearch/AGE2.h>

#include <shark/ObjectiveFunctions/Benchmarks/DTLZ1.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ2.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ3.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ4.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ5.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ6.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ7.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/units/systems/si.hpp>


#include <limits>

#include <omp.h>

int main( int argc, char ** argv ) {
	
	boost::program_options::options_description options;
	options.add_options()
		( "objectiveFunction", boost::program_options::value< std::string >(), "Name of the objective function." )
		( "seed", boost::program_options::value< unsigned int >()->default_value( 1 ), "Seed for the random number generator" )
		( "storageInterval", boost::program_options::value< unsigned int >()->default_value( 100 ), "Results are reported to the console for every i-th fitness function evaluation" )
		( "searchSpaceDimension", boost::program_options::value< unsigned int >()->default_value( 10 ), "Dimension n of the search space" )
		( "objectiveSpaceDimension", boost::program_options::value< unsigned int >()->default_value( 2 ), "Dimension m of the objective space" )
		( "maxNoEvaluations", boost::program_options::value< unsigned int >()->default_value( 50000 ), "Maximum number of fitness function evaluations as stopping criterion" )
		( "timeLimit", boost::program_options::value< unsigned int >()->default_value( 1000 ), "Time limit as stopping criterion, unit: hours" )
		( "resultDir", boost::program_options::value< std::string >()->default_value( "." ), "Directory to put results into" )
		( "algorithmUsage", "Reports the configuration options of the algorithm." )
		( "algorithmConfigFile", boost::program_options::value< std::string >(), "JSON config file for the algorithm" );

	boost::program_options::variables_map vm;
	try {		
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), vm);
		boost::program_options::notify(vm);
	} catch( ... ) {
		std::cerr << options << std::endl;
		return( EXIT_FAILURE );
	}

	if( vm.count( "algorithmUsage" ) > 0 ) {
		shark::OptimizerTraits< shark::detail::AGE2 >::usage( std::cout );
		return( EXIT_SUCCESS );
	}

	if( vm.count( "objectiveFunction" ) == 0 ) {
		std::cerr << options << std::endl;
		return( EXIT_FAILURE );
	}	


	boost::optional< boost::property_tree::ptree > configurationTree;

	if( vm.count( "algorithmConfigFile" ) ) {
		try {
			boost::property_tree::ptree pt;
			boost::property_tree::read_json( vm[ "algorithmConfigFile"].as<std::string>(), pt );
			configurationTree = pt;
		} catch( ... ) {
			std::cerr << "Problem reading algorithm configuration file: " << vm[ "algorithmConfigFile"].as<std::string>() << std::endl;
			return( EXIT_FAILURE );
		}

	}

	std::string objectiveFunction = vm[ "objectiveFunction" ].as<std::string>();

	if( objectiveFunction == "DTLZ1" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ1 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	if( objectiveFunction == "DTLZ2" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ2 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	if( objectiveFunction == "DTLZ3" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ3 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	if( objectiveFunction == "DTLZ4" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ4 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	if( objectiveFunction == "DTLZ5" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ5 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	if( objectiveFunction == "DTLZ6" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ6 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	if( objectiveFunction == "DTLZ7" ) {
		shark::moo::InterruptibleAlgorithmRunner< shark::detail::AGE2, shark::DTLZ7 > runner;
		runner.run( 
			vm[ "seed" ].as< unsigned int >(),
			vm[ "storageInterval" ].as< unsigned int >(),
			vm[ "searchSpaceDimension" ].as< unsigned int >(),
			vm[ "objectiveSpaceDimension" ].as< unsigned int >(),
			vm[ "maxNoEvaluations" ].as< unsigned int >(),
			vm[ "timeLimit" ].as< unsigned int >(),
			vm[ "resultDir" ].as< std::string >(),
			configurationTree
			);
	}

	return( EXIT_SUCCESS );
}
