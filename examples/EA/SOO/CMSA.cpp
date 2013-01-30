#include <shark/Algorithms/DirectSearch/Experiments/Experiment.h>

#include <shark/Algorithms/DirectSearch/InterruptibleAlgorithmRunner.h>
#include <shark/Algorithms/DirectSearch/CMSA.h>

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/units/systems/si.hpp>

#include <limits>

int main( int argc, char ** argv ) {
	return( 
		shark::soo::InterruptibleAlgorithmRunner< 
			shark::CMSA, 
			shark::AbstractObjectiveFunction< shark::VectorSpace< double >, double > 
		>::main( argc, argv )
	);
}