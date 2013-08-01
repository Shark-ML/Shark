#include <shark/Algorithms/DirectSearch/InterruptibleAlgorithmRunner.h>
#include <shark/Algorithms/DirectSearch/SMS-EMOA.h>

#include <shark/ObjectiveFunctions/Benchmarks/DTLZ1.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ2.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ3.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ4.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ5.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ6.h>
#include <shark/ObjectiveFunctions/Benchmarks/DTLZ7.h>

#include <shark/ObjectiveFunctions/Benchmarks/LZ1.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ2.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ3.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ4.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ5.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ6.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ7.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ8.h>
#include <shark/ObjectiveFunctions/Benchmarks/LZ9.h>

#include <shark/ObjectiveFunctions/Benchmarks/ZDT1.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT2.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT3.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT4.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT6.h>

#include <shark/ObjectiveFunctions/Benchmarks/IHR1.h>
#include <shark/ObjectiveFunctions/Benchmarks/IHR2.h>
#include <shark/ObjectiveFunctions/Benchmarks/IHR3.h>
#include <shark/ObjectiveFunctions/Benchmarks/IHR4.h>
#include <shark/ObjectiveFunctions/Benchmarks/IHR6.h>

#include <shark/ObjectiveFunctions/Benchmarks/CIGTAB1.h>
#include <shark/ObjectiveFunctions/Benchmarks/CIGTAB2.h>

#include <shark/ObjectiveFunctions/Benchmarks/ELLI1.h>
#include <shark/ObjectiveFunctions/Benchmarks/ELLI2.h>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/units/systems/si.hpp>

#include <limits>

int main( int argc, char ** argv ) {

	return( 
		shark::moo::InterruptibleAlgorithmRunner< 
		shark::SMSEMOA, 
		shark::MultiObjectiveFunction 
		>::main( argc, argv )
	);
}