Iterative Calculation of Statistics
===================================

The Shark machine learning library includes a component for
iteratively calculating simple descriptive statistics of a
sequence of values. The class :doxy:`Statistics` is a thin
wrapper around the boost::accumulators component. This tutorial
illustrates its usage.

We need the following header files: ::


	#include <shark/Statistics/Statistics.h>
	#include <shark/Rng/GlobalRng.h>
	

We start out by creating an object of class :doxy:`Statistics`: ::


		Statistics stats;
	

Now we feed numbers into this object. For demonstration purposes we
sample 100,000 i.i.d. standard normally distributed values from the
random number generator. ::


		// Sample 10000 standard normally distributed random numbers
		// and update statistics for these numbers iteratively.
		for (std::size_t i = 0; i < 100000; i++)
			stats( Rng::gauss() );
	

We can output a summary to the console: ::


		// Output results to the console.
		std::cout << stats << std::endl;
	

The results looks similar to: ::

	Sample size: 100000
	Min: -4.09568
	Max: 4.42802
	Mean: -0.000584566
	Variance: 0.992282
	Median: 0.00121767
	Lower Quantile: -0.673034
	Upper Quantile: 0.670621

Alternatively it is possible to access the individual values. The
round bracket operator is used for this purpose: ::


		std::cout << 
			stats( Statistics::NumSamples() ) << " " <<
			stats( Statistics::Min() ) << " " <<
			stats( Statistics::Max() ) << " " <<
			stats( Statistics::Mean() ) << " " << 
			stats( Statistics::Variance() ) << " " <<
			stats( Statistics::Median() ) << " " <<
			stats( Statistics::LowerQuartile() ) << " " <<
			stats( Statistics::UpperQuartile() ) << std::endl;
	
