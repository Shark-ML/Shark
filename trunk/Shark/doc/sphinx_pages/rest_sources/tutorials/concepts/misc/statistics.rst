Iterative Calculation of Statistics
===================================

The Shark machine learning library includes a component for
iteratively calculating statistical properties of a sequence of
values. The class :doxy:`Statistics` is a thin wrapper around the
boost::accumulators component and this tutorial illustrates its usage.

First of all, the following header files need to be included: ::

  #include <shark/Statistics/Statistics.h>
  #include <shark/Rng/GlobalRng.h>

Next, we need to instantiate an object of class :doxy:`Statistics` according to: ::

  shark::Statistics stats;

Finally, we feed in standard normally distributed values to the
component and output the results to the console with the following
lines of code: ::

    // Sample 10000 standard normally distributed random numbers
    // and update statistics for these numbers.
    for( std::size_t i = 0; i < 100000; i++ )
	stats( shark::Rng::gauss() );

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


