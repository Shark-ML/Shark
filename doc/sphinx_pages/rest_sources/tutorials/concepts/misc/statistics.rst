Iterative Calculation of Statistics
===================================

The Shark machine learning library includes a component for
iteratively calculating simple descriptive statistics of a
sequence of points for experimental evaluation. The class :doxy:`ResultTable`
includes a simple data aggregation tool that for a set of experiments
with different parameters  aggregates results over a set of trials. It 
supports missing values to reflect failed trials as well.
The class :doxy:`Statistics` takes these results to cpmpute a set of statistics.
The class offers the possibility to label the dimensions of the points and statistics
to automatically generate human readable output, for example in a csv table.

For this simple application, we are going to generate some points from
a gaussian distribution and then mark some points as missing.
For this experiment, we need the following header files: ::


	#include <shark/Statistics/Statistics.h>
	#include <shark/Rng/GlobalRng.h>
	

We start out by creating an object of class :doxy:`ResultTable`.
We give the table a name and also label the inputs as to generate 
a more readable output later on::


		statistics::ResultTable<double> table(2,"VarianceOfGaussian");//set a name for the results
		table.setDimensionName(0,"input1");
		table.setDimensionName(1,"input2");
	



Now we feed numbers into this object. For demonstration purposes we
sample 10,000 i.i.d. standard normally distributed values with varying
variance. To simulate a failed experiment, we make a coin toss for variable two
and mark this input as missing. Finally, we insert the values into the table::


		// Fill the table with randomly generated numbers for different variances and mean and also add missing values
		for(std::size_t k = 1; k != 10; ++k){
			double var= 10.0*k;
			for (std::size_t i = 0; i < 10000; i++){
				double value1=Rng::gauss(0,var);
				double value2=Rng::gauss(0,var);
				if(Rng::coinToss() == 1)
					value2=statistics::missingValue();
				table.update(var,value1,value2 );
			}
		}
	

Next, we generate a :doxy:`Statistics` object and add the statistics, here
we use Mean, Variance and Percentage of Missing values::


		statistics::Statistics<double> stats(&table);
		stats.addStatistic(statistics::Mean());//adds a statistic "Mean" for each variable
		stats.addStatistic("Variance", statistics::Variance());//explicit name
		stats.addStatistic("Missing", statistics::FractionMissing());
	

We can output a summary as csv file to the console: ::


		printCSV(stats);
	

The results looks similar to::

	# VarianceOfGausian Mean-input1 Mean-input2 Variance-input1 Variance-input2 Missing-input1 Missing-input2
	10 0.00500042 -0.073452 9.77016 10.1016 0 0.5061
	20 0.0359687 -0.0400334 20.1318 20.2767 0 0.5038
	30 0.0216264 -0.120275 30.3096 29.0293 0 0.5044
	40 -0.0301033 0.0995221 40.3523 40.4839 0 0.4961
	50 0.00692523 0.118349 48.9781 50.5156 0 0.4936
	60 -0.0133728 -0.0109795 57.4287 59.8386 0 0.4903
	70 -0.190326 0.0259554 67.0553 70.0034 0 0.4987
	80 -0.0198076 -0.0493343 83.1629 78.0985 0 0.4917
	90 -0.103546 -0.263991 92.152 89.3462 0 0.4992
