
#include <iostream>
#define BOOST_TEST_MODULE ML_LinearProgram
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/LP/GLPK.h>
#include <cmath>


using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_LP_LinearProgram)

BOOST_AUTO_TEST_CASE( ML_LinearProgram )
{
	// define the example linear program
	// found in the GLPK documentation
	LP lp;
	lp.setMaximize();
	lp.addRows(3);
	lp.addColumns(3);
	lp.setObjectiveCoefficient(1, 10.0);
	lp.setObjectiveCoefficient(2, 6.0);
	lp.setObjectiveCoefficient(3, 4.0);
	lp.setRowUpperBounded(1, 100.0);
	lp.setRowUpperBounded(2, 600.0);
	lp.setRowUpperBounded(3, 300.0);
	lp.setColumnLowerBounded(1, 0.0);
	lp.setColumnLowerBounded(2, 0.0);
	lp.setColumnLowerBounded(3, 0.0);
	std::vector<unsigned int> row(10);
	std::vector<unsigned int> col(10);
	std::vector<double> value(10);
	row[1] = 1; col[1] = 1; value[1] = 1.0;
	row[2] = 1; col[2] = 2; value[2] = 1.0;
	row[3] = 1; col[3] = 3; value[3] = 1.0;
	row[4] = 2; col[4] = 1; value[4] = 10.0;
	row[5] = 3; col[5] = 1; value[5] = 2.0;
	row[6] = 2; col[6] = 2; value[6] = 3.0;
	row[7] = 3; col[7] = 2; value[7] = 2.0;
	row[8] = 2; col[8] = 3; value[8] = 5.0;
	row[9] = 3; col[9] = 3; value[9] = 6.0;
	lp.setConstraintMatrix(row, col, value);

	// solve the linear program with the simplex algorithm
	lp.solve();

	// read out the solution
	double p[3];
	p[0] = lp.solution(1);
	p[1] = lp.solution(2);
	p[2] = lp.solution(3);

	// compare the solution to the true optimum with an accuracy
	// of 6 significant digits, which is more than enough to
	// identify the correct corner of the simplex
	double best[3] = {42.8571, 57.1429, 0.0};
	double diff = fabs(best[0] - p[0]) + fabs(best[1] - p[1]) + fabs(best[2] - p[2]);

	BOOST_CHECK_SMALL(diff, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()
