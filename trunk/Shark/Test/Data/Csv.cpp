
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE ML_Csv
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/Csv.h>
#include <shark/LinAlg/Base.h>

#include <iostream>
#include <sstream>


using namespace shark;


const char test[] = ",6,148,72,35,0,33.6,0.627,50,1,,,\n\
1,85,66,29,0,26.6,,,,0.351,31,0\r\
8,183,64,0,0,23.3,0.672,32,1\r\n\
1,89,66,23,94,28.1,0.167,21,0\n\
0,137,40,35,168,43.1,2.288,33,1\n\
5,116,74,0,0,25.6,0.201,30,0\n\
3,78,50,32,88,31.0,0.248,26,1\n\
10,115,0,0,0,35.3,0.134,29,0\n\
2,197,70,45,543,30.5,0.158,53,1\n\
8,125,96,0,0,0.0,0.232,54,1\n\
4,110,92,0,0,37.6,0.191,30,0\n\
10,168,74,0,0,38.0,0.537,34,1\n\
10,139,80,0,0,27.1,1.441,57,0\n\
1,189,60,23,846,30.1,0.398,59,1\n\
5,166,72,19,175,25.8,0.587,51,1\n\
7,100,0,0,0,30.0,0.484,32,1\n\
0,118,84,47,230,45.8,0.551,31,1\n\
7,107,74,0,0,29.6,0.254,31,1\n\
1,103,30,38,83,43.3,0.183,33,0\n\
1,115,70,30,96,34.6,0.529,32,1\n\
3,126,88,41,235,39.3,0.704,27,0\n\
8,99,84,0,0,35.4,0.388,50,0\n\
7,196,90,0,0,39.8,0.451,41,1\n\
9,119,80,35,0,29.0,0.263,29,1\n\
11,143,94,33,146,36.6,0.254,51,1\n\
10,125,70,26,115,31.1,0.205,41,1\n\
7,147,76,0,0,39.4,0.257,43,1\n\
1,97,66,15,140,23.2,0.487,22,0\n\
13,145,82,19,110,22.2,0.245,57,0\n\
5,117,92,0,0,34.1,0.337,38,0\n\
5,109,75,26,0,36.0,0.546,60,0\n\
3,158,76,36,245,31.6,0.851,28,1\n\
3,88,58,11,54,24.8,0.267,22,0\r";

const char test_missing_label[] = ",6,148,72,35,0,33.6,0.627,50,1,,,\n\
1,85,66,29,0,26.6,,,,0.351,31,0\r\
8,183,64,0,0,23.3,0.672,32,8\r\n\
1,89,66,23,94,28.1,0.167,21,0\n\
3,158,76,36,245,31.6,0.851,28,1\n\
3,88,58,11,54,24.8,0.267,22,0\r";

const char test_regression[] = ",6,148,72,35,0,33.6,0.627,50,1.1,,,\n\
1,85,66,29,0,26.6,,,,0.351,31,7.3\r\
8,183,64,0,0,23.3,0.672,32,2.2\r\n\
1,89,66,23,94,28.1,0.167,21,19.001\n\
3,158,76,36,245,31.6,0.851,28,1e-2\n\
3,88,58,11,54,24.8,0.267,22,33.3333\r";

BOOST_AUTO_TEST_CASE( Set_Csv )
{
	// DENSE
	std::stringstream ss(test);
	std::vector<RealVector> x;
	std::vector<unsigned int> y;
	detail::import_csv(x, y, ss, LAST_COLUMN, ",", "#" );
	LabeledData<RealVector, unsigned int> test_ds(x, y);
	BOOST_REQUIRE_EQUAL(test_ds.numberOfElements(), 33u);

	export_csv(test_ds, "test_output/check.csv", FIRST_COLUMN);
	LabeledData<RealVector, unsigned int> loaded;
	import_csv(loaded, "test_output/check.csv", FIRST_COLUMN);

	BOOST_REQUIRE_EQUAL(test_ds.size(), loaded.size());
	for (size_t i=0; i != test_ds.size(); ++i)
	{
		BOOST_REQUIRE_EQUAL(test_ds(i).input.size(), loaded(i).input.size());
		for (size_t j=0; j != test_ds(i).input.size(); ++j)
		{
			BOOST_CHECK_EQUAL(test_ds(i).input(j), loaded(i).input(j));
		}
		BOOST_CHECK_EQUAL(test_ds(i).label, loaded(i).label);
	}

	// SPARSE
	std::stringstream sss(test);
	std::vector<CompressedRealVector> sx;
	std::vector<unsigned int> sy;
	detail::import_csv(sx, sy, sss, LAST_COLUMN, ",", "#" );
	LabeledData<CompressedRealVector, unsigned int> test_ds_sparse(sx, sy);
	BOOST_REQUIRE_EQUAL(test_ds_sparse.numberOfElements(), 33);

	export_csv(test_ds_sparse, "test_output/check_sparse.csv", FIRST_COLUMN);
	LabeledData<CompressedRealVector, unsigned int> loaded_sparse;
	import_csv(loaded_sparse, "test_output/check_sparse.csv", FIRST_COLUMN);

	BOOST_REQUIRE_EQUAL(test_ds_sparse.numberOfElements(), loaded_sparse.numberOfElements());
	BOOST_REQUIRE_EQUAL( test_ds_sparse(test_ds_sparse.numberOfElements()-1).label, 0u );
	BOOST_REQUIRE_EQUAL( test_ds_sparse(test_ds_sparse.numberOfElements()-1).input(5), 24.8 );
	for (size_t i=0; i != test_ds_sparse.numberOfElements(); ++i)
	{
		BOOST_REQUIRE_EQUAL(test_ds_sparse(i).input.size(), loaded_sparse(i).input.size());
		for (size_t j=0; j != test_ds_sparse(i).input.size(); ++j)
		{
			BOOST_CHECK_EQUAL(test_ds_sparse(i).input(j), loaded_sparse(i).input(j));
		}
		BOOST_CHECK_EQUAL(test_ds_sparse(i).label, loaded_sparse(i).label);
	}

	// DENSE VS SPARSE
	for (size_t i=0; i != test_ds_sparse.numberOfElements(); ++i)
	{
		BOOST_REQUIRE_EQUAL(test_ds_sparse(i).input.size(), loaded(i).input.size());
		for (size_t j=0; j != test_ds_sparse(i).input.size(); ++j)
		{
			BOOST_CHECK_EQUAL(test_ds_sparse(i).input(j), loaded(i).input(j));
		}
		BOOST_CHECK_EQUAL(test_ds_sparse(i).label, loaded(i).label);
	}

}


BOOST_AUTO_TEST_CASE( Set_Csv_Missing_Label )
{
	// DENSE
	std::stringstream ss(test_missing_label);
	std::vector<RealVector> x;
	std::vector<unsigned int> y;
	detail::import_csv(x, y, ss, LAST_COLUMN, ",", "#", false, true );
	LabeledData<RealVector, unsigned int> test_ds(x, y);
	std::size_t test_ds_size = test_ds.numberOfElements();
	BOOST_REQUIRE_EQUAL(test_ds_size, 6u);
	BOOST_REQUIRE_EQUAL( test_ds(test_ds_size-1).label, 0u );
	BOOST_REQUIRE_EQUAL( test_ds(2).label, 8u );
	BOOST_REQUIRE_EQUAL( test_ds(test_ds_size-1).input(5), 24.8 );

	export_csv(test_ds, "test_output/check.csv", FIRST_COLUMN);
	LabeledData<RealVector, unsigned int> loaded;
	import_csv(loaded, "test_output/check.csv", FIRST_COLUMN, ",", "#", true);

	BOOST_REQUIRE_EQUAL(test_ds_size, loaded.numberOfElements());
	for (size_t i=0; i != test_ds_size; ++i)
	{
		BOOST_REQUIRE_EQUAL(test_ds(i).input.size(), loaded(i).input.size());
		for (size_t j=0; j != test_ds(i).input.size(); ++j)
		{
			BOOST_CHECK_EQUAL(test_ds(i).input(j), loaded(i).input(j));
		}
		BOOST_CHECK_EQUAL(test_ds(i).label, loaded(i).label);
	}

}

BOOST_AUTO_TEST_CASE( Set_Csv_Regression )
{
	// DENSE
	std::stringstream ss(test_regression);
	std::vector<RealVector> x;
	std::vector<double> y;
	detail::import_csv(x, y, ss, LAST_COLUMN, ",", "#");
	LabeledData<RealVector, double> test_ds(x, y);
	std::size_t test_ds_size = test_ds.numberOfElements();
	BOOST_REQUIRE_EQUAL(test_ds_size, 6u);
	BOOST_REQUIRE_EQUAL( test_ds(test_ds_size-1).label, 33.3333 );
	BOOST_REQUIRE_EQUAL( test_ds(test_ds_size-1).input(5), 24.8 );

	export_csv(test_ds, "test_output/check.csv", FIRST_COLUMN);
	LabeledData<RealVector, double> loaded;
	import_csv(loaded, "test_output/check.csv", FIRST_COLUMN);

	BOOST_REQUIRE_EQUAL(test_ds_size, loaded.numberOfElements());
	for (size_t i=0; i != test_ds_size; ++i)
	{
		BOOST_REQUIRE_EQUAL(test_ds(i).input.size(), loaded(i).input.size());
		for (size_t j=0; j != test_ds(i).input.size(); ++j)
		{
			BOOST_CHECK_EQUAL(test_ds(i).input(j), loaded(i).input(j));
		}
		BOOST_CHECK_EQUAL(test_ds(i).label, loaded(i).label);
	}

}

