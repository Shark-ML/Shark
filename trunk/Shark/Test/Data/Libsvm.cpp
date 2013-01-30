
#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE ML_Libsvm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/Libsvm.h>
#include <shark/Data/Csv.h> //for safety export test

#include <iostream>
#include <sstream>

using namespace shark;

const char test_classification[] = "-1 1:0.3 4:-4 8:1.1 \n\
+1 2:1.2 3:8.82 7:1e-4\r\n\
1 1:0.0   9:0.12437\r\
-1\n\
 1 2:4.6 4:1000 8:-0.7 11:0\n\
   -1 14:17.4 19:-2.24\n";
const char test_mc_classification[] = "4 1:0.3 4:-4 8:1.1 \n\
3 2:1.2 3:8.82 7:1e-4\n\
2 1:0.0   9:0.12437\n\
1\n\
3 2:4.6 4:1000 8:-0.7 11:0\n\
4 14:17.4 19:-2.24\r\n";
const char test_mc_classification_missing_label[] = "4 1:0.3 4:-4 8:1.1 \n\
3 2:1.2 3:8.82 7:1e-4\n\
2 1:0.0   9:0.12437\n\
1\n\
3 2:4.6 4:1000 8:-0.7 11:0\n\
19 14:17.4 19:-2.24\r\n";
const char test_regression[] = "7.1 1:0.3 4:-4 8:1.1\n\
9.99 2:1.2 3:8.82 7:1e-4\r\
-5 1:0.0 9:0.12437\n\
10000.7\n\
5e2 2:4.6 4:1000 8:-0.7 11:0\r\n\
4.002 14:17.4 19:-2.24\r";
const char test_mc_classification_labelmap[] = "8 1:0.3 4:-4 8:1.1 \n\
4 2:1.2 3:8.82 7:1e-4\n\
2 1:0.0   9:0.12437\n\
19\n\
4 2:4.6 4:1000 8:-0.7 11:0\n\
8 14:17.4 19:-2.24\r\n";
const char test_export[] = "0.3 0.0 0.0 -4 0.0 0.0 0.0 1.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0\n\
0.0 1.2 8.82 0.0 0.0 0.0 1e-4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.12437 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0\n\
0.0 4.6 0.0 1000 0.0 0.0 0.0 -0.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 17.4 0.0 0.0 0.0 0.0 -2.24 0\n";

BOOST_AUTO_TEST_CASE( Set_Libsvm )
{
	std::stringstream ssc(test_classification),  ssmcc(test_mc_classification),  ssr(test_regression); //dense: classif., mc-classif., regr.
	std::stringstream sssc(test_classification), sssmcc(test_mc_classification), sssr(test_regression); //sparse: classif., mc-classif., regr.
	std::stringstream sscm(test_mc_classification_labelmap);       //with labelmap
	std::stringstream sssmcch(test_mc_classification);             //for highestIndex
	std::stringstream sscml(test_mc_classification_missing_label); //for missing label
	std::stringstream sse(test_export); //for export test

	std::vector<RealVector> xc, xmcc, xr, xmccm, xcml, xe;             //last 2 for labelmap and missing label
	std::vector<CompressedRealVector> xsc, xsmcc, xsr, xsmcch;     //last for highestindex
	std::vector<unsigned int> yc, ymcc, ysc, ysmcc, ymccm, ysmcch, ycml, ye; //last 3 for labelmap, highestindex, missing label
	std::vector<double> yr, ysr;

	detail::import_libsvm(xc, yc, ssc); //dense classif.
	LabeledData<RealVector, unsigned int> test_ds_c(xc, yc);
	BOOST_REQUIRE_EQUAL(test_ds_c.numberOfElements(), 6u);
	detail::import_libsvm(xmcc, ymcc, ssmcc); //dense mc-classif.
	LabeledData<RealVector, unsigned int> test_ds_mcc(xmcc, ymcc);
	BOOST_REQUIRE_EQUAL(test_ds_mcc.numberOfElements(), 6u);
	detail::import_libsvm(xr, yr, ssr); //dense regression
	LabeledData<RealVector, double> test_ds_r(xr, yr);
	BOOST_REQUIRE_EQUAL(test_ds_r.numberOfElements(), 6u);
	detail::import_libsvm(xsc, ysc, sssc); //sparse classif.
	LabeledData<CompressedRealVector, unsigned int> test_ds_sc(xsc, ysc);
	BOOST_REQUIRE_EQUAL(test_ds_sc.numberOfElements(), 6u);
	detail::import_libsvm(xsmcc, ysmcc, sssmcc); //sparse mc-classif.
	LabeledData<CompressedRealVector, unsigned int> test_ds_smcc(xsmcc, ysmcc);
	BOOST_REQUIRE_EQUAL(test_ds_smcc.numberOfElements(), 6u);
	detail::import_libsvm(xsr, ysr, sssr); //sparse regression
	LabeledData<CompressedRealVector, double> test_ds_sr(xsr, ysr);
	BOOST_REQUIRE_EQUAL(test_ds_sr.numberOfElements(), 6u);
	std::map<unsigned int, unsigned int> labelmap;
	labelmap[19] = 0;
	labelmap[2] = 1;
	labelmap[4] = 2;
	labelmap[8] = 3;
	detail::import_libsvm(xmccm, ymccm, sscm, 0, false, &labelmap); //with label map
	LabeledData<RealVector, unsigned int> test_ds_mccm(xmccm, ymccm);
	BOOST_REQUIRE_EQUAL(ymccm.size(), 6u);

//	//this should fail:
//	detail::import_libsvm(xsmcch, ysmcch, sssmcch, 2); //with highestIndex
	detail::import_libsvm(xsmcch, ysmcch, sssmcch, 5000000); //with highestIndex
	LabeledData<CompressedRealVector, unsigned int> test_ds_smcch(xsmcch, ysmcch);
	BOOST_REQUIRE_EQUAL(test_ds_smcch.numberOfElements(), 6u);

	detail::import_libsvm(xcml, ycml, sscml, 0, true); //with missing label
	LabeledData<RealVector, unsigned int> test_ds_cml(xcml, ycml);
	BOOST_REQUIRE_EQUAL(test_ds_smcch.numberOfElements(), 6);

    // test export functionality
    detail::import_csv( xe, ye, sse, LAST_COLUMN, " ", "#" );
    LabeledData<RealVector, unsigned int> test_ds_e( xe, ye );
    BOOST_REQUIRE_EQUAL(test_ds_e.numberOfElements(), 6);
    export_libsvm( test_ds_e, "test_output/check1.libsvm" );
    export_libsvm( test_ds_c, "test_output/check2.libsvm" );
    LabeledData<RealVector, unsigned int> import_of_export_1, import_of_export_2;
    import_libsvm( import_of_export_1, "test_output/check1.libsvm" );
    import_libsvm( import_of_export_2, "test_output/check2.libsvm" );


    // a few reference datasets to raw values
	BOOST_REQUIRE_EQUAL(0u, test_ds_c(0).label);
	BOOST_REQUIRE_EQUAL(1u, test_ds_c(1).label);
	BOOST_REQUIRE_EQUAL(1u, test_ds_c(2).label);
	BOOST_REQUIRE_EQUAL(0u, test_ds_c(3).label);
	BOOST_REQUIRE_EQUAL(1u, test_ds_c(4).label);
	BOOST_REQUIRE_EQUAL(0u, test_ds_c(5).label);
	BOOST_REQUIRE_EQUAL(3u, test_ds_mcc(0).label);
	BOOST_REQUIRE_EQUAL(2u, test_ds_mcc(1).label);
	BOOST_REQUIRE_EQUAL(1u, test_ds_mcc(2).label);
	BOOST_REQUIRE_EQUAL(0u, test_ds_mcc(3).label);
	BOOST_REQUIRE_EQUAL(2u, test_ds_mcc(4).label);
	BOOST_REQUIRE_EQUAL(3u, test_ds_mcc(5).label);
	BOOST_REQUIRE_EQUAL(3u, test_ds_cml(0).label);
	BOOST_REQUIRE_EQUAL(2u, test_ds_cml(1).label);
	BOOST_REQUIRE_EQUAL(1u, test_ds_cml(2).label);
	BOOST_REQUIRE_EQUAL(0u, test_ds_cml(3).label);
	BOOST_REQUIRE_EQUAL(2u, test_ds_cml(4).label);
	BOOST_REQUIRE_EQUAL(18u, test_ds_cml(5).label);
	BOOST_REQUIRE_EQUAL(7.1, test_ds_r(0).label);
	BOOST_REQUIRE_EQUAL(9.99, test_ds_r(1).label);
	BOOST_REQUIRE_EQUAL(-5, test_ds_r(2).label);
	BOOST_REQUIRE_EQUAL(10000.7, test_ds_r(3).label);
	BOOST_REQUIRE_EQUAL(500, test_ds_r(4).label);
	BOOST_REQUIRE_EQUAL(4.002, test_ds_r(5).label);
	BOOST_REQUIRE_EQUAL(-4, test_ds_mcc(0).input(3));
	BOOST_REQUIRE_EQUAL(0, test_ds_c(0).input(4));
	BOOST_REQUIRE_EQUAL(-2.24, test_ds_c(5).input(18));


    // inputs between datasets
	for ( unsigned int i=0; i<6; i++ ) {
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_c(i).input(j));
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_mcc(i).input(j));
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_sr(i).input(j));
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_sc(i).input(j));
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_smcc(i).input(j));
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_mccm(i).input(j)); //with labelmap
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_cml(i).input(j)); //with missing labels
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ )
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_smcch(i).input(j)); //with highestIndex
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ ) {
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), test_ds_e(i).input(j)); //pre-check before export-check
        }
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ ) {
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), import_of_export_1(i).input(j)); //export-check
        }
		for ( unsigned int j=0; j<test_ds_r(i).input.size(); j++ ) {
			BOOST_REQUIRE_EQUAL(test_ds_r(i).input(j), import_of_export_2(i).input(j)); //export-check
        }
	}

    // labels between datasets
	for ( unsigned int i=0; i<6; i++ ) {
		BOOST_REQUIRE_EQUAL(test_ds_c(i).label, test_ds_sc(i).label);
		BOOST_REQUIRE_EQUAL(test_ds_mcc(i).label, test_ds_smcc(i).label);
		BOOST_REQUIRE_EQUAL(test_ds_mcc(i).label, test_ds_mccm(i).label);
		BOOST_REQUIRE_EQUAL(test_ds_mcc(i).label, test_ds_smcch(i).label);
		BOOST_REQUIRE_EQUAL(test_ds_r(i).label, test_ds_sr(i).label);
        BOOST_REQUIRE_EQUAL(test_ds_c(i).label, test_ds_e(i).label); //pre-check before export-check
		BOOST_REQUIRE_EQUAL(test_ds_c(i).label, import_of_export_1(i).label); //export-check
		BOOST_REQUIRE_EQUAL(test_ds_c(i).label, import_of_export_2(i).label); //export-check
    }

}
