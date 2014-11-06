#define BOOST_TEST_MODULE ML_SparseData
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/SparseData.h>
#include <shark/Data/Csv.h> //for safety export test

#include <iostream>
#include <sstream>

using namespace shark;

std::size_t const VectorSize = 11;
std::size_t const NumLines = 5;
double test_classification_values[NumLines][VectorSize] ={
	{0.3,0,0,-4,0,0,0,1.1,0,0,0},
	{0,1.2,8.82,0,0,0,1.e-4,0,0,0,0},
	{0,0,0,0,0,0,0,0,0.124,0,0},
	{0,0,0,0,0,0,0,0,0,0,0},
	{0,4.6,0,1000,0,0,0,-0.7,0,0,0.1}
};

const char test_classification[] =
"-1 1:0.3 4:-4 8:1.1 \n\
+1 2:1.2 3:8.82 7:1e-4\r\n\
1 1:0.0   9:0.124\r\n\
-1\n\
 1 2:4.6 4:1000 8:-0.7 11:0.1\n";

const char test_mc_classification[] =
"4 1:0.3 4:-4 8:1.1 \n\
3 2:1.2 3:8.82 7:1e-4\n\
2 1:0.0   9:0.124\n\
1\n\
3 2:4.6 4:1000 8:-0.7 11:0.1\n";
const char test_mc_classification_missing_label[] = 
"4 1:0.3 4:-4 8:1.1 \n\
3 2:1.2 3:8.82 7:1e-4\n\
2 1:0.0   9:0.124\n\
1\n\
3 2:4.6 4:1000 8:-0.7 11:0.1\n";
const char test_regression[] = 
"7.1 1:0.3 4:-4 8:1.1\n\
9.99 2:1.2 3:8.82 7:1e-4\r\
-5 1:0.0 9:0.124\n\
1\n\
5e2 2:4.6 4:1000 8:-0.7 11:0.1\r\n";
const char test_mc_classification_labelmap[] = 
"8 1:0.3 4:-4 8:1.1 \n\
4 2:1.2 3:8.82 7:1e-4\n\
2 1:0.0   9:0.124\n\
19\n\
4 2:4.6 4:1000 8:-0.7 11:0.1\n";
const char test_export[] = "0.3 0.0 0.0 -4 0.0 0.0 0.0 1.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0\n\
0.0 1.2 8.82 0.0 0.0 0.0 1e-4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.12437 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0\n\
0.0 4.6 0.0 1000 0.0 0.0 0.0 -0.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 17.4 0.0 0.0 0.0 0.0 -2.24 0\n";

BOOST_AUTO_TEST_SUITE (Data_SparseData)

BOOST_AUTO_TEST_CASE( Set_SparseData )
{
	std::stringstream ssc(test_classification),  ssmcc(test_mc_classification); //dense: classif., mc-classif., regr.
	std::stringstream sssc(test_classification), sssmcc(test_mc_classification); //sparse: classif., mc-classif., regr.

	LabeledData<RealVector,unsigned int> test_ds_c;
	LabeledData<RealVector,unsigned int> test_ds_mcc;
	LabeledData<CompressedRealVector,unsigned int> test_ds_sc;
	LabeledData<CompressedRealVector,unsigned int> test_ds_smcc;
	
	importSparseData(test_ds_c, ssc); //dense classif.
	importSparseData(test_ds_mcc, ssmcc); //dense mc-classif.
	importSparseData(test_ds_sc, sssc); //sparse classif.
	importSparseData(test_ds_smcc, sssmcc); //sparse mc-classif.
	
	//check that we got the proper number of lines
	BOOST_REQUIRE_EQUAL(test_ds_c.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_mcc.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_sc.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_smcc.numberOfElements(), NumLines);

	// check abels of binary/multiclass readin
	BOOST_CHECK_EQUAL(0u, test_ds_c.element(0).label);
	BOOST_CHECK_EQUAL(1u, test_ds_c.element(1).label);
	BOOST_CHECK_EQUAL(1u, test_ds_c.element(2).label);
	BOOST_CHECK_EQUAL(0u, test_ds_c.element(3).label);
	BOOST_CHECK_EQUAL(1u, test_ds_c.element(4).label);
	
	BOOST_CHECK_EQUAL(3u, test_ds_mcc.element(0).label);
	BOOST_CHECK_EQUAL(2u, test_ds_mcc.element(1).label);
	BOOST_CHECK_EQUAL(1u, test_ds_mcc.element(2).label);
	BOOST_CHECK_EQUAL(0u, test_ds_mcc.element(3).label);
	BOOST_CHECK_EQUAL(2u, test_ds_mcc.element(4).label);
	

	for ( unsigned int i=0; i<NumLines; i++ ) {
		//check proper sizes of inputs of all dataset
		BOOST_REQUIRE_EQUAL(test_ds_c.element(i).input.size(),VectorSize);
		BOOST_REQUIRE_EQUAL(test_ds_sc.element(i).input.size(),VectorSize);
		BOOST_REQUIRE_EQUAL(test_ds_smcc.element(i).input.size(),VectorSize);
		BOOST_REQUIRE_EQUAL(test_ds_mcc.element(i).input.size(),VectorSize);
		//check that all elements have the right values
		for ( unsigned int j=0; j<VectorSize; j++ ){
			BOOST_CHECK_EQUAL(test_ds_c.element(i).input(j), test_classification_values[i][j]);
			BOOST_CHECK_EQUAL(test_ds_sc.element(i).input(j), test_classification_values[i][j]);
			BOOST_CHECK_EQUAL(test_ds_smcc.element(i).input(j), test_classification_values[i][j]);
			BOOST_CHECK_EQUAL(test_ds_mcc.element(i).input(j), test_classification_values[i][j]);
		}
	}

    // labels between datasets
	for ( unsigned int i=0; i<6; i++ ) {
		BOOST_CHECK_EQUAL(test_ds_c.element(i).label, test_ds_sc.element(i).label);
		BOOST_CHECK_EQUAL(test_ds_mcc.element(i).label, test_ds_smcc.element(i).label);
		//~ BOOST_CHECK_EQUAL(test_ds_c.element(i).label, test_ds_e.element(i).label); //pre-check before export-check
		//~ BOOST_CHECK_EQUAL(test_ds_c.element(i).label, import_of_export_1.element(i).label); //export-check
		//~ BOOST_CHECK_EQUAL(test_ds_c.element(i).label, import_of_export_2.element(i).label); //export-check
	}
    
    
        //~ // test export functionality
    //~ detail::importCSV( xe, ye, sse, LAST_COLUMN, " ", "#" );
    //~ LabeledData<RealVector, unsigned int> test_ds_e = createLabeledDataFromRange( xe, ye );
    //~ BOOST_CHECK_EQUAL(test_ds_e.numberOfElements(), 6);
    //~ exportSparseData( test_ds_e, "test_output/check1.libsvm" );
    //~ exportSparseData( test_ds_c, "test_output/check2.libsvm" );
    //~ LabeledData<RealVector, unsigned int> import_of_export_1, import_of_export_2;
    //~ importSparseData( import_of_export_1, "test_output/check1.libsvm" );
    //~ importSparseData( import_of_export_2, "test_output/check2.libsvm" );

}

BOOST_AUTO_TEST_SUITE_END()
