#define BOOST_TEST_MODULE ML_SparseData
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/SparseData.h>

#include <iostream>
#include <sstream>

using namespace shark;

std::size_t const VectorSize = 11;
std::size_t const NumLines = 5;
double input_values[NumLines][VectorSize] ={
	{0.3,0,0,-4,0,0,0,1.1,0,0,0},
	{0,1.2,8.82,0,0,0,1.e-4,0,0,0,0},
	{0,0,0,0,0,0,0,0,0.124,0,0},
	{0,0,0,0,0,0,0,0,0,0,0},
	{0,4.6,0,1000,0,0,0,-0.7,0,0,0.1}
};

const char test_binary_classification[] =
"-1 1:0.3 4:-4 8:1.1 \n"
"+1 2:1.2 3:8.82 7:1e-4\r\n"
"1 1:0.0   9:0.124\r\n"
"-1\n"
" 1 2:4.6 4:1000 8:-0.7 11:0.1\n";

const char test_mc_classification[] =
"4 1:0.3 4:-4 8:1.1 \n"
"3 2:1.2 3:8.82 7:1e-4\n"
"2 1:0.0   9:0.124\n"
"1\n"
"3 2:4.6 4:1000 8:-0.7 11:0.1\n";

//const char test_mc_classification_missing_label[] = 
//"4 1:0.3 4:-4 8:1.1 \n"
//"3 2:1.2 3:8.82 7:1e-4\n"
//"2 1:0.0   9:0.124\n"
//"1\n"
//"3 2:4.6 4:1000 8:-0.7 11:0.1\n";

const char test_regression[] = 
"7.1 1:0.3 4:-4 8:1.1\n"
"9.99 2:1.2 3:8.82 7:1e-4\r\n"
"-5 1:0.0 9:0.124\n"
"1\n"
"5e2 2:4.6 4:1000 8:-0.7 11:0.1\r\n";

//const char test_mc_classification_labelmap[] = 
//"8 1:0.3 4:-4 8:1.1 \n"
//"4 2:1.2 3:8.82 7:1e-4\n"
//"2 1:0.0   9:0.124\n"
//"19\n"
//"4 2:4.6 4:1000 8:-0.7 11:0.1\n";

//const char test_export[] = "0.3 0.0 0.0 -4 0.0 0.0 0.0 1.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0\n"
//"0.0 1.2 8.82 0.0 0.0 0.0 1e-4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n"
//"0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.12437 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n"
//"0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0\n"
//"0.0 4.6 0.0 1000 0.0 0.0 0.0 -0.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n"
//"0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 17.4 0.0 0.0 0.0 0.0 -2.24 0\n";

template <typename InputType>
void TestExportImport_classification(LabeledData<InputType, unsigned int> const& dataset)
{
	std::stringstream ss;
	exportSparseData(dataset, ss);

	LabeledData<InputType, unsigned int> dataset2;
	importSparseData(dataset2, ss, inputDimension(dataset));

	BOOST_CHECK_EQUAL(dataset.numberOfElements(), dataset2.numberOfElements());
	for (std::size_t i=0; i<dataset.numberOfElements(); i++)
	{
		InputType x1 = elements(dataset)[i].input;
		InputType x2 = elements(dataset2)[i].input;
		BOOST_CHECK_EQUAL(x1.size(), x2.size());
		for (std::size_t j=0; j<x1.size(); j++)
		{
			BOOST_CHECK_EQUAL(x1(j), x2(j));
		}
		BOOST_CHECK_EQUAL(elements(dataset)[i].label, elements(dataset2)[i].label);
	}
}

template <typename InputType>
void TestExportImport_regression(LabeledData<InputType, RealVector> const& dataset)
{
	std::stringstream ss;
	exportSparseData(dataset, ss);

	LabeledData<InputType, RealVector> dataset2;
	importSparseData(dataset2, ss, inputDimension(dataset));

	BOOST_CHECK_EQUAL(dataset.numberOfElements(), dataset2.numberOfElements());
	for (std::size_t i=0; i<dataset.numberOfElements(); i++)
	{
		InputType x1 = elements(dataset)[i].input;
		InputType x2 = elements(dataset2)[i].input;
		BOOST_CHECK_EQUAL(x1.size(), x2.size());
		for (std::size_t j=0; j<x1.size(); j++)
		{
			BOOST_CHECK_EQUAL(x1(j), x2(j));
		}
		BOOST_CHECK_EQUAL(elements(dataset)[i].label.size(), 1);
		BOOST_CHECK_EQUAL(elements(dataset2)[i].label.size(), 1);
		BOOST_CHECK_EQUAL(elements(dataset)[i].label(0), elements(dataset2)[i].label(0));
	}
}


BOOST_AUTO_TEST_SUITE (Data_SparseData)

BOOST_AUTO_TEST_CASE (Set_SparseData)
{
	std::stringstream ssbc(test_binary_classification);
	std::stringstream ssmc(test_mc_classification);
	std::stringstream ssreg(test_regression);
	std::stringstream sssbc(test_binary_classification);
	std::stringstream sssmc(test_mc_classification);
	std::stringstream sssreg(test_regression);

	LabeledData<RealVector,           unsigned int> test_ds_bc;
	LabeledData<CompressedRealVector, unsigned int> test_ds_sbc;
	LabeledData<RealVector,           unsigned int> test_ds_mc;
	LabeledData<CompressedRealVector, unsigned int> test_ds_smc;
	LabeledData<RealVector,           RealVector>   test_ds_reg;
	LabeledData<CompressedRealVector, RealVector>   test_ds_sreg;

	importSparseData(test_ds_bc, ssbc);         // dense binary classification
	importSparseData(test_ds_sbc, sssbc);       // sparse binary classification
	importSparseData(test_ds_mc, ssmc);         // dense multi-class classification
	importSparseData(test_ds_smc, sssmc);       // sparse multi-class classification
	importSparseData(test_ds_reg, ssreg);       // dense regression
	importSparseData(test_ds_sreg, sssreg);     // sparse regression
	
	// check that we got the proper number of lines
	BOOST_REQUIRE_EQUAL(test_ds_bc.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_sbc.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_mc.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_smc.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_reg.numberOfElements(), NumLines);
	BOOST_REQUIRE_EQUAL(test_ds_sreg.numberOfElements(), NumLines);
	
	//chekc that we got the correct shape
	BOOST_CHECK_EQUAL(test_ds_bc.inputShape(), Shape({VectorSize}));
	BOOST_CHECK_EQUAL(test_ds_sbc.inputShape(), Shape({VectorSize}));
	BOOST_CHECK_EQUAL(test_ds_mc.inputShape(), Shape({VectorSize}));
	BOOST_CHECK_EQUAL(test_ds_smc.inputShape(), Shape({VectorSize}));
	BOOST_CHECK_EQUAL(test_ds_reg.inputShape(), Shape({VectorSize}));
	BOOST_CHECK_EQUAL(test_ds_reg.labelShape(), Shape({1}));
	BOOST_CHECK_EQUAL(test_ds_sreg.inputShape(), Shape({VectorSize}));
	BOOST_CHECK_EQUAL(test_ds_sreg.labelShape(), Shape({1}));

	// check labels of read-in
	BOOST_CHECK_EQUAL(0u, elements(test_ds_bc)[0].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_bc)[1].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_bc)[2].label);
	BOOST_CHECK_EQUAL(0u, elements(test_ds_bc)[3].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_bc)[4].label);
	BOOST_CHECK_EQUAL(0u, elements(test_ds_sbc)[0].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_sbc)[1].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_sbc)[2].label);
	BOOST_CHECK_EQUAL(0u, elements(test_ds_sbc)[3].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_sbc)[4].label);

	BOOST_CHECK_EQUAL(3u, elements(test_ds_mc)[0].label);
	BOOST_CHECK_EQUAL(2u, elements(test_ds_mc)[1].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_mc)[2].label);
	BOOST_CHECK_EQUAL(0u, elements(test_ds_mc)[3].label);
	BOOST_CHECK_EQUAL(2u, elements(test_ds_mc)[4].label);
	BOOST_CHECK_EQUAL(3u, elements(test_ds_smc)[0].label);
	BOOST_CHECK_EQUAL(2u, elements(test_ds_smc)[1].label);
	BOOST_CHECK_EQUAL(1u, elements(test_ds_smc)[2].label);
	BOOST_CHECK_EQUAL(0u, elements(test_ds_smc)[3].label);
	BOOST_CHECK_EQUAL(2u, elements(test_ds_smc)[4].label);

	BOOST_CHECK_EQUAL(1, elements(test_ds_reg)[0].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_reg)[1].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_reg)[2].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_reg)[3].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_reg)[4].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_sreg)[0].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_sreg)[1].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_sreg)[2].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_sreg)[3].label.size());
	BOOST_CHECK_EQUAL(1, elements(test_ds_sreg)[4].label.size());
	BOOST_CHECK_EQUAL(  7.1, elements(test_ds_reg)[0].label(0));
	BOOST_CHECK_EQUAL( 9.99, elements(test_ds_reg)[1].label(0));
	BOOST_CHECK_EQUAL( -5.0, elements(test_ds_reg)[2].label(0));
	BOOST_CHECK_EQUAL(  1.0, elements(test_ds_reg)[3].label(0));
	BOOST_CHECK_EQUAL(500.0, elements(test_ds_reg)[4].label(0));
	BOOST_CHECK_EQUAL(  7.1, elements(test_ds_sreg)[0].label(0));
	BOOST_CHECK_EQUAL( 9.99, elements(test_ds_sreg)[1].label(0));
	BOOST_CHECK_EQUAL( -5.0, elements(test_ds_sreg)[2].label(0));
	BOOST_CHECK_EQUAL(  1.0, elements(test_ds_sreg)[3].label(0));
	BOOST_CHECK_EQUAL(500.0, elements(test_ds_sreg)[4].label(0));

	for (std::size_t i=0; i<NumLines; i++)
	{
		// check proper sizes of inputs of all dataset
		BOOST_REQUIRE_EQUAL(elements(test_ds_bc)[i].input.size(), VectorSize);
		BOOST_REQUIRE_EQUAL(elements(test_ds_sbc)[i].input.size(), VectorSize);
		BOOST_REQUIRE_EQUAL(elements(test_ds_mc)[i].input.size(), VectorSize);
		BOOST_REQUIRE_EQUAL(elements(test_ds_smc)[i].input.size(), VectorSize);
		BOOST_REQUIRE_EQUAL(elements(test_ds_reg)[i].input.size(), VectorSize);
		BOOST_REQUIRE_EQUAL(elements(test_ds_sreg)[i].input.size(), VectorSize);

		// check that all elements have the correct values
		for (std::size_t j=0; j<VectorSize; j++)
		{
			BOOST_CHECK_EQUAL(elements(test_ds_bc)[i].input(j), input_values[i][j]);
			//~ BOOST_CHECK_EQUAL(elements(test_ds_sbc)[i].input(j), input_values[i][j]);
			BOOST_CHECK_EQUAL(elements(test_ds_mc)[i].input(j), input_values[i][j]);
			//~ BOOST_CHECK_EQUAL(elements(test_ds_smc)[i].input(j), input_values[i][j]);
			BOOST_CHECK_EQUAL(elements(test_ds_reg)[i].input(j), input_values[i][j]);
			//~ BOOST_CHECK_EQUAL(elements(test_ds_sreg)[i].input(j), input_values[i][j]);
		}
	}

    // check that labels of dense and sparse datasets agree
	for (std::size_t i=0; i<NumLines; i++)
	{
		BOOST_CHECK_EQUAL(elements(test_ds_bc)[i].label, elements(test_ds_sbc)[i].label);
		BOOST_CHECK_EQUAL(elements(test_ds_mc)[i].label, elements(test_ds_smc)[i].label);
		BOOST_CHECK_EQUAL(elements(test_ds_reg)[i].label(0), elements(test_ds_sreg)[i].label(0));
	}

	// test export + import round trip
	TestExportImport_classification(test_ds_bc);
	//~ TestExportImport_classification(test_ds_sbc);
	TestExportImport_classification(test_ds_mc);
	//~ TestExportImport_classification(test_ds_smc);
	TestExportImport_regression(test_ds_reg);
	//~ TestExportImport_regression(test_ds_sreg);
}

BOOST_AUTO_TEST_SUITE_END()
