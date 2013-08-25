#define BOOST_TEST_MODULE ML_Csv
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/Csv.h>
#include <shark/LinAlg/Base.h>

#include <boost/math/special_functions/fpclassify.hpp>

using namespace shark;

const char test_separator[] = "\
1.000,148.0,72,35,0,33.6,,?,-1\n\
1,66,29,0,26.6,,,?,-1\r\
1,183,64,0,0, 23.3,0.672,32,1\r\n\
2.000,116,\t74\t,\t0 \t\t,0,25.6,0.201 , 30,-1\n\
1.,78,50,32,88 ,31.0, 0.248,26,1\n\
2,196,90,0,0,39.8,0.451,41,1\n\
1.0,119,80,35,0,29.0,0.263,29,1\n\
2.,143,94,33,146,36.6,0.254,51,1\n\
1,125,70,26,115,31.1,0.205,41,1\n\
1,147,76,0,0,39.4,0.257,43,1\n\
1,97,66,15,140,23.2,0.487,22,-1\n\
0,145,82,19,110,22.2,0.245,57,-1\n\
0,117,92,0,0,34.1,0.337,38,-1\n\
1,109,75,26,0,36.0,0.546,60,-1\n\
1,158,76,36,245,31.6,0.851,28,1\n\
0,88,58,11,54,24.8,0.267,22,-1\r";

const char test_no_separator[] = "\
1.000 148.0 \t72 35\t0 33.6 ? ? -1\n\
1 66 29 0 26.6 ? ? ? -1\r\
1 183 64 0 0 23.3 0.672 32 1\r\n\
2.000 116 74 0\t0 25.6 0.201 30 -1\n\
1. 78 50 32 88 31.0 0.248 26 1\n\
2 196 90 0 0 39.8\t\t\t0.451 41 1\n\
1.0 119 80 35 0 29.0 0.263 29 1\n\
2. 143 94 33 146 36.6 0.254 51 1\n\
1 125 70 26 115 31.1 0.205 41 1\n\
1 147 76 0 0 39.4 0.257 43 1\n\
1 97 66 15 140 23.2 0.487 22 -1\n\
0 145 82 19 110 22.2 0.245 57 -1\n\
0 117 92 0 0 34.1 0.337 38 -1\n\
1 109 75 26 0 36.0 0.546 60 -1\n\
1 158 76 36 245 31.6 0.851 28 1\n\
0 88 58 11 54 24.8 0.267 22 -1\r";

const double qnan = std::numeric_limits<double>::quiet_NaN();

std::size_t const numDimensions = 8;
std::size_t const numInputs = 16;
double test_values_1[numInputs * numDimensions] ={
	148.0,72,35,0,33.6,qnan,qnan,-1,
	66,29,0,26.6,qnan,qnan,qnan,-1,
	183,64,0,0,23.3,0.672,32,1,
	116,74,0,0,25.6,0.201,30,-1,
	78,50,32,88,31.0,0.248,26,1,
	196,90,0,0,39.8,0.451,41,1,
	119,80,35,0,29.0,0.263,29,1,
	143,94,33,146,36.6,0.254,51,1,
	125,70,26,115,31.1,0.205,41,1,
	147,76,0,0,39.4,0.257,43,1,
	97,66,15,140,23.2,0.487,22,-1,
	145,82,19,110,22.2,0.245,57,-1,
	117,92,0,0,34.1,0.337,38,-1,
	109,75,26,0,36.0,0.546,60,-1,
	158,76,36,245,31.6,0.851,28,1,
	88,58,11,54,24.8,0.267,22,-1
};

double test_values_2[numInputs*numDimensions] ={
	1.000,148.0,72,35,0,33.6,qnan,qnan,
	1,66,29,0,26.6,qnan,qnan,qnan,
	1,183,64,0,0,23.3,0.672,32,
	2.000,116,74,0,0,25.6,0.201,30,
	1.,78,50,32,88,31.0,0.248,26,
	2,196,90,0,0,39.8,0.451,41,
	1.0,119,80,35,0,29.0,0.263,29,
	2.,143,94,33,146,36.6,0.254,51,
	1,125,70,26,115,31.1,0.205,41,
	1,147,76,0,0,39.4,0.257,43,
	1,97,66,15,140,23.2,0.487,22,
	0,145,82,19,110,22.2,0.245,57,
	0,117,92,0,0,34.1,0.337,38,
	1,109,75,26,0,36.0,0.546,60,
	1,158,76,36,245,31.6,0.851,28,
	0,88,58,11,54,24.8,0.267,22
};

unsigned int labels_1[numInputs] = {1,1,1,2,1,2,1,2,1,1,1,0,0,1,1,0};
unsigned int labels_2[numInputs] = {0,0,1,0,1,1,1,1,1,1,0,0,0,0,1,0};

template<class T, class U, class V>
void checkDataEquality(T* values, unsigned int* labels, LabeledData<V,U> const& loaded){
	BOOST_REQUIRE_EQUAL(loaded.numberOfElements(),numInputs);
	BOOST_REQUIRE_EQUAL(inputDimension(loaded),numDimensions);
	for (size_t i=0; i != numInputs; ++i){
		for (size_t j=0; j != numDimensions; ++j)
		{
			if( boost::math::isnan(values[i*numDimensions+j])){
				BOOST_CHECK(boost::math::isnan(loaded.element(i).input(j)));
			}
			else
			{
				BOOST_CHECK_EQUAL(loaded.element(i).input(j), values[i*numDimensions+j]);
			}
		}
		BOOST_CHECK_EQUAL(loaded.element(i).label, labels[i]);
	}
}

BOOST_AUTO_TEST_CASE( Data_Csv_Separator_First_Column )
{
	// DENSE
	LabeledData<RealVector, unsigned int> test_ds;
	csvStringToData(test_ds, test_separator, FIRST_COLUMN, ',','#',3);
	BOOST_CHECK_EQUAL(test_ds.numberOfElements(), 16u);
	BOOST_CHECK_EQUAL(test_ds.numberOfBatches(), 6);
	BOOST_CHECK_EQUAL(numberOfClasses(test_ds), 3);
	std::cout << test_ds<<std::endl;
	
	checkDataEquality(test_values_1,labels_1,test_ds);

	//~ export_csv(test_ds, "test_output/check.csv", FIRST_COLUMN);
	//~ LabeledData<RealVector, unsigned int> loaded;
	//~ import_csv(loaded, "test_output/check.csv", FIRST_COLUMN);

	//~ checkDataEquality(test_ds,loaded);

	//~ // SPARSE
	//~ std::stringstream sss(test);
	//~ std::vector<CompressedRealVector> sx;
	//~ std::vector<unsigned int> sy;
	//~ detail::import_csv(sx, sy, sss, LAST_COLUMN, ",", "#" );
	//~ LabeledData<CompressedRealVector, unsigned int> test_ds_sparse = createLabeledDataFromRange(sx, sy);
	//~ BOOST_REQUIRE_EQUAL(test_ds_sparse.numberOfElements(), 33);

	//~ export_csv(test_ds_sparse, "test_output/check_sparse.csv", FIRST_COLUMN);
	//~ LabeledData<CompressedRealVector, unsigned int> loaded_sparse;
	//~ import_csv(loaded_sparse, "test_output/check_sparse.csv", FIRST_COLUMN);

	
	//~ BOOST_REQUIRE_EQUAL( test_ds_sparse.element(test_ds_sparse.numberOfElements()-1).label, 0u );
	//~ BOOST_REQUIRE_EQUAL( test_ds_sparse.element(test_ds_sparse.numberOfElements()-1).input(5), 24.8 );
	//~ checkDataEquality(test_ds_sparse,loaded_sparse);
	//~ checkDataEquality(test_ds_sparse,loaded);
}


BOOST_AUTO_TEST_CASE( Data_Csv_No_Separator_First_Column )
{
	// DENSE
	LabeledData<RealVector, unsigned int> test_ds;
	csvStringToData(test_ds, test_no_separator, FIRST_COLUMN, 0,'#',3);
	BOOST_CHECK_EQUAL(test_ds.numberOfElements(), 16u);
	BOOST_CHECK_EQUAL(test_ds.numberOfBatches(), 6);
	BOOST_CHECK_EQUAL(numberOfClasses(test_ds), 3);
	
	std::cout << test_ds<<std::endl;
	
	checkDataEquality(test_values_1,labels_1,test_ds);
}

BOOST_AUTO_TEST_CASE( Data_Csv_No_Separator_Last_Column )
{
	// DENSE
	LabeledData<RealVector, unsigned int> test_ds;
	csvStringToData(test_ds, test_no_separator, LAST_COLUMN, 0,'#',3);
	BOOST_CHECK_EQUAL(test_ds.numberOfElements(), 16u);
	BOOST_CHECK_EQUAL(test_ds.numberOfBatches(), 6);
	BOOST_CHECK_EQUAL(numberOfClasses(test_ds), 2);
	
	std::cout << test_ds<<std::endl;
	
	checkDataEquality(test_values_2,labels_2,test_ds);
}

BOOST_AUTO_TEST_CASE( Data_Csv_Separator_Last_Column )
{
	// DENSE
	LabeledData<RealVector, unsigned int> test_ds;
	csvStringToData(test_ds, test_separator, LAST_COLUMN, ',','#',3);
	BOOST_CHECK_EQUAL(test_ds.numberOfElements(), 16u);
	BOOST_CHECK_EQUAL(test_ds.numberOfBatches(), 6);
	BOOST_CHECK_EQUAL(numberOfClasses(test_ds), 2);
	
	std::cout << test_ds<<std::endl;
	
	checkDataEquality(test_values_2,labels_2,test_ds);
}