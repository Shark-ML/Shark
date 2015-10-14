#define BOOST_TEST_MODULE CoreHDF5TestModule

#include "shark/Data/HDF5.h"

#include <boost/assign/list_of.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

namespace shark {

/// Fixture for testing HDF5 file import
class HDF5Fixture
{
public:
	HDF5Fixture()
	:
		m_exampleFileName("./test_data/testfile_for_import.h5"),
		m_datasetNameData1("data/data1"),
		m_labelNameLabel1("data/label1"),
		m_labelNameWrongLabel("data/wrong_label"),
		m_dsNameOneDimension("data/one_dimension"),
		m_dsNameThreeDimension("data/three_dimension"),
		m_dsNameCscVal("csc/data"),
		m_dsNameIndices("csc/indices"),
		m_dsNameIndPtr("csc/indptr")
	{
		using namespace boost::assign;

		// 3 x 4 matrix
		m_expectedFromData1 +=
			list_of(1.0)(2.0)(3.0),
			list_of(4.0)(5.0)(6.0),
			list_of(7.0)(8.0)(9.0),
			list_of(10.0)(11.0)(12.0);

		// vector of 4 elements
		m_expectedFromLabel1 = list_of(80.0)(81.0)(82.0)(83.0);
	}

	/// Verify the @a actual matrix is the same as @a expected
	template<typename MatrixType, typename ExpectedType>
	boost::test_tools::predicate_result verify(
		const MatrixType& actual,
		const ExpectedType& expected
	);

	/// Validate @exp has @msg
	bool validate(const std::string& msg, const shark::Exception& exp);

	/// Name of test file
	const std::string m_exampleFileName;

	/// dataset: data/data1
	const std::string m_datasetNameData1;
	std::vector<std::vector<double> > m_expectedFromData1;

	/// dataset: data/label1
	const std::string m_labelNameLabel1;
	const std::string m_labelNameWrongLabel;
	std::vector<boost::int32_t> m_expectedFromLabel1;

	const std::string m_dsNameOneDimension;
	const std::string m_dsNameThreeDimension;

	const std::string m_dsNameCscVal;
	const std::string m_dsNameIndices;
	const std::string m_dsNameIndPtr;
};

template<typename MatrixType, typename ExpectedType>
boost::test_tools::predicate_result HDF5Fixture::verify(
	const MatrixType& actual,
	const ExpectedType& expected)
{
	boost::test_tools::predicate_result res( true );
	if (actual.numberOfElements() != expected.size()) {
		res = false;
		res.message() << boost::format("\nActual size: %1%, expected size: %2%") % actual.numberOfElements() % expected.size();
	}

	for (size_t i = 0; i < actual.numberOfElements(); ++i) {
		if (actual.element(i).size() != expected[i].size()) {
			res = false;
			res.message() <<
				boost::format("\nIndex: %1%, actual size: %2%, expected size: %3%")
					% i
					% actual.element(i).size()
					% expected[i].size();
		}

		for (size_t j = 0; j < actual.element(i).size(); ++j) {
			if (actual.element(i)(j) != expected[i][j]) { // floating point comparison here should also work in our case
				res = false;
				res.message() <<
					boost::format("\nElements not equal: actual[%1%][%2%]=%3%, expected[%1%][%2%]=%4%")
						% i
						% j
						% actual.element(i)(j)
						% expected[i][j];
			}
		}
	}

	return res;
}

bool HDF5Fixture::validate(const std::string& msg, const shark::Exception& exp)
{
	return exp.what() == msg;
}

BOOST_FIXTURE_TEST_SUITE (Data_HDF5Tests, HDF5Fixture)

BOOST_AUTO_TEST_CASE(BasicTests)
{
	// Test that importing a basic 2-dimension matrix of double type works fine
	{
		Data<RealVector> data;
		importHDF5<RealVector>(data, m_exampleFileName, m_datasetNameData1);
		BOOST_CHECK(verify(data, m_expectedFromData1));
	}

	// Test that importing a basic 2-dimension labeled matrix of double type works fine
	{
		LabeledData<RealVector, boost::int32_t> data;
		importHDF5<RealVector, boost::int32_t>(data, m_exampleFileName, m_datasetNameData1, m_labelNameLabel1);
		BOOST_CHECK(verify(data.inputs(), m_expectedFromData1));
		BOOST_CHECK_EQUAL_COLLECTIONS(
			data.labels().elements().begin(), 
			data.labels().elements().end(), 
			m_expectedFromLabel1.begin(), 
			m_expectedFromLabel1.end()
		);
	}

	// Test same thing for compressed vector

	{
		Data<CompressedRealVector> data;
		importHDF5<CompressedRealVector>(data, m_exampleFileName, m_datasetNameData1);
		BOOST_CHECK(verify(data, m_expectedFromData1));
	}

	{
		LabeledData<CompressedRealVector, boost::int32_t> data;
		importHDF5<CompressedRealVector, boost::int32_t>(data, m_exampleFileName, m_datasetNameData1, m_labelNameLabel1);
		BOOST_CHECK(verify(data.inputs(), m_expectedFromData1));
		BOOST_CHECK_EQUAL_COLLECTIONS(
			data.labels().elements().begin(), 
			data.labels().elements().end(), 
			m_expectedFromLabel1.begin(), 
			m_expectedFromLabel1.end()
		);
	}
}

BOOST_AUTO_TEST_CASE(CscTests)
{
	// Basic Test for CSC
	{
		using namespace boost::assign;
		std::vector<std::string> csc;
		csc += m_dsNameCscVal, m_dsNameIndices, m_dsNameIndPtr;
		Data<RealVector> data;
		importHDF5<RealVector>(data, m_exampleFileName, csc);

		std::vector<std::vector<double> > expected;
		expected +=
			list_of(1.0)(0.0)(2.0),
			list_of(0.0)(0.0)(3.0),
			list_of(4.0)(5.0)(6.0);

		BOOST_CHECK(verify(data, expected));
	}

	// Test (CSC + compressed vector) works
	{
		// Python:
		// >>> from scipy.sparse import *
		// >>> from scipy import *
		// >>> data=array([10,20,30,50,40,60,70,80])
		// >>> indices=array([0,0,1,2,1,2,2,3])
		// >>> idxptr=array([0,1,3,4,6,7,8])
		// >>> csc_matrix((data,indices,idxptr)).todense()
		// matrix([[10, 20,  0,  0,  0,  0],
		//		   [ 0, 30,  0, 40,  0,  0],
		//		   [ 0,  0, 50, 60, 70,  0],
		//		   [ 0,  0,  0,  0,  0, 80]])
		// >>>

		using namespace boost::assign;
		std::vector<std::string> csc;
		csc += "csc2/data", "csc2/indices", "csc2/idxptr";
		LabeledData<CompressedIntVector, boost::int32_t> data;
		importHDF5<CompressedIntVector, boost::int32_t>(data, m_exampleFileName, csc, "csc2/label");

		std::vector<std::vector<boost::int32_t> > expectedInputs;
		expectedInputs +=
			list_of(10)(0)(0)(0),
			list_of(20)(30)(0)(0),
			list_of(0)(0)(50)(0),
			list_of(0)(40)(60)(0),
			list_of(0)(0)(70)(0),
			list_of(0)(0)(0)(80);

		std::vector<boost::int32_t> expectedLabels;
		expectedLabels += 100,200,300,400,500,600;

		BOOST_CHECK(verify(data.inputs(), expectedInputs));
		BOOST_CHECK_EQUAL_COLLECTIONS(
			data.labels().elements().begin(), 
			data.labels().elements().end(), 
			expectedLabels.begin(), 
			expectedLabels.end()
		);
	}
}

BOOST_AUTO_TEST_CASE(OneDimension)
{
	// Test that accessing one-dimension dataset works fine

	using namespace boost::assign;
	std::vector<std::vector<double> > expected;
	expected += list_of(1.0)(2.0)(3.0);

	Data<CompressedRealVector> data;
	importHDF5<CompressedRealVector>(data, m_exampleFileName, m_dsNameOneDimension);
	BOOST_CHECK(verify(data, expected));
}

BOOST_AUTO_TEST_CASE(NegativeTests)
{
	// Test that trying to import for a non-exist file will throw an exception
	{
		const std::string nonExistFileName = "non-exist.h5";
		Data<RealVector> data;
		BOOST_CHECK_EXCEPTION(
			importHDF5<RealVector>(data, nonExistFileName, "dummy/dummy"),
			shark::Exception,
			boost::bind(
				&HDF5Fixture::validate,
				this,
				(boost::format("[loadIntoMatrix] open file name: %1% (FAILED)") % nonExistFileName).str(),
				_1));
	}

	// Test that accessing a non-exist dataset will throw an exception
	{
		const std::string dummyDataset = "data/dummy";
		Data<RealVector> data;
		BOOST_CHECK_EXCEPTION(
			importHDF5<RealVector>(data, m_exampleFileName, dummyDataset),
			shark::Exception,
			boost::bind(
				&HDF5Fixture::validate,
				this,
				(boost::format("[importHDF5] Get data set(%1%) info from file(%2%).") % dummyDataset % m_exampleFileName).str(),
				_1));
	}

	// Test that accessing a 3-dimension dataset will throw exception
	{
		Data<RealVector> data;
		BOOST_CHECK_EXCEPTION(
			importHDF5<RealVector>(data, m_exampleFileName, m_dsNameThreeDimension),
			shark::Exception,
			boost::bind(
				&HDF5Fixture::validate,
				this,
				(boost::format("[loadIntoMatrix][%2%][%1%] Support 1 or 2 dimensions, but this dataset has at least 3 dimensions.")
					% m_dsNameThreeDimension
					% m_exampleFileName).str(),
				_1));
	}

	// Test that accessing with unmatched data type will throw exception
	{
		Data<IntVector> data; // 'data/data1' is actual of 64-bit float
		BOOST_CHECK_EXCEPTION(
			importHDF5<IntVector>(data, m_exampleFileName, m_datasetNameData1),
			shark::Exception,
			boost::bind(
				&HDF5Fixture::validate,
				this,
				(boost::format("[loadIntoMatrix] DataType doesn't match. HDF5 data type in dataset(%1%::%2%): 1, size: 8")
					% m_exampleFileName
					% m_datasetNameData1).str(),
				_1));
	}
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace shark
