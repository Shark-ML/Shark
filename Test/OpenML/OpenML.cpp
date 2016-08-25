#define BOOST_TEST_MODULE OpenML_OpenML
#include <boost/test/unit_test.hpp>

#include <shark/OpenML/OpenML.h>
#include <shark/Data/Arff.h>

#include <boost/lexical_cast.hpp>
#include <cstdio>

using namespace shark;
using namespace openML;


static const std::string demo_api_key = "0076c004519625ecc7ad51e553f40dff";


BOOST_AUTO_TEST_SUITE (OpenML_OpenML)

BOOST_AUTO_TEST_CASE(OpenML_Dataset)
{
	// prepare connection to the test server
	connection.enableTestMode();
	connection.setKey(demo_api_key);

	// construct data set from ID
	std::shared_ptr<Dataset> ds = Dataset::get(11);
	BOOST_CHECK_EQUAL(ds->id(), 11);

	// basic getters
	BOOST_CHECK_EQUAL(ds->name(), "balance-scale");
	BOOST_CHECK_EQUAL(ds->format(), "ARFF");

	// feature getters
	BOOST_CHECK_EQUAL(ds->numberOfFeatures(), 5);
	BOOST_CHECK_EQUAL(ds->feature(0).type, NUMERIC);
	BOOST_CHECK_EQUAL(ds->feature(0).name, "left-weight");
	BOOST_CHECK_EQUAL(ds->feature(0).target, false);
	BOOST_CHECK_EQUAL(ds->feature(0).ignore, false);
	BOOST_CHECK_EQUAL(ds->feature(0).rowIdentifier, false);
	BOOST_CHECK_EQUAL(ds->feature(4).type, NOMINAL);
	BOOST_CHECK_EQUAL(ds->feature(4).name, "class");
	BOOST_CHECK_EQUAL(ds->feature(4).target, true);
	BOOST_CHECK_EQUAL(ds->feature(4).ignore, false);
	BOOST_CHECK_EQUAL(ds->feature(4).rowIdentifier, false);
	BOOST_CHECK_EQUAL(ds->featureIndex("class"), 4);

	// cached file
	CachedFile const& f = ds->datafile();
	BOOST_CHECK_EQUAL(f.filename(), CachedFile::cacheDirectory() / "dataset_11.arff");
	f.download();
	BOOST_CHECK(boost::filesystem::exists(f.filename()));

	// file content
	ClassificationDataset data;
	importARFF(f.filename().string(), "class", data);
	BOOST_CHECK_EQUAL(data.numberOfElements(), 625);
	BOOST_CHECK_EQUAL(inputDimension(data), 4);
	BOOST_CHECK_EQUAL(numberOfClasses(data), 3);

	// remove cached file
	std::remove(f.filename().string().c_str());
}

BOOST_AUTO_TEST_CASE(OpenML_Task)
{
	// prepare connection to the test server
	connection.enableTestMode();
	connection.setKey(demo_api_key);

	// construct task from ID
	std::shared_ptr<Task> task = Task::get(11);
	BOOST_CHECK_EQUAL(task->id(), 11);
// TODO: obtain the properties of task 11 ON THE TEST SERVER

	// basic getters
	BOOST_CHECK_EQUAL(task->tasktype(), SupervisedClassification);
	BOOST_CHECK_EQUAL(task->targetFeature(), "class");
	BOOST_CHECK_EQUAL(task->repetitions(), 1);
	BOOST_CHECK_EQUAL(task->folds(), 10);

	// access to data set
	std::shared_ptr<const Dataset> ds = task->dataset();
	BOOST_CHECK_EQUAL(ds->id(), 11);

	// load data
	ClassificationDataset data;
	task->loadData(data);
	BOOST_CHECK_EQUAL(data.numberOfElements(), 625);
	BOOST_CHECK_EQUAL(inputDimension(data), 4);
	BOOST_CHECK_EQUAL(numberOfClasses(data), 3);

	// obtain folds
	CVFolds<ClassificationDataset> folds = task->split(0, data);
	BOOST_CHECK_EQUAL(folds.size(), 10);

	// obtain "raw" split indices
	std::vector<std::size_t> const& idx = task->splitIndices(0);
	BOOST_CHECK_EQUAL(idx.size(), 625);

	// cached file
	CachedFile const& f = task->splitsfile();
	BOOST_CHECK_EQUAL(f.filename(), CachedFile::cacheDirectory() / "task_11_splits.arff");
	f.download();
	BOOST_CHECK(boost::filesystem::exists(f.filename()));

	// remove cached files
	std::remove(f.filename().string().c_str());
	std::remove(ds->datafile().filename().string().c_str());
}

BOOST_AUTO_TEST_CASE(OpenML_Flow)
{
	// prepare connection to the test server
	connection.enableTestMode();
	connection.setKey(demo_api_key);

	// construct flow
	std::vector<Hyperparameter> hyperparameters;
	hyperparameters.push_back(Hyperparameter("a", "first dummy parameter", "numeric"));
	hyperparameters.push_back(Hyperparameter("b", "second dummy parameter", "integer"));
	std::shared_ptr<Flow> flow = Flow::get("Shark_unit_test_flow", "This flow is for unit testing only, please ignore.", hyperparameters);

	// basic getters
	BOOST_CHECK_EQUAL(flow->name(), "Shark_unit_test_flow");
	BOOST_CHECK_EQUAL(flow->version(), Flow::sharkVersion());
	BOOST_CHECK_EQUAL(flow->description(), "This flow is for unit testing only, please ignore.");

	// hyperparameters
	BOOST_CHECK_EQUAL(flow->numberOfHyperparameters(), 2);
	BOOST_CHECK_EQUAL(flow->hyperparameter(0).name, "a");
	BOOST_CHECK_EQUAL(flow->hyperparameter(0).description, "first dummy parameter");
	BOOST_CHECK_EQUAL(flow->hyperparameter(0).datatype, "numeric");
	BOOST_CHECK_EQUAL(flow->hyperparameter(1).name, "b");
	BOOST_CHECK_EQUAL(flow->hyperparameter(1).description, "second dummy parameter");
	BOOST_CHECK_EQUAL(flow->hyperparameter(1).datatype, "integer");
	BOOST_CHECK_EQUAL(flow->hyperparameterIndex("b"), 1);
}

BOOST_AUTO_TEST_CASE(OpenML_Run)
{
	// prepare connection to the test server
	connection.enableTestMode();
	connection.setKey(demo_api_key);

	// construct task
	std::shared_ptr<Task> task = Task::get(11);

	// construct flow
	IDType flowID = getFlow("Shark_unit_test_flow", Flow::sharkVersion());
	BOOST_REQUIRE(flowID != invalidID);
	std::shared_ptr<Flow> flow = Flow::get(flowID);

	// construct run from task and flow
	Run run(task, flow);

	// set hyperparameters
	BOOST_CHECK_EQUAL(run.numberOfHyperparameters(), 2);
	run.setHyperparameterValue("a", 3.14);
	run.setHyperparameterValue("b", 42);

	// obtain data splits from the task
	ClassificationDataset data;
	CVFolds<ClassificationDataset> folds;
	task->loadData(data);
	task->split(0, data);

	// store test labels as predictions
	// (in terms of machine learning this is cheating! but it is a useful unit test)
	for (std::size_t f=0; f<10; f++)
	{
		run.setPredictions(0, f, folds.validation(f).labels());
	}

	// store the "predictions" in OpenML
	run.commit();

	// check the run ID
	IDType runID = run.id();
	BOOST_CHECK(runID != invalidID);

	// cached file
	CachedFile const& f = run.predictionsfile();
	BOOST_CHECK_EQUAL(f.filename(), CachedFile::cacheDirectory() / ("run_" + boost::lexical_cast<std::string>(runID) + ".arff"));
	BOOST_CHECK(boost::filesystem::exists(f.filename()));

	// remove cached file
	std::remove(f.filename().string().c_str());

	// obtain the same run by ID from the server
	Run run2(runID);
	BOOST_CHECK_EQUAL(run2.id(), runID);

	// basic getters
	BOOST_CHECK_EQUAL(run2.task(), task);
	BOOST_CHECK_EQUAL(run2.flow(), flow);

	// hyperparameters
	BOOST_CHECK_EQUAL(run2.numberOfHyperparameters(), 2);
	BOOST_CHECK_CLOSE(run2.hyperparameterValue<double>("a"), 3.14, 1e-14);
	BOOST_CHECK_EQUAL(run2.hyperparameterValue<int>("b"), 42);

	// predictions
	typedef Data<unsigned int>::element_range Elements;
	for (std::size_t f=0; f<10; f++)
	{
		Data<unsigned int> p1 = folds.validation(f).labels();
		Data<unsigned int> p2;
		run2.predictions(0, f, p2);

		// check that number of predictions per fold match
		BOOST_CHECK_EQUAL(p1.numberOfElements(), p2.numberOfElements());

		// check that individual predictions match
		Elements e1 = p1.elements();
		Elements e2 = p2.elements();
		Elements::const_iterator it1 = e1.begin();
		Elements::const_iterator it2 = e2.begin();
		for (; it1 != e1.end(); ++it1, ++it2)
		{
			BOOST_CHECK_EQUAL(*it1, *it2);
		}
	}

	// cached file
	CachedFile const& f2 = run2.predictionsfile();
	BOOST_CHECK_EQUAL(f2.filename(), CachedFile::cacheDirectory() / ("run_" + boost::lexical_cast<std::string>(runID) + ".arff"));
	BOOST_CHECK(boost::filesystem::exists(f2.filename()));

	// remove cached files
	std::remove(f2.filename().string().c_str());
	std::remove(task->dataset()->datafile().filename().string().c_str());
	std::remove(task->splitsfile().filename().string().c_str());
}


BOOST_AUTO_TEST_SUITE_END()
