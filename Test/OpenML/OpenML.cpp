#define BOOST_TEST_MODULE OpenML_OpenML
#include <boost/test/unit_test.hpp>

#include <shark/OpenML/OpenML.h>
#include <shark/Data/Arff.h>

#include <boost/lexical_cast.hpp>
#include <cstdio>
#include <thread>
#include <chrono>

using namespace shark;
using namespace openML;


static const std::string demo_api_key = "0412af014f75cb8162542e34a0c86b2f";


BOOST_AUTO_TEST_SUITE(OpenML_OpenML)

BOOST_AUTO_TEST_CASE(OpenML_Workflow)
{
	// prepare connection to the test server
	openML::connection.enableTestMode();
	openML::connection.setKey(demo_api_key);

	// create a dummy data set
	std::string arff = "% for unit testing only, please ignore\n"
	                   "@relation unit-test-data\n"
	                   "@attribute 'number' numeric\n"
	                   "@attribute 'animal' {cat, dog, mouse, horse, donkey}\n"
	                   "@attribute 'class' {good, bad}\n"
	                   "@data\n"
	                   "3.14, cat, good\n"
	                   "42, cat, bad\n"
	                   "-1.23456789, cat, good\n"
	                   "1.123456789e50, cat, bad\n"
	                   "3.14, dog, good\n"
	                   "42, dog, bad\n"
	                   "-1.23456789, dog, good\n"
	                   "1.123456789e50, dog, bad\n"
	                   "3.14, mouse, good\n"
	                   "42, mouse, bad\n"
	                   "-1.23456789, mouse, good\n"
	                   "1.123456789e50, mouse, bad\n"
	                   "3.14, horse, good\n"
	                   "42, horse, bad\n"
	                   "-1.23456789, horse, good\n"
	                   "1.123456789e50, horse, bad\n"
	                   "3.14, donkey, good\n"
	                   "42, donkey, bad\n"
	                   "-1.23456789, donkey, good\n"
	                   "1.123456789e50, donkey, bad\n";
	openML::IDType datasetID = openML::createDataset(
	                   arff,
	                   "unit-test-data",
	                   "for unit testing only, please ignore",
	                   "class");

	std::cout << "data set ID: " << datasetID << std::endl;

	// construct a data set object from the ID
	std::cout << "please wait while OpenML processes the newly uploaded data set " << std::flush;
	std::shared_ptr<openML::Dataset> ds;
	for (int trials=0; trials<50; trials++)
	{
		try
		{
			ds = openML::Dataset::get(datasetID);
			break;
		}
		catch (...)
		{
			// give the server some time to digest the upload
			std::cout << "." << std::flush;
			std::this_thread::sleep_for(std::chrono::seconds(3));
		}
	}
	if (! ds)
	{
		std::cout << " giving up." << std::endl;
		openML::deleteDataset(datasetID);
		throw SHARKEXCEPTION("failed to load data set");
	}
	std::cout << " done." << std::endl;

	ds->print();
	BOOST_CHECK_EQUAL(ds->id(), datasetID);

	// basic getters
	BOOST_CHECK_EQUAL(ds->name(), "unit-test-data");
	BOOST_CHECK_EQUAL(ds->description(), "for unit testing only, please ignore");
	BOOST_CHECK_EQUAL(ds->format(), "arff");
	BOOST_CHECK_EQUAL(ds->visibility(), "public");

	// attribute getters
	BOOST_CHECK_EQUAL(ds->numberOfAttributes(), 3);
	BOOST_CHECK_EQUAL(ds->attribute(0).type, NUMERIC);
	BOOST_CHECK_EQUAL(ds->attribute(0).name, "number");
	BOOST_CHECK_EQUAL(ds->attribute(0).target, false);
	BOOST_CHECK_EQUAL(ds->attribute(0).ignore, false);
	BOOST_CHECK_EQUAL(ds->attribute(0).rowIdentifier, false);
	BOOST_CHECK_EQUAL(ds->attribute(1).type, NOMINAL);
	BOOST_CHECK_EQUAL(ds->attribute(1).name, "animal");
	BOOST_CHECK_EQUAL(ds->attribute(1).target, false);
	BOOST_CHECK_EQUAL(ds->attribute(1).ignore, false);
	BOOST_CHECK_EQUAL(ds->attribute(1).rowIdentifier, false);
	BOOST_CHECK_EQUAL(ds->attribute(2).type, NOMINAL);
	BOOST_CHECK_EQUAL(ds->attribute(2).name, "class");
	BOOST_CHECK_EQUAL(ds->attribute(2).target, true);
	BOOST_CHECK_EQUAL(ds->attribute(2).ignore, false);
	BOOST_CHECK_EQUAL(ds->attribute(2).rowIdentifier, false);
	BOOST_CHECK_EQUAL(ds->attributeIndex("number"), 0);
	BOOST_CHECK_EQUAL(ds->attributeIndex("animal"), 1);
	BOOST_CHECK_EQUAL(ds->attributeIndex("class"), 2);

	// cached file
	openML::CachedFile const& f1 = ds->datafile();
	BOOST_CHECK_EQUAL(f1.filename(), CachedFile::cacheDirectory() / ("dataset_" + boost::lexical_cast<std::string>(datasetID) + ".arff"));
	f1.download();
	BOOST_CHECK(boost::filesystem::exists(f1.filename()));

	// file content
	ClassificationDataset data;
	importARFF(f1.filename().string(), "class", data);
	BOOST_CHECK_EQUAL(data.numberOfElements(), 20);
	BOOST_CHECK_EQUAL(inputDimension(data), 6);   // 1 + 5, the latter due to one-hot-encoding
	BOOST_CHECK_EQUAL(numberOfClasses(data), 2);

	// tag the data set
	ds->tag("shark-unit-test");
	BOOST_CHECK(ds->tags().count("shark-unit-test") > 0);

	// create a task for the data set
	openML::IDType taskID = openML::createSupervisedClassificationTask(ds);

	std::cout << "task ID: " << taskID << std::endl;

	// construct a task object from the ID
	std::this_thread::sleep_for(std::chrono::seconds(3));   // let the server breath...
	std::shared_ptr<openML::Task> task = openML::Task::get(taskID);

	task->print();
	BOOST_CHECK_EQUAL(task->id(), taskID);

	// basic getters
	BOOST_CHECK_EQUAL(task->dataset(), ds);
	BOOST_CHECK_EQUAL(task->targetAttribute(), "class");
	BOOST_CHECK_EQUAL(task->repetitions(), 1);
	BOOST_CHECK_EQUAL(task->folds(), 10);

	// cached file
	openML::CachedFile const& f2 = task->splitsfile();
	BOOST_CHECK_EQUAL(f2.filename(), CachedFile::cacheDirectory() / ("task_" + boost::lexical_cast<std::string>(taskID) + "_splits.arff"));
	f2.download();
	BOOST_CHECK(boost::filesystem::exists(f2.filename()));

	// tag the task
	task->tag("shark-unit-test");
	BOOST_CHECK(task->tags().count("shark-unit-test") > 0);

	// construct flow
	IDType flowID = findFlow("shark-unit-test-flow");
	if (flowID != invalidID) deleteFlow(flowID);
	std::vector<Hyperparameter> hyperparameters {
				Hyperparameter("a", "first dummy parameter", "numeric"),
				Hyperparameter("b", "second dummy parameter", "integer"),
			};
	std::shared_ptr<Flow> flow = Flow::create("shark-unit-test-flow", "for unit testing only, please ignore", hyperparameters);
	flowID = flow->id();

	std::cout << "flow ID: " << flowID << std::endl;

	// basic getters
	BOOST_CHECK_EQUAL(flow->name(), "shark-unit-test-flow");
	BOOST_CHECK_EQUAL(flow->version(), Flow::sharkVersion());
	BOOST_CHECK_EQUAL(flow->description(), "for unit testing only, please ignore");

	// hyperparameters
	BOOST_CHECK_EQUAL(flow->numberOfHyperparameters(), 2);
	BOOST_CHECK_EQUAL(flow->hyperparameter(0).name, "a");
	BOOST_CHECK_EQUAL(flow->hyperparameter(0).description, "first dummy parameter");
	BOOST_CHECK_EQUAL(flow->hyperparameter(0).datatype, "numeric");
	BOOST_CHECK_EQUAL(flow->hyperparameter(1).name, "b");
	BOOST_CHECK_EQUAL(flow->hyperparameter(1).description, "second dummy parameter");
	BOOST_CHECK_EQUAL(flow->hyperparameter(1).datatype, "integer");
	BOOST_CHECK_EQUAL(flow->hyperparameterIndex("a"), 0);
	BOOST_CHECK_EQUAL(flow->hyperparameterIndex("b"), 1);

	// tag the flow
	flow->tag("shark-unit-test");
	BOOST_CHECK(flow->tags().count("shark-unit-test") > 0);

	// construct run from task and flow
	Run run(task, flow);

	// set hyperparameters
	BOOST_CHECK_EQUAL(run.numberOfHyperparameters(), 2);
	run.setHyperparameterValue("a", 3.14);
	run.setHyperparameterValue("b", 42);

	// obtain data splits from the task
	CVFolds<ClassificationDataset> folds = task->split(0, data);
	BOOST_REQUIRE_EQUAL(folds.size(), 10);

	// store test labels as predictions
	// (in terms of machine learning this is cheating! but it is a useful unit test)
	for (std::size_t f=0; f<10; f++)
	{
		auto labels = folds.validation(f).labels();
		run.setPredictions(0, f, labels);
	}

	// store the "predictions" on the server
	run.commit();

	// check the run ID
	IDType runID = run.id();
	BOOST_CHECK(runID != invalidID);

	// tag the run
	run.tag("shark-unit-test");
	BOOST_CHECK(run.tags().count("shark-unit-test") > 0);

	// cached file
	CachedFile const& f3 = run.predictionsfile();
	BOOST_CHECK_EQUAL(f3.filename(), CachedFile::cacheDirectory() / ("run_" + boost::lexical_cast<std::string>(runID) + ".arff"));
	BOOST_CHECK(boost::filesystem::exists(f3.filename()));

	// remove the cached predictions file
	std::remove(f3.filename().string().c_str());
	BOOST_CHECK(! boost::filesystem::exists(f3.filename()));

	// obtain the same run by it ID from the server
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
		BOOST_REQUIRE_EQUAL(p1.numberOfElements(), p2.numberOfElements());

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
	CachedFile const& f4 = run2.predictionsfile();
	BOOST_CHECK_EQUAL(f4.filename(), CachedFile::cacheDirectory() / ("run_" + boost::lexical_cast<std::string>(runID) + ".arff"));
	BOOST_CHECK(boost::filesystem::exists(f4.filename()));

	// untag the run, here using a different object than for tagging
	run2.untag("shark-unit-test");
	BOOST_CHECK(run.tags().count("shark-unit-test") == 0);

	// untag the flow
	flow->untag("shark-unit-test");
	BOOST_CHECK(flow->tags().count("shark-unit-test") == 0);

	// untag the task
	task->untag("shark-unit-test");
	BOOST_CHECK(task->tags().count("shark-unit-test") == 0);

	// untag the data set
	ds->untag("shark-unit-test");
	BOOST_CHECK(ds->tags().count("shark-unit-test") == 0);

	// remove the run from the server
	openML::deleteRun(run2);

	// remove the flow from the server
	openML::deleteFlow(flow);

	// remove the task from the server
	openML::deleteTask(task);

	// remove the data set from the server
	openML::deleteDataset(ds);

	// remove the cached predictions file
	std::remove(f4.filename().string().c_str());

	// remove the cached splits file
	std::remove(f2.filename().string().c_str());

	// remove the cached data set file
	std::remove(f1.filename().string().c_str());
}

BOOST_AUTO_TEST_CASE(OpenML_Query)
{
	// prepare connection to the test server
	openML::connection.enableTestMode();
	openML::connection.setKey(demo_api_key);

	// query data sets
	{
		openML::QueryResult q = openML::queryDatasets("/limit/10/offset/20");
		BOOST_CHECK(q.size() == 10);
	}

	// query tasks
	{
		openML::QueryResult q = openML::queryTasks("/limit/10/offset/20");
		BOOST_CHECK(q.size() == 10);
	}

	// query flows
	{
		openML::QueryResult q = openML::queryFlows("/limit/10/offset/20");
		BOOST_CHECK(q.size() == 10);
	}

	// query runs
	{
		openML::QueryResult q = openML::queryRuns("/limit/10/offset/20");
		BOOST_CHECK(q.size() == 10);
	}
}


BOOST_AUTO_TEST_SUITE_END()
