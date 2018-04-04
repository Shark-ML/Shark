#include <shark/Models/Ensemble.h>
#include <shark/Models/LinearModel.h>

#define BOOST_TEST_MODULE Models_Ensemble
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <sstream>
#include <shark/Core/Random.h>

using namespace std;
using namespace boost::archive;
using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_Ensemble)

BOOST_AUTO_TEST_CASE( Ensemble_Test )
{
	std::vector<LinearModel<> > models(5);
	std::vector<LinearClassifier<> > modelsClass(5);
	Ensemble<LinearModel<> > linEnsemble;
	Ensemble<LinearModel<>* > linEnsemblePtr;
	Ensemble<LinearModel<>, unsigned int> linEnsembleClass;
	Ensemble<LinearClassifier<> > classEnsemble;
	Ensemble<LinearClassifier<>* > classEnsemblePtr;
	Ensemble<LinearClassifier<>*, RealVector> classEnsembleLin;


	RealMatrix weights(4,2,0.0);
	RealVector bias(4,0.0);
	double alphaSum = 0.0;
	
	for(std::size_t i = 0; i != 5; ++i){
		//create parameters of linear models
		RealMatrix curWeights = blas::normal(random::globalRng, 4, 2, 0.0, 1.0, blas::cpu_tag());
		RealVector curBias = blas::normal(random::globalRng, 4, 0.0, 1.0, blas::cpu_tag());
		double curAlpha = random::uni(random::globalRng,0.1,1);
		alphaSum +=curAlpha;
		weights += curAlpha * curWeights;
		bias +=curAlpha * curBias;
		
		//create ensembles of linear models
		models[i].setStructure(curWeights,curBias);
		linEnsemble.addModel(models[i],curAlpha);
		linEnsemblePtr.addModel(&models[i],curAlpha);
		linEnsembleClass.addModel(models[i],curAlpha);
		
		//create ensembles of classifiers
		modelsClass[i].setStructure(curWeights,curBias);
		classEnsemble.addModel(modelsClass[i],curAlpha);
		classEnsemblePtr.addModel(&modelsClass[i],curAlpha);
		classEnsembleLin.addModel(&modelsClass[i],curAlpha);
		
		//check that adding the model worked
		BOOST_CHECK_EQUAL(linEnsemble.weight(i), curAlpha);
		BOOST_CHECK_EQUAL(linEnsemble.numberOfModels(), i+1);
	}
	//create the ground truth model for ensemble of linear models
	weights/=alphaSum;
	bias/=alphaSum;
	LinearModel<> linear(weights,bias);
	LinearClassifier<> linearClass(weights,bias);

	//test whether model responses agree
	//first the ensembles of linear models
	for(std::size_t i = 0; i != 100; ++i){
		//the testpoint2
		RealMatrix point = blas::uniform(random::globalRng, 5, 2, 0.0, 1.0, blas::cpu_tag());
		
		RealMatrix truth = linear(point);
		RealMatrix test = linEnsemble(point);
		RealMatrix testPtr = linEnsemblePtr(point);
		RealMatrix testDecisionFun = linEnsembleClass.decisionFunction()(point);
		BOOST_CHECK_SMALL(max(abs(truth-test)),1.e-10);
		BOOST_CHECK_SMALL(max(abs(truth-testPtr)),1.e-10);
		BOOST_CHECK_SMALL(max(abs(truth-testDecisionFun)),1.e-10);
		
		UIntVector truthClass = linearClass(point);
		UIntVector testClass = linEnsembleClass(point);
		for(std::size_t i = 0; i != point.size1(); ++i){
			BOOST_CHECK_EQUAL(truthClass(i), testClass(i));
		}
	}
		
	//now the ensembles of classifiers
	for(std::size_t i = 0; i != 100; ++i){
		//the testpoint2
		RealMatrix point = blas::uniform(random::globalRng, 5, 2, 0.0, 1.0, blas::cpu_tag());
		
		//create ground truth of decision function and response
		RealMatrix truthVotes(5,4,0.0);
		for(std::size_t i = 0; i != models.size(); ++i){
			UIntVector response = modelsClass[i](point);
			for(std::size_t j = 0; j != point.size1(); ++j){
				truthVotes(j, response(j) ) += linEnsemble.weight(i);
			}
		}
		truthVotes /= alphaSum;
		UIntVector truthLabel(point.size1(),0);
		for(std::size_t j = 0; j != point.size1(); ++j){
			truthLabel(j) = arg_max(row(truthVotes,j));
		}
		UIntVector testClass = classEnsemble(point);
		UIntVector testClassPtr = classEnsemblePtr(point);
		RealMatrix testResponse = classEnsemble.decisionFunction()(point);
		RealMatrix testResponsePtr = classEnsemblePtr.decisionFunction()(point);
		RealMatrix testResponseLin = classEnsembleLin(point);
		for(std::size_t i = 0; i != point.size1(); ++i){
			BOOST_CHECK_EQUAL(truthLabel(i), testClass(i));
			BOOST_CHECK_EQUAL(truthLabel(i), testClassPtr(i));
		}
		BOOST_CHECK_SMALL(max(abs(testResponse - truthVotes)), 1.e-10);
		BOOST_CHECK_SMALL(max(abs(testResponsePtr - truthVotes)), 1.e-10);
		BOOST_CHECK_SMALL(max(abs(testResponseLin - truthVotes)), 1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( Ensemble_Serialize )
{
	
	Ensemble<LinearModel<>, unsigned int> model;
	Ensemble<LinearModel<> > modelLin;
	for(std::size_t i = 0; i != 5; ++i){
		//create parameters of linear models
		RealMatrix curWeights = blas::normal(random::globalRng, 4, 2, 0.0, 1.0, blas::cpu_tag());
		RealVector curBias = blas::normal(random::globalRng, 4, 0.0, 1.0, blas::cpu_tag());
		double curAlpha = random::uni(random::globalRng,0.1,1);
		
		model.addModel(LinearModel<>(curWeights,curBias),curAlpha);
		modelLin.addModel(LinearModel<>(curWeights,curBias),curAlpha);
	}

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	std::vector<unsigned int > labels;
	RealVector input(2);
	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=2;++j)
		{
			input(j)=random::uni(random::globalRng,-1,1);
		}
		data.push_back(input);
		target.push_back(model.decisionFunction()(input));
		labels.push_back(model(input));
	}

	//now we serialize the model
	ostringstream outputStream;
	{
		TextOutArchive oa(outputStream);  
		oa << model;
		oa << modelLin;
	}
	//and create a new model from the serialization
	Ensemble<LinearModel<>, unsigned int > modelDeserialized;
	Ensemble<LinearModel<> > modelLinDeserialized;
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	ia >> modelLinDeserialized;
	
	BOOST_CHECK_EQUAL(modelDeserialized.inputShape(), Shape({2}));
	BOOST_CHECK_EQUAL(modelLinDeserialized.inputShape(), Shape({2}));
	BOOST_CHECK_EQUAL(modelDeserialized.outputShape(), Shape({4}));
	BOOST_CHECK_EQUAL(modelLinDeserialized.outputShape(), Shape({4}));
	
	for (size_t i=0; i<1000; i++)
	{
		unsigned int label = modelDeserialized(data[i]);
		RealVector f = modelDeserialized.decisionFunction()(data[i]);
		RealVector fLin = modelLinDeserialized(data[i]);
		BOOST_CHECK_SMALL(norm_inf(f - target[i]),1.e-7);
		BOOST_CHECK_SMALL(norm_inf(fLin - target[i]),1.e-7);
		BOOST_CHECK_EQUAL(label, labels[i]);
	}
}

BOOST_AUTO_TEST_SUITE_END()
