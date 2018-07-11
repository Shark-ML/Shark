#define BOOST_TEST_MODULE Data_Generators
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/DataDistribution.h>
#include <shark/Data/Statistics.h>
#include <shark/Models/LinearModel.h>

using namespace shark;



BOOST_AUTO_TEST_SUITE (Data_Generators)

//in this suit, we check:
//generation using op()
//transform, transformInputs, transformLabels w/o model
//We make sure, that at least one of those tests has a different output than input type

//check whether the datapoints of a specific ditribution fulfill the real distribution
BOOST_AUTO_TEST_CASE( Data_Basic_Test){
	NormalDistributedPoints distId(2);
	Generator<RealVector> genId = distId.generator(100,100);
	

	RealVector b = {-2.0, 2.0};
	RealMatrix A={{2, 0}, {0,1}};
	RealMatrix C=A % trans(A);
	NormalDistributedPoints distC(C,b);
	Generator<RealVector> genC = distC.generator(100,100);
	Generator<RealVector> genA = transform(genId, [&](RealVector const& vec){return eval_block(b + A % vec);}, {2});
	LinearModel<RealVector> model(A,b);
	Generator<RealVector> genA2 = transform(genId, model);
	
	
	Data<RealVector> dataId;
	Data<RealVector> dataC;
	Data<RealVector> dataA;
	Data<RealVector> dataA2;
	for(std::size_t i = 0; i != 1000; ++i){
		dataId.push_back(genId());
		dataC.push_back(genC());
		dataA.push_back(genA());
		dataA2.push_back(genA2());
	}
	dataId.setShape(genId.shape());
	dataC.setShape(genC.shape());
	dataA.setShape(genA.shape());
	dataA2.setShape(genA2.shape());
	
	//test ID
	{
		BOOST_CHECK_EQUAL(dataId.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genId.shape(), 2);
		RealMatrix cov;
		RealVector mean;
		meanvar(dataId,mean,cov);
		BOOST_CHECK_SMALL(norm_sqr(mean),1.e-2);
		BOOST_CHECK_SMALL(sum(sqr(cov - blas::identity_matrix<double>(2))),1.e-4);
	}
	
	//test C
	{
		BOOST_CHECK_EQUAL(dataC.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genC.shape(), 2);
		RealMatrix cov;
		RealVector mean;
		meanvar(dataC,mean,cov);
		BOOST_CHECK_SMALL(norm_sqr(mean - b),1.e-2);
		BOOST_CHECK_SMALL(sum(sqr(cov - C)),1.e-2);
	}
	
	//test A
	{
		BOOST_CHECK_EQUAL(dataA.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genA.shape(), 2);
		RealMatrix cov;
		RealVector mean;
		meanvar(dataA,mean,cov);
		BOOST_CHECK_SMALL(norm_sqr(mean - b),1.e-2);
		BOOST_CHECK_SMALL(sum(sqr(cov - C)),1.e-2);
	}
	
	//test A2
	{
		BOOST_CHECK_EQUAL(dataA2.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genA2.shape(), 2);
		RealMatrix cov;
		RealVector mean;
		meanvar(dataA2,mean,cov);
		BOOST_CHECK_SMALL(norm_sqr(mean - b),1.e-2);
		BOOST_CHECK_SMALL(sum(sqr(cov - C)),1.e-2);
	}
}


BOOST_AUTO_TEST_CASE( Data_TransformInputs_Test){
	Chessboard problem;
	LabeledDataGenerator<RealVector, unsigned int> gen = problem.generator(100,100);
	

	RealVector b = {-2.0, 2.0};
	RealMatrix A={{2, 0}, {0,1}};
	LabeledDataGenerator<RealVector, unsigned int> genA = transformInputs(gen, [&](RealVector const& vec){return eval_block(b + A % vec);}, {2});
	LinearModel<RealVector> model(A,b);
	LabeledDataGenerator<RealVector, unsigned int> genA2 = transformInputs(gen, model);
	
	LabeledData<RealVector, unsigned int> data;
	LabeledData<RealVector, unsigned int> dataA;
	LabeledData<RealVector, unsigned int> dataA2;
	for(std::size_t i = 0; i != 1000; ++i){
		data.push_back(gen());
		dataA.push_back(genA());
		dataA2.push_back(genA2());
	}
	data.setShape(gen.shape());
	dataA.setShape(genA.shape());
	dataA2.setShape(genA2.shape());
	
	
	RealMatrix C0;
	RealVector b0;
	meanvar(data.inputs(),b0,C0);
	RealVector expMean = A % b0+b;
	RealMatrix expCovar = A % C0 % trans(A);
	
	//test A
	{
		BOOST_CHECK_EQUAL(dataA.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genA.shape().input, 2);
		BOOST_CHECK_EQUAL(genA.shape().label, 2);
		RealMatrix cov;
		RealVector mean;
		meanvar(dataA.inputs(),mean,cov);
		BOOST_CHECK_SMALL(norm_sqr(mean - expMean),1.e-2);
		BOOST_CHECK_SMALL(sum(sqr(cov - expCovar)),1.e-2);
	}
	
	//test A2
	{
		BOOST_CHECK_EQUAL(dataA2.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genA2.shape().input, 2);
		BOOST_CHECK_EQUAL(genA2.shape().label, 2);
		RealMatrix cov;
		RealVector mean;
		meanvar(dataA2.inputs(),mean,cov);
		BOOST_CHECK_SMALL(norm_sqr(mean - expMean),1.e-2);
		BOOST_CHECK_SMALL(sum(sqr(cov - expCovar)),1.e-2);
	}
}

class TestModel : public AbstractModel<unsigned int, RealVector, RealVector>{

	std::string name() const
	{ return "Test"; }
	
	ParameterVectorType parameterVector() const{return {};}

	void setParameterVector(ParameterVectorType const&){}

	Shape inputShape() const{
		return {2};
	}
	Shape outputShape() const{
		return {1};
	}
	
	void eval(UIntVector const& input, RealMatrix& output, State& state)const{
		output.resize(input.size(),1);
		noalias(column(output,0)) = 2.0 * input;
	}
};

BOOST_AUTO_TEST_CASE( Data_TransformLabels_Test){
	Chessboard problem;
	LabeledDataGenerator<RealVector, unsigned int> gen = problem.generator(100,100);

	LabeledDataGenerator<RealVector, RealVector> genA = transformLabels(gen, [&](unsigned int i){return RealVector(1,2.0 * i);}, {1});
	TestModel model;
	LabeledDataGenerator<RealVector, RealVector> genA2 = transformLabels(gen, model);
	
	LabeledData<RealVector, unsigned int> data;
	LabeledData<RealVector, RealVector> dataA;
	LabeledData<RealVector, RealVector> dataA2;
	for(std::size_t i = 0; i != 1000; ++i){
		data.push_back(gen());
		dataA.push_back(genA());
		dataA2.push_back(genA2());
	}
	data.setShape(gen.shape());
	dataA.setShape(genA.shape());
	dataA2.setShape(genA2.shape());
	
	auto counts = classSizes(data);
	double expected = counts[1]*2.0/1000.0/100.0;
	
	
	//test A
	{
		BOOST_CHECK_EQUAL(dataA.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genA.shape().input, 2);
		BOOST_CHECK_EQUAL(genA.shape().label, 1);
		double result = mean(dataA.labels())(0);
		BOOST_CHECK_SMALL(sqr(result - expected),1.e-2);
	}
	
	//test A2
	{
		BOOST_CHECK_EQUAL(dataA2.numberOfElements(), 1000*100);
		BOOST_CHECK_EQUAL(genA2.shape().input, 2);
		BOOST_CHECK_EQUAL(genA2.shape().label, 1);
		double result = mean(dataA2.labels())(0);
		BOOST_CHECK_SMALL(sqr(result - expected),1.e-2);
	}
}


BOOST_AUTO_TEST_SUITE_END()
