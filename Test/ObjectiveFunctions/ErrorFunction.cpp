#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Data/DataDistribution.h>

#define BOOST_TEST_MODULE ObjFunct_ErrorFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

class TestModel : public AbstractModel<RealVector,RealVector>
{
private:
	double m_c;
	size_t m_dim;
public:
	TestModel(int dim,double c):m_c(c),m_dim(dim){
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	std::string name() const
	{ return "TestModel"; }

	//this model doesn't use any parameters...it just pretends!
	virtual RealVector parameterVector()const{return RealVector(10);}
	virtual void setParameterVector(RealVector const& newParameters){}
	virtual size_t numberOfParameters()const{return 1;}

	Shape inputShape() const{
		return m_dim;
	}
	Shape outputShape() const{
		return m_dim;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	// adds just a value c on the input
	void eval(RealMatrix const& patterns, RealMatrix& output, State& state)const
	{
		output.resize(patterns.size1(),m_dim);
		for(std::size_t  p = 0; p != patterns.size1();++p){
			for (size_t i=0;i!=m_dim;++i)
				output(p,i)=patterns(p,i)+m_c;
		}
	}
	using AbstractModel<RealVector,RealVector>::eval;

	virtual void weightedParameterDerivative( RealMatrix const& input, RealMatrix const&, RealMatrix const& coefficients, State const& state, RealVector& derivative)const
	{
		derivative.resize(1);
		derivative(0)=0;
		for (size_t p = 0; p < coefficients.size1(); p++)
		{
			derivative(0) +=sum(row(coefficients,p));
		}
	}
};

struct TestFunction : public SingleObjectiveFunction
{
	typedef SingleObjectiveFunction Base;

	std::string name() const
	{ return "TestFunction"; }

	RealVector weights;
	TestFunction():weights(3){
		weights(0)=1;
		weights(1)=2;
		weights(2)=-1;
	}
	std::size_t numberOfVariables()const{
		return 3;
	}

	virtual double eval(RealVector const& pattern)const
	{
		return inner_prod(weights,pattern);
	}
};


BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_ErrorFunction)

BOOST_AUTO_TEST_CASE( ObjFunct_ErrorFunction_BASE )
{
	RealVector zero(10);
	zero.clear();
	std::vector<RealVector> input(3,zero);//input just zeros
	std::vector<RealVector> target=input;//target same as input
	RegressionDataset dataset = createLabeledDataFromRange(input,target);
	RealVector parameters(1);//parameters also zero
	parameters.clear();

	//every value of input gets 2 added. so the square error of each example is 10*4=40
	TestModel model(10,2);
	SquaredLoss<> loss;
	ErrorFunction<> mse(dataset, &model,&loss);

	double error=mse.eval(parameters);
	BOOST_CHECK_SMALL(error-20,1.e-15);

	//calculate derivative - it should also be 40
	ErrorFunction<>::FirstOrderDerivative derivative;
	mse.evalDerivative(parameters,derivative);
	BOOST_CHECK_SMALL(derivative(0)-20,1.e-15);
}

//test whether we can get to the same result as linear regression
//this is very similar to the linear regrssion test. first we create a model,
//let it evaluate some responses + gaussian noise. then we let it forget
//and create an ErrorFunction which is supposed to have a minimum at the point.
BOOST_AUTO_TEST_CASE( ObjFunct_ErrorFunction_LinearRegression ){
	const size_t trainExamples = 60000;
	LinearRegression trainer;
	LinearModel<> model;
	RealMatrix matrix(2, 2);
	RealVector offset(2);
	matrix(0,0) = 3;
	matrix(1,1) = -5;
	matrix(0,1) = -2;
	matrix(1,0) = 7;
	offset(0) = 3;
	offset(1) = -6;
	model.setStructure(matrix, offset);
	RealVector optimum=model.parameterVector();

	// create datatset - the model output + gaussian noise
	RealMatrix covariance(2, 2);
	covariance(0,0) = 1;
	covariance(0,1) = 0;
	covariance(1,0) = 0;
	covariance(1,1) = 1;
	MultiVariateNormalDistribution noise(covariance);

	// create samples
	std::vector<RealVector> input(trainExamples,RealVector(2));
	std::vector<RealVector> trainTarget(trainExamples,RealVector(2));
	std::vector<RealVector> testTarget(trainExamples,RealVector(2));
	double optimalMSE = 0;
	for (size_t i=0;i!=trainExamples;++i) {
		input[i](0) = random::uni(random::globalRng, -3.0,3.0);
		input[i](1) = random::uni(random::globalRng, -3.0,3.0);
		testTarget[i] =  model(input[i]);
		RealVector noiseVal = noise(random::globalRng).first;
		trainTarget[i] = noiseVal + testTarget[i];
		optimalMSE+=norm_sqr(noiseVal);
	}
	optimalMSE/=2*trainExamples;
	
	//create loss function and internal implementations to check everything is working
	RegressionDataset trainset = createLabeledDataFromRange(input, trainTarget);
	SquaredLoss<> loss;
	
	{
		ErrorFunction<> mse(trainset, &model,&loss);
		double val = mse.eval(optimum);
		BOOST_CHECK_CLOSE(optimalMSE,val,1.e-10);
		
		ErrorFunction<>::FirstOrderDerivative d;
		double valGrad = mse.evalDerivative(optimum,d);
		double gradNorm = norm_2(d);
		BOOST_CHECK_CLOSE(optimalMSE,valGrad,1.e-10);
		BOOST_CHECK_SMALL(gradNorm,1.e-1);
		
		//let the model forget by reinitializing with random values
		initRandomNormal(model,2);
		//optimize with rprop
		Rprop<> rprop;
		rprop.init(mse);
		for(std::size_t i = 0; i != 100; ++i){
			rprop.step(mse);
		}
		double diff = norm_sqr(rprop.solution().point-optimum);
		std::cout<<diff<<rprop.solution().point<<" "<<optimum<<std::endl;
		
		BOOST_CHECK_SMALL(diff, 1.e-3);
	}
	
	{
		detail::ErrorFunctionImpl<RealVector,RealVector,RealVector,RealVector> mse(trainset,&model,&loss,false);
		double val = mse.eval(optimum);
		BOOST_CHECK_CLOSE(optimalMSE,val,1.e-10);
		
		ErrorFunction<>::FirstOrderDerivative d;
		double valGrad = mse.evalDerivative(optimum,d);
		double gradNorm = norm_2(d);
		BOOST_CHECK_CLOSE(optimalMSE,valGrad,1.e-10);
		BOOST_CHECK_SMALL(gradNorm,1.e-1);
		
		//let the model forget by reinitializing with random values
		initRandomNormal(model,2);
		//optimize with rprop
		Rprop<> rprop;
		rprop.init(mse);
		for(std::size_t i = 0; i != 100; ++i){
			rprop.step(mse);
		}
		double diff = norm_sqr(rprop.solution().point-optimum);
		std::cout<<diff<<rprop.solution().point<<" "<<optimum<<std::endl;
		
		BOOST_CHECK_SMALL(diff, 1.e-3);
	}
}

BOOST_AUTO_TEST_CASE( ObjFunct_WeightedErrorFunction_LinearRegression )
{
	WeightedLabeledData<RealVector,RealVector> weightedData;
	RegressionDataset unweightedData(1);
	{
		Wave problem;
		RegressionDataset data = problem.generateDataset(50,50);
		UnlabeledData<double> weights(1);
		weights.batch(0).resize(50);
		
		unweightedData.batch(0).input.resize(100,1);
		unweightedData.batch(0).label.resize(100,1);
		for(std::size_t i = 0; i != 100; ++i){
			std::size_t e = random::discrete(random::globalRng, 0,49);
			unweightedData.element(i).input = data.element(e).input;
			unweightedData.element(i).label = data.element(e).label;
			weights.element(e) += 1.0;
		}
		weightedData = WeightedLabeledData<RealVector,RealVector>(data, weights);
	}
	weightedData.repartition(std::vector<std::size_t>({10,10,10,10,10}));

	LinearModel<> model;
	model.setStructure(1,1,true);
	SquaredLoss<> loss;
	ErrorFunction<> unweightedError(unweightedData, &model,&loss);
	ErrorFunction<> weightedError(weightedData, &model,&loss);
	RealVector point(2,0.0);

	ErrorFunction<>::FirstOrderDerivative unWDerivative;
	double unWError1 = unweightedError.eval(point);
	double unWError2 = unweightedError.evalDerivative(point,unWDerivative);
	
	ErrorFunction<>::FirstOrderDerivative WDerivative;
	double WError1 = weightedError.eval(point);
	double WError2 = weightedError.evalDerivative(point,WDerivative);
	
	BOOST_CHECK_CLOSE(unWError1, unWError2,1.e-11);
	BOOST_CHECK_CLOSE(WError1, WError2,1.e-11);
	BOOST_CHECK_CLOSE(WError1, unWError1,1.e-11);
	
	BOOST_CHECK_SMALL(norm_sqr(unWDerivative - WDerivative),1.e-8);
}

BOOST_AUTO_TEST_CASE( ObjFunct_ErrorFunction_Noisy )
{
	//create regression data from the testfunction
	TestFunction function;
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(3);
	RealVector output(1);

	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=3;++j)
		{
			input(j)=random::uni(random::globalRng, -1,1);
		}
		data.push_back(input);
		output(0)=function.eval(input);
		target.push_back(output);
	}
	// batchsize 1 corresponds to stochastic gradient descent
	RegressionDataset dataset = createLabeledDataFromRange(data,target,1);

	//startingPoint
	RealVector point(3);
	point(0) = 0;
	point(1) = 0;
	point(2) = 0;
	SteepestDescent<> optimizer;
	SquaredLoss<> loss;
	LinearModel<> model(3);
	
	ErrorFunction<> mse(dataset,&model,&loss, true);
	mse.init();
	optimizer.init(mse, point);
	// train the cmac
	double error = 0.0;
	for (size_t iteration=0; iteration<701; ++iteration){
		optimizer.step(mse);
		if (iteration % 100 == 0){
			error = optimizer.solution().value;
			RealVector best = optimizer.solution().point;
			std::cout << iteration << " error:" << error << " parameter:" << best << std::endl;
		}
	}
	std::cout << "Optimization done. Error:" << error << std::endl;
	BOOST_CHECK_SMALL(error, 1.e-15);
}

BOOST_AUTO_TEST_SUITE_END()
