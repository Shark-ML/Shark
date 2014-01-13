#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Rng/Uniform.h>

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

		virtual size_t inputSize()const
		{
			return m_dim;
		}
		virtual size_t outputSize()const
		{
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

		virtual void weightedParameterDerivative( RealMatrix const& input, RealMatrix const& coefficients, State const& state, RealVector& derivative)const
		{
			derivative.resize(1);
			derivative(0)=0;
			for (size_t p = 0; p < coefficients.size1(); p++)
			{
				derivative(0) +=sum(row(coefficients,p));
			}
		}
};


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
	ErrorFunction<RealVector,RealVector> mse(&model,&loss);

	mse.setDataset(dataset);

	double error=mse.eval(parameters);
	BOOST_CHECK_SMALL(error-40,1.e-15);

	//calculate derivative - it should also be 40
	ErrorFunction<RealVector,RealVector>::FirstOrderDerivative derivative;
	mse.evalDerivative(parameters,derivative);
	BOOST_CHECK_SMALL(derivative(0)-40,1.e-15);
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
	MultiVariateNormalDistribution noise;
	RealMatrix covariance(2, 2);
	covariance(0,0) = 1;
	covariance(0,1) = 0;
	covariance(1,0) = 0;
	covariance(1,1) = 1;
	noise.setCovarianceMatrix(covariance);

	Uniform<> uniform(Rng::globalRng,-3.0, 3.0);

	// create samples
	std::vector<RealVector> input(trainExamples,RealVector(2));
	std::vector<RealVector> trainTarget(trainExamples,RealVector(2));
	std::vector<RealVector> testTarget(trainExamples,RealVector(2));
	double optimalMSE = 0;
	for (size_t i=0;i!=trainExamples;++i) {
		input[i](0) = uniform();
		input[i](1) = uniform();
		testTarget[i] =  model(input[i]);
		RealVector noiseVal = noise().first;
		trainTarget[i] = noiseVal + testTarget[i];
		optimalMSE+=norm_sqr(noiseVal);
	}
	optimalMSE/=trainExamples;
	
	//create loss function and internal implementations to check everything is working
	RegressionDataset trainset = createLabeledDataFromRange(input, trainTarget);
	SquaredLoss<> loss;
	
	{
		ErrorFunction<RealVector,RealVector> mse(&model,&loss);
		mse.setDataset(trainset);
		double val = mse.eval(optimum);
		BOOST_CHECK_CLOSE(optimalMSE,val,1.e-10);
		
		ErrorFunction<RealVector,RealVector>::FirstOrderDerivative d;
		double valGrad = mse.evalDerivative(optimum,d);
		double gradNorm = norm_2(d);
		BOOST_CHECK_CLOSE(optimalMSE,valGrad,1.e-10);
		BOOST_CHECK_SMALL(gradNorm,1.-10);
		
		//let the model forget by reinitializing with random values
		initRandomNormal(model,2);
		//optimize with rprop
		IRpropPlus rprop;
		rprop.init(mse);
		for(std::size_t i = 0; i != 100; ++i){
			rprop.step(mse);
		}
		double diff = norm_sqr(rprop.solution().point-optimum);
		std::cout<<diff<<rprop.solution().point<<" "<<optimum<<std::endl;
		
		BOOST_CHECK_SMALL(diff, 1.e-3);
	}
	
	{
		detail::LossBasedErrorFunctionImpl<RealVector,RealVector,RealVector> mse(&model,&loss);
		mse.setDataset(trainset);
		double val = mse.eval(optimum);
		BOOST_CHECK_CLOSE(optimalMSE,val,1.e-10);
		
		ErrorFunction<RealVector,RealVector>::FirstOrderDerivative d;
		double valGrad = mse.evalDerivative(optimum,d);
		double gradNorm = norm_2(d);
		BOOST_CHECK_CLOSE(optimalMSE,valGrad,1.e-10);
		BOOST_CHECK_SMALL(gradNorm,1.-10);
		
		//let the model forget by reinitializing with random values
		initRandomNormal(model,2);
		//optimize with rprop
		IRpropPlus rprop;
		rprop.init(mse);
		for(std::size_t i = 0; i != 100; ++i){
			rprop.step(mse);
		}
		double diff = norm_sqr(rprop.solution().point-optimum);
		std::cout<<diff<<rprop.solution().point<<" "<<optimum<<std::endl;
		
		BOOST_CHECK_SMALL(diff, 1.e-3);
	}
	
	{
		detail::ParallelLossBasedErrorFunctionImpl<RealVector,RealVector,RealVector> mse(&model,&loss);
		mse.setDataset(trainset);
		double val = mse.eval(optimum);
		BOOST_CHECK_CLOSE(optimalMSE,val,1.e-10);
		
		ErrorFunction<RealVector,RealVector>::FirstOrderDerivative d;
		double valGrad = mse.evalDerivative(optimum,d);
		double gradNorm = norm_2(d);
		BOOST_CHECK_CLOSE(optimalMSE,valGrad,1.e-10);
		BOOST_CHECK_SMALL(gradNorm,1.-10);
		
		//let the model forget by reinitializing with random values
		initRandomNormal(model,2);
		//optimize with rprop
		IRpropPlus rprop;
		rprop.init(mse);
		for(std::size_t i = 0; i != 100; ++i){ 
			rprop.step(mse);
		}
		double diff = norm_sqr(rprop.solution().point-optimum);
		std::cout<<diff<<rprop.solution().point<<" "<<optimum<<std::endl;
		
		BOOST_CHECK_SMALL(diff, 1.e-3);
	}
	
	
}
