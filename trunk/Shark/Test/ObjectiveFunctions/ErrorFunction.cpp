#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>

#define BOOST_TEST_MODULE ML_ErrorFunction
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
			m_name="TestModel";
		}
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


BOOST_AUTO_TEST_CASE( ML_ErrorFunction )
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
	BOOST_CHECK_SMALL(derivative.m_gradient(0)-40,1.e-15);
}
