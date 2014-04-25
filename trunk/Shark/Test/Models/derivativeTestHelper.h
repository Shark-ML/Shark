#ifndef SHARK_TEST_DERIVATIVETESTHELPER_H
#define SHARK_TEST_DERIVATIVETESTHELPER_H

#include <vector>
#include <shark/LinAlg/Base.h>
#include <shark/Models/AbstractModel.h>

#include  <shark/Rng/GlobalRng.h>

namespace shark{
//estimates Derivative using the formula:
//df(x)/dx~=(f(x+e)-f(x-e))/2e
template<class Model,class Point>
std::vector<RealVector> estimateDerivative(Model& net,const Point& point,double epsilon=1.e-10){
	std::size_t outputSize = net(point).size();

	RealVector parameters=net.parameterVector();
	std::vector<RealVector> gradients(parameters.size(),RealVector(outputSize));
	for(size_t parameter=0;parameter!=parameters.size();++parameter){
		RealVector testPoint1=parameters;
		testPoint1(parameter)+=epsilon;
		net.setParameterVector(testPoint1);
		RealVector result1=net(point);

		RealVector testPoint2=parameters;
		testPoint2(parameter)-=epsilon;
		net.setParameterVector(testPoint2);
		RealVector result2=net(point);

		gradients[parameter]=(result1-result2)/(2*epsilon);
	}
	return gradients;
}
template<class Model,class Point>
std::vector<RealMatrix> estimateSecondDerivative(Model& net,const Point& point,double epsilon=1.e-10){
	std::size_t outputSize = net(point).size();

	RealVector parameters=net.parameterVector();
	std::vector<RealMatrix> hessians(parameters.size(),RealMatrix(parameters.size(),outputSize));
	for(size_t parameter=0;parameter!=parameters.size();++parameter){
		RealVector testPoint1=parameters;
		testPoint1(parameter)+=epsilon;
		net.setParameterVector(testPoint1);
		std::vector<RealVector> grad1 = estimateDerivative(net,point);

		RealVector testPoint2=parameters;
		testPoint2(parameter)-=epsilon;
		net.setParameterVector(testPoint2);
		std::vector<RealVector> grad2 = estimateDerivative(net,point);

		for(size_t param = 0; param != parameters.size(); ++param){
			row(hessians[parameter],param)=(grad1[param]-grad2[param])/(2*epsilon);
		}
	}
	return hessians;
}
//input Derivative for Models
template<class Model,class Point>
std::vector<RealVector> estimateInputDerivative(Model& net,const Point& point,double epsilon=1.e-10){
	std::size_t outputSize = net(point).size();
	std::vector<RealVector> gradients(point.size(),RealVector(outputSize));
	for(size_t dim=0;dim!=point.size();++dim){
		RealVector testPoint1=point;
		testPoint1(dim)+=epsilon;
		RealVector result1=net(testPoint1);

		RealVector testPoint2=point;
		testPoint2(dim)-=epsilon;
		RealVector result2=net(testPoint2);
		gradients[dim]=(result1-result2)/(2*epsilon);
	}
	return gradients;
}

inline void testDerivative(const std::vector<RealVector>& g1,const std::vector<RealVector>& g2,double epsilon=1.e-5){
	BOOST_REQUIRE_EQUAL(g1.size(),g2.size());
	for(size_t output=0;output!=g1.size();++output){
		BOOST_REQUIRE_EQUAL(g1[output].size(),g2[output].size());
		BOOST_CHECK_SMALL(norm_2(g1[output]-g2[output]),epsilon);
	}
}
//general functions to estimate and test derivatives
template<class Model,class Point>
void testWeightedDerivative(Model& net,const Point& point,const RealVector& coefficients,double epsilon=1.e-5,double estimationEpsilon = 1.e-5){
	RealMatrix pointBatch(1,point.size());
	row(pointBatch,0)=point;
	boost::shared_ptr<State> state = net.createState();
	typename Model::BatchOutputType output; 
	net.eval(pointBatch,output,*state);

	std::vector<RealVector> derivative=estimateDerivative(net,point, estimationEpsilon);

	//check every coefficient independent of the others
	for(std::size_t coeff = 0; coeff!= coefficients.size(); ++coeff){
		RealMatrix coeffBatch(1,coefficients.size());
		coeffBatch.clear();
		coeffBatch(0,coeff)=coefficients(coeff);

		RealVector testGradient;
		net.weightedParameterDerivative(pointBatch,coeffBatch,*state,testGradient);
		//this makes the result again independent of the coefficient
		//provided that the computation is correct
		testGradient/=coefficients(coeff);

		//calculate error between both
		BOOST_REQUIRE_EQUAL(testGradient.size(),derivative.size());
		for(std::size_t i = 0; i != testGradient.size(); ++i){
			double error=sqr(testGradient(i)-derivative[i](coeff));
			BOOST_CHECK_SMALL(error,epsilon);
		}
	}
//	double error=norm_2(testGradient-resultGradient);
//	BOOST_CHECK_SMALL(error,epsilon);
}

template<class Model,class Point>
void testWeightedSecondDerivative(Model& net,const Point& point,const RealVector& coefficients, const RealMatrix& coeffHessian, double epsilon=1.e-5,double estimationEpsilon = 1.e-10){
	boost::shared_ptr<State> state = net.createState();
	typename Model::BatchOutputType output; 
	net.eval(point,output,*state);
	//now calculate the nets weighted gradient
	RealVector testGradient;
	RealMatrix testHessian;
	net.weightedParameterDerivative(point,coefficients,coeffHessian,*state,testGradient,testHessian);

	//estimate hessians and derivatives for every output
	std::vector<RealMatrix> hessians = estimateSecondDerivative(net,point);
	std::vector<RealVector> derivative = estimateDerivative(net,point, estimationEpsilon);

	//calculate testresult
	RealMatrix resultHessian(derivative.size(),derivative.size());
	RealVector resultGradient(derivative.size());
	//this is the weighted gradient calculated the naiive way
	for(size_t i=0;i!=derivative.size();++i)
		resultGradient(i)=inner_prod(derivative[i],coefficients);

	//this is the weighted hessian calculated the naiive way
	for(size_t wi=0;wi!=derivative.size();++wi){
		for(size_t wj=0;wj!=derivative.size();++wj){
			resultHessian(wi,wj) = inner_prod(row(hessians[wi],wj),coefficients);
			for(size_t output=0;output!=coefficients.size();++output){
				for(size_t output2=0;output2!=coefficients.size();++output2){
					resultHessian(wi,wj) += coeffHessian(output,output2)*derivative[wi](output)*derivative[wj](output2);
				}
			}
		}
	}
	//test the gradient
	double error=norm_2(testGradient-resultGradient);
	BOOST_REQUIRE_SMALL(error,epsilon);

	//test hessian
	for(size_t wi=0;wi!=derivative.size();++wi){
		for(size_t wj=0;wj!=derivative.size();++wj){
			BOOST_CHECK_SMALL(resultHessian(wi,wj)-testHessian(wi,wj),epsilon);
		}
	}
}
template<class Model,class Point>
void testWeightedInputDerivative(Model& net,const Point& point,const RealVector& coefficients,double epsilon=1.e-5, double estimationEpsilon=1.e-5){
	//now calculate the nets weighted gradient
	RealMatrix coeffBatch(1,coefficients.size());
	RealMatrix pointBatch(1,point.size());
	row(coeffBatch,0)=coefficients;
	row(pointBatch,0)=point;
	
	boost::shared_ptr<State> state = net.createState();
	typename Model::BatchOutputType output; 
	net.eval(pointBatch,output,*state);
	
	RealMatrix testGradient;
	net.weightedInputDerivative(pointBatch,coeffBatch,*state,testGradient);

	//calculate testresult
	//this is the weighted gradient calculated the naive way
	std::vector<RealVector> derivative=estimateInputDerivative(net,point,estimationEpsilon);
	RealVector resultGradient(derivative.size());
	for(size_t i=0;i!=derivative.size();++i)
		resultGradient(i)=inner_prod(derivative[i],coefficients);

	//calculate error between both
	double error=norm_2(row(testGradient,0)-resultGradient);
	BOOST_CHECK_SMALL(error,epsilon);
}

///convenience function which does automatic sampling of points,parameters and coefficients
///and evaluates and tests the parameter derivative.
///it is assumed that the function has the methods inputSize() and  outputSize()
///samples are taken from the interval -10,10
template<class Model>
void testWeightedDerivative(Model& net,unsigned int numberOfTests = 1000, double epsilon=1.e-5,double estimationEpsilon = 1.e-5) {
	BOOST_CHECK_EQUAL(net.hasFirstParameterDerivative(),true);
	RealVector parameters(net.numberOfParameters());
	RealVector coefficients(net.outputSize());
	RealVector point(net.inputSize());
	for(unsigned int test = 0; test != numberOfTests; ++test){
		for(size_t i = 0; i != net.numberOfParameters();++i){
			parameters(i) = Rng::uni(-5,5);
		}
		for(size_t i = 0; i != net.outputSize();++i){
			coefficients(i) = Rng::uni(-5,5);
		}
		for(size_t i = 0; i != net.inputSize();++i){
			point(i) = Rng::uni(-5,5);
		}

		net.setParameterVector(parameters);
		testWeightedDerivative(net, point, coefficients, epsilon,estimationEpsilon);
	}
}
///convenience function which does automatic sampling of points,parameters and coefficients
///and evaluates and tests the input derivative.
///it is assumed that the function has the methods inputSize() and  outputSize()
///samples are taken from the interval -10,10
template<class Model>
void testWeightedInputDerivative(Model& net,unsigned int numberOfTests = 1000, double epsilon=1.e-5,double estimationEpsilon = 1.e-5) {
	BOOST_CHECK_EQUAL(net.hasFirstInputDerivative(),true);
	RealVector parameters(net.numberOfParameters());
	RealVector coefficients(net.outputSize());
	RealVector point(net.inputSize());
	for(unsigned int test = 0; test != numberOfTests; ++test){
		for(size_t i = 0; i != net.numberOfParameters();++i){
			parameters(i) = Rng::uni(-10,10);
		}
		for(size_t i = 0; i != net.outputSize();++i){
			coefficients(i) = Rng::uni(-10,10);
		}
		for(size_t i = 0; i != net.inputSize();++i){
			point(i) = Rng::uni(-10,10);
		}

		net.setParameterVector(parameters);
		testWeightedInputDerivative(net, point, coefficients, epsilon,estimationEpsilon);
	}
}

namespace detail{
//small helper functions which are used in testEval() to get the error between two samples
double elementEvalError(unsigned int a, unsigned int b){
	return (double) (a > b? a-b: b-a);
}
template<class T, class U>
double elementEvalError(T a, U b){
	return distance(a,b);
}
}

template<class T, class R>
void testBatchEval(AbstractModel<T, R>& model, typename Batch<T>::type const& sampleBatch){
	std::size_t batchSize = size(sampleBatch);

	//evaluate batch of inputs using a state and without stat.
	typename Batch<R>::type resultBatch = model(sampleBatch);
	typename Batch<R>::type resultBatch2;
	boost::shared_ptr<State> state = model.createState();
	model.eval(sampleBatch,resultBatch2,*state);
	
	//sanity check. if we don't get a result for every input something is seriously broken
	BOOST_REQUIRE_EQUAL(size(resultBatch),batchSize);
	BOOST_REQUIRE_EQUAL(size(resultBatch2),batchSize);

	//eval every element of the batch independently and compare the batch result with it
	for(std::size_t i = 0; i != batchSize; ++i){
		R result = model(get(sampleBatch,i));
		double error = detail::elementEvalError(result, get(resultBatch,i));
		double error2 = detail::elementEvalError(result, get(resultBatch2,i));
		BOOST_CHECK_SMALL(error, 1.e-7);
		BOOST_CHECK_SMALL(error2, 1.e-7);
	}
}

}
#endif
