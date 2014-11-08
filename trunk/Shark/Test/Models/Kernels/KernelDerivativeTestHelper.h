#ifndef SHARK_TEST_KERNELDERIVATIVETESTHELPER_H
#define SHARK_TEST_KERNELDERIVATIVETESTHELPER_H

#include <vector>
#include <shark/LinAlg/Base.h>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include  <shark/Rng/GlobalRng.h>
#include  <shark/Models/Kernels/AbstractKernelFunction.h>
namespace shark{

template<class Kernel,class Point>
RealVector estimateDerivative(Kernel& kernel, const Point& point1, const Point& point2, double epsilon = 1.e-10) {
	RealVector parameters = kernel.parameterVector();
	RealVector gradient = parameters;
	for (size_t parameter=0; parameter != parameters.size(); ++parameter) {
		RealVector testPoint1 = parameters;
		testPoint1(parameter) += epsilon;
		kernel.setParameterVector(testPoint1);
		double result1 = kernel.eval(point1, point2);

		RealVector testPoint2 = parameters;
		testPoint2(parameter) -= epsilon;
		kernel.setParameterVector(testPoint2);
		double result2 = kernel.eval(point1, point2);

		double estimatedDerivative = (result1 - result2) / (2 * epsilon);

		gradient(parameter) = estimatedDerivative;
	}
	kernel.setParameterVector(parameters);
	return gradient;
}

template<class Kernel,class Point>
RealVector estimateInputDerivative(Kernel& kernel,const Point& point1,const Point& point2,double epsilon=1.e-10){
	size_t size = point1.size();
	RealVector gradient(size);
	for(size_t dim=0;dim!=size;++dim){
		Point testPoint1=point1;
		testPoint1(dim)+=epsilon;
		double result1=kernel(testPoint1,point2);

		Point testPoint2=point1;
		testPoint2(dim)-=epsilon;
		double result2=kernel(testPoint2,point2);

		double estimatedDerivative=(result1-result2)/(2*epsilon);
		gradient(dim)=estimatedDerivative;
	}
	return gradient;
}

/// Convenience function for automatic sampling of both points
/// of a kernel and evaluatation and test of the parameter derivative.
/// Samples are taken from the interval [-5,5] (for each component).
///it is assumed, that the input points are vectors and batches are matrices
template<class T>
void testKernelDerivative(AbstractKernelFunction<T>& kernel,std::size_t inputSize, double epsilon=1.e-8, double testEpsilon= 1.e-6, unsigned int numberOfTests = 5, std::size_t batchSize = 20) {
	BOOST_REQUIRE_EQUAL(kernel.hasFirstParameterDerivative(),true);
	for(unsigned int test = 0; test != numberOfTests; ++test) {
		//create data
		typename Batch<T>::type batch1(batchSize,inputSize);
		typename Batch<T>::type batch2(2*batchSize,inputSize);
		for(std::size_t  i = 0; i != batchSize; ++i){
			for(std::size_t j = 0; j != inputSize;++j) {
				batch1(i,j) = Rng::uni(-3,3);
				batch2(i,j) = Rng::uni(-3,3);
				batch2(i+batchSize,j) = Rng::uni(-3,3);
			}
		}
		
		//evaluate batched derivative
		boost::shared_ptr<State> state = kernel.createState();
		RealMatrix kernelBatchRes;
		kernel.eval(batch1,batch2,kernelBatchRes,*state);
		
		//check that single derivatives fit the estimated derivative
		//also calculate the batxh derivative but set only 1 weight to 1.
		for(std::size_t  i = 0; i != batchSize; ++i){
			T x1=row(batch1,i);
			for(std::size_t j = 0; j != 2*batchSize;++j) {
				T x2=row(batch2,j);
				
				//calculate batch derivative but set only the current derivative to one.
				RealMatrix singleCoeff(batchSize,2*batchSize);
				singleCoeff.clear();
				singleCoeff(i,j) = 1.0;
				RealVector singleBatchGradient;
				kernel.weightedParameterDerivative(batch1,batch2,singleCoeff,*state,singleBatchGradient);
				
				//compare with estimated derivative
				RealVector estimatedDerivative = estimateDerivative(kernel, x1, x2, epsilon);
				BOOST_REQUIRE_EQUAL(estimatedDerivative.size(), singleBatchGradient.size());
				BOOST_CHECK_SMALL(norm_2(estimatedDerivative - singleBatchGradient), testEpsilon);
				for(std::size_t k = 0; k != estimatedDerivative.size();++k){
					if(singleBatchGradient(k)-estimatedDerivative(k)> epsilon)
						BOOST_CHECK_CLOSE(singleBatchGradient(k),estimatedDerivative(k), 0.01);
				}
				//std::cout<<singleBatchGradient<<std::endl;
			}
		}
	}
}

/// Convenience function for automatic sampling of both points
/// of a kernel and evaluatation and test of the input derivative.
/// Samples are taken from the interval [-5,5] (for each component).
template<class T>
void testKernelInputDerivative(AbstractKernelFunction<T>& kernel,std::size_t inputSize, double epsilon=1.e-8, double testEpsilon= 1.e-6, unsigned int numberOfTests = 5, std::size_t batchSize = 20) {
	BOOST_REQUIRE_EQUAL(kernel.hasFirstInputDerivative(),true);
	for(unsigned int test = 0; test != numberOfTests; ++test) {
		//create data
		typename Batch<T>::type batch1(batchSize,inputSize);
		typename Batch<T>::type batch2(batchSize+1,inputSize);
		for(std::size_t  i = 0; i != batchSize; ++i){
			for(std::size_t j = 0; j != inputSize;++j) {
				batch1(i,j) = Rng::uni(-3,3);
				batch2(i,j) = Rng::uni(-3,3);
			}
		}
		for(std::size_t j = 0; j != inputSize;++j) {
			batch2(batchSize,j) = Rng::uni(-3,3);
		}
		
		
		//evaluate batched derivative
		boost::shared_ptr<State> state = kernel.createState();
		RealMatrix kernelBatchRes;
		kernel.eval(batch1,batch2,kernelBatchRes,*state);
		
		//check that single derivatives fit the estimated derivative
		RealMatrix singleCoeff(batchSize,batchSize+1,0.0);
		for(std::size_t  j = 0; j != batchSize+1; ++j){
			T x2=row(batch2,j);
			
			for(std::size_t  i = 0; i != batchSize; ++i){
				//calculate batch derivative but set only the coefficient of the i-th sample in Batch2 to one
				singleCoeff(i,j) = 1.0;
				
				//calculate the input derivative for all elements of Batch1
				typename Batch<T>::type derivativeBatch;
				kernel.weightedInputDerivative(batch1,batch2,singleCoeff,*state,derivativeBatch);
				singleCoeff(i,j) = 0;
				//check whether the previously calculated
				//is the same as the estimated input derivative
				T x1=row(batch1,i);
				
				//estimate the derivative
				RealVector estimatedDerivative = estimateInputDerivative(kernel, x1, x2, epsilon);
				BOOST_CHECK_SMALL(norm_2(estimatedDerivative - row(derivativeBatch,i)), testEpsilon);
			}
		}
	}
}


template<class T>
void testEval(AbstractKernelFunction<T>& kernel, typename Batch<T>::type const& sampleBatch1,typename Batch<T>::type const& sampleBatch2){
	std::size_t batchSize1 = size(sampleBatch1);
	std::size_t batchSize2 = size(sampleBatch2);
	
	//evaluate batch on the kernels
	boost::shared_ptr<State> state = kernel.createState();
	RealMatrix kernelResultsIntermediate;
	RealMatrix kernelResults = kernel(sampleBatch1,sampleBatch2);
	kernel.eval(sampleBatch1,sampleBatch2,kernelResultsIntermediate,*state);
	
	//eval every single combination and compare with kernel results, 
	//also check that the single eval version returns the same results
	BOOST_REQUIRE_EQUAL(kernelResults.size1(),batchSize1);
	BOOST_REQUIRE_EQUAL(kernelResults.size2(),batchSize2);
	BOOST_REQUIRE_EQUAL(kernelResultsIntermediate.size1(),batchSize1);
	BOOST_REQUIRE_EQUAL(kernelResultsIntermediate.size2(),batchSize2);
	for(std::size_t i = 0; i != batchSize1; ++i){
		T x1 = get(sampleBatch1,i);
		for(std::size_t j = 0; j != batchSize2; ++j){
			double result = kernel.eval(x1,get(sampleBatch2,j));

			BOOST_CHECK_SMALL(result-kernelResults(i,j), 1.e-13);
			BOOST_CHECK_SMALL(result-kernelResultsIntermediate(i,j), 1.e-13);
		}
	}
}

}
#endif
