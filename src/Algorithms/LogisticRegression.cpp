//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of Logistic- Regression
 * 
 * 
 *
 * \author      O.Krause, T. Glasmachers
 * \date        2010-2018
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#define SHARK_COMPILE_DLL
#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <cmath>

namespace shark{
template<class SearchPointType>
class L1Reformulation: public AbstractObjectiveFunction<SearchPointType, double>{
public:
	L1Reformulation(ErrorFunction<SearchPointType>* error, double lambda1, std::size_t regularizedParams)
	: mep_error(error), m_lambda1(lambda1), m_regularizedParams(regularizedParams){
		this->m_features |= this->CAN_PROPOSE_STARTING_POINT;
		this->m_features |= this->HAS_FIRST_DERIVATIVE;
		
		std::size_t dim = numberOfVariables();
		double unconstrained =  1e100;
		SearchPointType lower(dim,0.0);
		subrange(lower, 2 * m_regularizedParams,dim) = blas::repeat(-unconstrained,dim - 2 * m_regularizedParams);
		SearchPointType upper(dim,unconstrained);
		m_handler.setBounds(lower,upper);
		this->announceConstraintHandler(&m_handler);
	}
	
	SearchPointType proposeStartingPoint()const {
		return SearchPointType(numberOfVariables(),0.0);
	}
	
	std::size_t numberOfVariables()const{
		return mep_error->numberOfVariables() + m_regularizedParams;
	}
	
	double eval(SearchPointType const& input) const{
		std::size_t dim = input.size();
		std::size_t n = m_regularizedParams;
		SearchPointType params = (subrange(input,0,n) - subrange(input,n, 2 * n)) | subrange(input,2*n,dim);
		return mep_error->eval(params) + m_lambda1 * sum(subrange(input,0,2*n));
	}
	double evalDerivative( SearchPointType const& input, SearchPointType & derivative ) const{
		std::size_t dim = input.size();
		std::size_t n = m_regularizedParams;
		SearchPointType params = (subrange(input,0,n) - subrange(input,n, 2 * n)) | subrange(input,2*n,dim);
		SearchPointType paramDerivative;
		double error = mep_error->evalDerivative(params, paramDerivative);
		derivative.resize(numberOfVariables());
		noalias(subrange(derivative,0,n)) = m_lambda1 + subrange(paramDerivative,0,n);
		noalias(subrange(derivative,n,2 * n)) = m_lambda1 - subrange(paramDerivative,0,n);
		noalias(subrange(derivative,2 * n,dim)) = subrange(paramDerivative,n,dim - n);
		return error + m_lambda1 * sum(subrange(input,0,2*n));
	}
	
private:
	ErrorFunction<SearchPointType>* mep_error;
	double m_lambda1;
	BoxConstraintHandler<SearchPointType> m_handler;
	std::size_t m_regularizedParams;
};

template<class ModelType, class DatasetT>
void logisticRegressionOptimize(ModelType& model, DatasetT const& dataset, double lambda1, double lambda2, double accuracy, bool useBias){
	typedef typename ModelType::ParameterVectorType SearchPointType;
	//initialize model
	std::size_t numOutputs = numberOfClasses(dataset);
	if(numOutputs == 2) numOutputs = 1;
	auto& innerModel = model.decisionFunction();
	innerModel.setStructure(inputDimension(dataset),numOutputs, useBias);
	std::size_t dim = innerModel.numberOfParameters();
	innerModel.setParameterVector(SearchPointType(dim,0.0));
	
	//setup error function
	CrossEntropy<unsigned int, typename ModelType::DecisionFunctionType::OutputType> loss;
	ErrorFunction<SearchPointType> error(dataset, &innerModel, &loss);//note: chooses a different implementation depending on the dataset type
	
	//handle two-norm regularization
	TwoNormRegularizer<SearchPointType> regularizer;
	if(lambda2 > 0.0){
		//set mask to skip bias weights
		if(useBias){
			SearchPointType mask(dim,1.0);
			subrange(mask,dim - numOutputs, dim).clear();
			regularizer.setMask(mask);
		}
		error.setRegularizer(lambda2, &regularizer);
	}
	
	//no l1-regularization needed -> simple case
	if(lambda1 == 0){
		LBFGS<SearchPointType> optimizer;
		error.init();
		optimizer.init(error);
		SearchPointType lastPoint = optimizer.solution().point;
		while(norm_inf(optimizer.derivative()) > accuracy){
			optimizer.step(error);
			//if no progress has been made, something is wrong or we have numerical problems
			//=> abort.
			if(norm_sqr(optimizer.solution().point - lastPoint) == 0) break;
			noalias(lastPoint) = optimizer.solution().point;
		}
		model.setParameterVector(lastPoint);
		return;
	}
	
	//l1-regularization is more painful.
	//we transform the l1-regularization |w|
	// by adding two sets of parameters, w=u-v , u >= 0, v>=0 and |w| = 1^Tu +1^Tv
	// the resulting function is differentiable, however we have added constraints
	L1Reformulation<SearchPointType> function(&error, lambda1, dim - useBias * numOutputs);
	LBFGS<SearchPointType> optimizer;
	function.init();
	optimizer.init(function);
	SearchPointType lastPoint = optimizer.solution().point;
	for(;;){
		//check whether we are done
		bool optimal= true;
		auto const& derivative = optimizer.derivative();
		for(std::size_t i = 0; i != lastPoint.size(); ++i){
			if(lastPoint(i) < 1.e-13 && -derivative(i) > accuracy){//coordinate on constraint and derivative pushes away from constraint
				optimal = false;
				break;
			}else if(lastPoint(i) > 1.e-13 && std::abs(derivative(i)) > accuracy){//free coordinate and derivative is not close to 0
				optimal = false;
				break;
			}
		}
		if(optimal)
			break;
		
		optimizer.step(function);
		//if no progress has been made, something is wrong or we have numerical problems
		//=> abort.
		if(norm_sqr(optimizer.solution().point - lastPoint) == 0) break;
		noalias(lastPoint) = optimizer.solution().point;
	}
	
	std::size_t n = dim - useBias * numOutputs;
	//construct parameter vector from solution
	SearchPointType param = (subrange(lastPoint,0,n) - subrange(lastPoint,n, 2 * n)) | subrange(lastPoint,2*n,lastPoint.size());
	model.setParameterVector(param);
}

template<class VectorType>
void LogisticRegression<VectorType>::train(ModelType& model, DatasetType const& dataset){
	logisticRegressionOptimize(model, dataset, m_lambda1, m_lambda2, m_accuracy, m_bias);
}

template<class VectorType>
void LogisticRegression<VectorType>::train(ModelType& model, WeightedDatasetType const& dataset){
	logisticRegressionOptimize(model, dataset, m_lambda1, m_lambda2, m_accuracy, m_bias);
}


//explicit instantiation of templates
template class SHARK_EXPORT_SYMBOL LogisticRegression<RealVector>;
template class SHARK_EXPORT_SYMBOL LogisticRegression<FloatVector>;
template class SHARK_EXPORT_SYMBOL LogisticRegression<CompressedRealVector>;
template class SHARK_EXPORT_SYMBOL LogisticRegression<CompressedFloatVector>;
}
