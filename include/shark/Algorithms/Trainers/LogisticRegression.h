//===========================================================================
/*!
 *
 *
 * \brief       Logistic Regression
 *
 *
 *
 * \author      O.Krause
 * \date        2017
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


#ifndef SHARK_ALGORITHMS_TRAINERS_LOGISTICREGRESSION_H
#define SHARK_ALGORITHMS_TRAINERS_LOGISTICREGRESSION_H

#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractWeightedTrainer.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <cmath>


namespace shark {

/// \brief Trainer for Logistic regression
///
/// Logistic regression solves the following optimization problem:
/// \f[ \min_{w,b} \sum_i u_i l(y_i,f(x_i^Tw+b)) +\lambda_1 |w|_1 +\lambda_2 |w|^2_2 \f]
/// Where \f$l\f$ is the cross-entropy loss and \f$u_i\f$ are individual weuights for each point(assumed to be 1).
/// Logistic regression is one of the most well known
/// machine learning algorithms for classification using linear models.
///
/// The solver is based on LBFGS for the case where no l1-regularization is used. Otherwise
/// the problem is transformed into a constrained problem and the constrined-LBFGS algorithm
/// is used. This is one of the most efficient solvers for logistic regression as long as the
/// number of data points is not too large.
template <class InputVectorType = RealVector>
class LogisticRegression : public AbstractWeightedTrainer<LinearClassifier<InputVectorType> >, public IParameterizable<>
{
private:
	typedef AbstractWeightedTrainer<LinearClassifier<InputVectorType> > base_type;
public:
	typedef typename base_type::ModelType ModelType;
	typedef typename base_type::DatasetType DatasetType;
	typedef typename base_type::WeightedDatasetType WeightedDatasetType;

	/// \brief Constructor.
	///
	/// \param  lambda1    value of the 1-norm regularization parameter (see class description)
	/// \param  lambda2    value of the 2-norm regularization parameter (see class description)
	/// \param  lbias          whether to train with bias or not
	/// \param  accuracy  stopping criterion for the iterative solver, maximal gradient component of the objective function (see class description)
	LogisticRegression(double lambda1 = 0, double lambda2 = 0, bool bias = true, double accuracy = 1.e-8)
	: m_bias(bias){
		setLambda1(lambda1);
		setLambda2(lambda2);
		setAccuracy(accuracy);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LogisticRegression"; }


	/// \brief Return the current setting of the l1-regularization parameter.
	double lambda1() const{
		return m_lambda1;
	}
	
	/// \brief Return the current setting of the l2-regularization parameter.
	double lambda2() const{
		return m_lambda2;
	}

	/// \brief Set the l1-regularization parameter.
	void setLambda1(double lambda){
		SHARK_RUNTIME_CHECK(lambda >= 0.0, "Lambda1 must be positive");
		m_lambda1 = lambda;
	}

	/// \brief Set the l2-regularization parameter.
	void setLambda2(double lambda){
		SHARK_RUNTIME_CHECK(lambda >= 0.0, "Lambda2 must be positive");
		m_lambda2 = lambda;
	}
	/// \brief Return the current setting of the accuracy (maximal gradient component of the optimization problem).
	double accuracy() const{
		return m_accuracy;
	}

	/// \brief Set the accuracy (maximal gradient component of the optimization problem).
	void setAccuracy(double accuracy){
		SHARK_RUNTIME_CHECK(accuracy > 0.0, "Accuracy must be positive");
		m_accuracy = accuracy;
	}

	/// \brief Get the regularization parameters lambda1 and lambda2 through the IParameterizable interface.
	RealVector parameterVector() const{
		return {m_lambda1,m_lambda2};
	}

	/// \brief Set the regularization parameters lambda1 and lambda2 through the IParameterizable interface.
	void setParameterVector(const RealVector& param){
		SIZE_CHECK(param.size() == 1);
		setLambda1(param(0));
		setLambda2(param(1));
	}

	/// \brief Return the number of parameters (one in this case).
	size_t numberOfParameters() const{
		return 2;
	}

	/// \brief Train a linear model with logistic regression.
	void train(ModelType& model, DatasetType const& dataset){
		optimize(model, dataset);
	}
	
	/// \brief Train a linear model with logistic regression using weights.
	void train(ModelType& model, WeightedDatasetType const& dataset){
		optimize(model, dataset);
	}
	
private:
	
	template<class DatasetT>
	void optimize(ModelType& model, DatasetT const& dataset){
		//initialize model
		std::size_t numOutputs = numberOfClasses(dataset);
		if(numOutputs == 2) numOutputs = 1;
		auto& innerModel = model.decisionFunction();
		innerModel.setStructure(inputDimension(dataset),numOutputs, m_bias);
		std::size_t dim = innerModel.numberOfParameters();
		innerModel.setParameterVector(RealVector(dim,0.0));
		
		//setup error function
		CrossEntropy loss;
		ErrorFunction error(dataset, &innerModel, &loss);//note: chooses a different implementation depending on the dataset type
		
		//handle two-norm regularization
		TwoNormRegularizer regularizer;
		if(m_lambda2 > 0.0){
			//set mask to skip bias weights
			if(m_bias){
				RealVector mask(dim,1.0);
				subrange(mask,dim - numOutputs, dim).clear();
				regularizer.setMask(mask);
			}
			error.setRegularizer(m_lambda2, &regularizer);
		}
		
		//no l1-regularization needed -> simple case
		if(m_lambda1 == 0){
			LBFGS optimizer;
			error.init();
			optimizer.init(error);
			RealVector lastPoint = optimizer.solution().point;
			while(norm_inf(optimizer.derivative()) > m_accuracy){
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
		L1Reformulation function(&error, m_lambda1, dim - m_bias * numOutputs);
		LBFGS optimizer;
		function.init();
		optimizer.init(function);
		RealVector lastPoint = optimizer.solution().point;
		for(;;){
			//check whether we are done
			bool optimal= true;
			auto const& derivative = optimizer.derivative();
			for(std::size_t i = 0; i != lastPoint.size(); ++i){
				if(lastPoint(i) < 1.e-13 && -derivative(i) > m_accuracy){//coordinate on constraint and derivative pushes away from constraint
					optimal = false;
					break;
				}else if(lastPoint(i) > 1.e-13 && std::abs(derivative(i)) > m_accuracy){//free coordinate and derivative is not close to 0
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
		
		std::size_t n = dim - m_bias * numOutputs;
		//construct parameter vector from solution
		RealVector param = (subrange(lastPoint,0,n) - subrange(lastPoint,n, 2 * n)) | subrange(lastPoint,2*n,lastPoint.size());
		model.setParameterVector(param);
	}

private:
	bool m_bias; ///< whether to train with the bias parameter or not
	double m_lambda1;             ///< l1-regularization parameter
	double m_lambda2;             ///< l2-regularization parameter
	double m_accuracy;           ///< gradient accuracy

	class L1Reformulation: public SingleObjectiveFunction{
	public:
		L1Reformulation(ErrorFunction* error, double lambda1, std::size_t regularizedParams)
		: mep_error(error), m_lambda1(lambda1), m_regularizedParams(regularizedParams){
			m_features |= CAN_PROPOSE_STARTING_POINT;
			m_features |= HAS_FIRST_DERIVATIVE;
			
			std::size_t dim = numberOfVariables();
			double unconstrained =  1e100;
			RealVector lower(dim,0.0);
			subrange(lower, 2 * m_regularizedParams,dim) = blas::repeat(-unconstrained,dim - 2 * m_regularizedParams);
			RealVector upper(dim,unconstrained);
			m_handler.setBounds(lower,upper);
			announceConstraintHandler(&m_handler);
		}
		
		SearchPointType proposeStartingPoint()const {
			return RealVector(numberOfVariables(),0.0);
		}
		
		std::size_t numberOfVariables()const{
			return mep_error->numberOfVariables() + m_regularizedParams;
		}
		
		double eval(RealVector const& input) const{
			std::size_t dim = input.size();
			std::size_t n = m_regularizedParams;
			RealVector params = (subrange(input,0,n) - subrange(input,n, 2 * n)) | subrange(input,2*n,dim);
			return mep_error->eval(params) + m_lambda1 * sum(subrange(input,0,2*n));
		}
		ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const{
			std::size_t dim = input.size();
			std::size_t n = m_regularizedParams;
			RealVector params = (subrange(input,0,n) - subrange(input,n, 2 * n)) | subrange(input,2*n,dim);
			FirstOrderDerivative paramDerivative;
			double error = mep_error->evalDerivative(params, paramDerivative);
			derivative.resize(numberOfVariables());
			noalias(subrange(derivative,0,n)) = m_lambda1 + subrange(paramDerivative,0,n);
			noalias(subrange(derivative,n,2 * n)) = m_lambda1 - subrange(paramDerivative,0,n);
			noalias(subrange(derivative,2 * n,dim)) = subrange(paramDerivative,n,dim - n);
			return error + m_lambda1 * sum(subrange(input,0,2*n));
		}
		
	private:
		ErrorFunction* mep_error;
		double m_lambda1;
		BoxConstraintHandler<RealVector> m_handler;
		std::size_t m_regularizedParams;
	};
};


}
#endif
