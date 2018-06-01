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
/// \ingroup supervised_trainer
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
	void setParameterVector(RealVector const& param){
		SIZE_CHECK(param.size() == 2);
		setLambda1(param(0));
		setLambda2(param(1));
	}

	/// \brief Return the number of parameters (one in this case).
	size_t numberOfParameters() const{
		return 2;
	}

	/// \brief Train a linear model with logistic regression.
	void train(ModelType& model, DatasetType const& dataset);
	
	/// \brief Train a linear model with logistic regression using weights.
	void train(ModelType& model, WeightedDatasetType const& dataset);
private:
	bool m_bias; ///< whether to train with the bias parameter or not
	double m_lambda1;             ///< l1-regularization parameter
	double m_lambda2;             ///< l2-regularization parameter
	double m_accuracy;           ///< gradient accuracy
};

//reference to explicit external template instantiation
extern template class LogisticRegression<RealVector>;
extern template class LogisticRegression<FloatVector>;
extern template class LogisticRegression<CompressedRealVector>;
extern template class LogisticRegression<CompressedFloatVector>;

}
#endif
