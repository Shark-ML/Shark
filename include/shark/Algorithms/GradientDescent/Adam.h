//===========================================================================
/*!
 * 
 *
 * \brief       Adam
 * 
 * 
 *
 * \author      O. Krause
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
#ifndef SHARK_ML_OPTIMIZER_ADAM_H
#define SHARK_ML_OPTIMIZER_ADAM_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

namespace shark{

/// \brief Adaptive Moment Estimation Algorithm (ADAM)
///
/// Performs SGD by using a long term average of the gradient as well as its second moment to adapt
/// a step size for each coordinate.
/// \ingroup gradientopt
template<class SearchPointType = RealVector>
class Adam : public AbstractSingleObjectiveOptimizer<SearchPointType >
{
public:
	typedef AbstractObjectiveFunction<SearchPointType,double> ObjectiveFunctionType;
	Adam() {
		this->m_features |= this->REQUIRES_FIRST_DERIVATIVE;

		m_beta1 = 0.9;
		m_beta2 = 0.999;
		m_epsilon = 1.e-8;
		m_eta = 0.001;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Adam"; }

	void init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint) {
		this-> checkFeatures(objectiveFunction);
		SHARK_RUNTIME_CHECK(startingPoint.size() == objectiveFunction.numberOfVariables(), "Initial starting point and dimensionality of function do not agree");
		
		//initialize long term averages
		m_avgGrad = SearchPointType(startingPoint.size(),0.0);
		m_secondMoment = SearchPointType(startingPoint.size(),0.0);
		m_counter = 0;
		
		//set point to the current starting point
		this->m_best.point = startingPoint;
		this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
	}
	using AbstractSingleObjectiveOptimizer<SearchPointType >::init;

	/// \brief get learning rate eta
	double eta() const {
		return m_eta;
	}

	/// \brief set learning rate eta
	void setEta(double eta) {
		SHARK_RUNTIME_CHECK(eta > 0, "eta must be positive.");
		m_eta = eta;
	}
	
	/// \brief get gradient averaging parameter beta1
	double beta1() const {
		return m_beta1;
	}

	/// \brief set gradient averaging parameter beta1
	void setBeta1(double beta1) {
		SHARK_RUNTIME_CHECK(beta1 > 0, "beta1 must be positive.");
		m_beta1 = beta1;
	}
	
	/// \brief get gradient averaging parameter beta2
	double beta2() const {
		return m_beta2;
	}

	/// \brief set gradient averaging parameter beta2
	void setBeta2(double beta2) {
		SHARK_RUNTIME_CHECK(beta2 > 0, "beta2 must be positive.");
		m_beta2 = beta2;
	}
	
	/// \brief get minimum noise estimate epsilon
	double epsilon() const {
		return m_epsilon;
	}

	/// \brief set minimum noise estimate epsilon
	void setEpsilon(double epsilon) {
		SHARK_RUNTIME_CHECK(epsilon > 0, "epsilon must be positive.");
		m_epsilon = epsilon;
	}
	/// \brief Performs a step of the optimization.
	///
	/// First the current guess for gradient and its second moment are updated using
	/// \f[ g_t = \beta_1 g_{t-1} + (1-\beta1) \frac{\partial}{\partial x} f(x_{t-1})\f]
	/// \f[ v_t = \beta_2 v_{t-1} + (1-\beta2) (\frac{\partial}{\partial x} f(x_{t-1}))^2\f]
	///
	/// The step is then performed as
	/// \f[ x_{t} = x_{t-1} - \eta * g_t *(sqrt(v_t) + \epsilon)^{-1} \f]
	/// where a slight step correction is used to remove the bias in the first few iterations where the means are close to 0.
	void step(ObjectiveFunctionType const& objectiveFunction) {
		//update long term averages of the gradient and its variance
		noalias(m_avgGrad) = m_beta1 * m_avgGrad + (1-m_beta1) * m_derivative;
		noalias(m_secondMoment) = m_beta2 * m_secondMoment + (1-m_beta2)* sqr(m_derivative);
		//for the first few iterations, we need bias correction
		++m_counter;
		double bias1 = 1-std::pow(m_beta1,m_counter);
		double bias2 = 1-std::pow(m_beta2,m_counter);
		
		noalias(this->m_best.point) -= (m_eta/bias1) * m_avgGrad/(m_epsilon + sqrt(m_secondMoment/bias2));
		this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
	}
	virtual void read( InArchive & archive ){
		archive>>m_avgGrad;
		archive>>m_secondMoment;
		archive>>m_counter;
		archive>>m_derivative;
		archive>>this->m_best;
		
		archive>>m_beta1;
		archive>>m_beta2;
		archive>>m_epsilon;
		archive>>m_eta;
	}

	virtual void write( OutArchive & archive ) const
	{
		archive<<m_avgGrad;
		archive<<m_secondMoment;
		archive<<m_counter;
		archive<<m_derivative;
		archive<<this->m_best;
		
		archive<<m_beta1;
		archive<<m_beta2;
		archive<<m_epsilon;
		archive<<m_eta;
	}

private:
	SearchPointType m_avgGrad;
	SearchPointType m_secondMoment;
	unsigned int m_counter;
	SearchPointType m_derivative;
	
	double m_beta1;
	double m_beta2;
	double m_epsilon;
	double m_eta;
};

}
#endif

