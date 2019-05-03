//===========================================================================
/*!
 * 
 *
 * \brief       AdaDelta
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
#ifndef SHARK_ML_OPTIMIZER_ADADELTA_H
#define SHARK_ML_OPTIMIZER_ADADELTA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

namespace shark{

/// \brief ADADELTA: An Adaptive Learning-Rate Method
///
/// As implemented in Zeiler, M. D. (2012). ADADELTA: an adaptive learning rate method. https://arxiv.org/abs/1212.5701
/// \ingroup gradientopt
template<class SearchPointType = RealVector>
class AdaDelta : public AbstractSingleObjectiveOptimizer<SearchPointType >{
private:
	typedef typename SearchPointType::value_type scalar_type;
public:
	typedef AbstractObjectiveFunction<SearchPointType,double> ObjectiveFunctionType;
	AdaDelta() {
		this->m_features |= this->REQUIRES_FIRST_DERIVATIVE;

		m_rho = scalar_type(0.95);
		m_epsilon = scalar_type(1.e-6);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "AdaDelta"; }

	void init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint) {
		this-> checkFeatures(objectiveFunction);
		SHARK_RUNTIME_CHECK(startingPoint.size() == objectiveFunction.numberOfVariables(), "Initial starting point and dimensionality of function do not agree");
		
		//initialize long term averages
		m_x2 = SearchPointType(startingPoint.size(),0.0);
		m_g2 = SearchPointType(startingPoint.size(),0.0);
		
		//set point to the current starting point
		this->m_best.point = startingPoint;
		this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
	}
	using AbstractSingleObjectiveOptimizer<SearchPointType >::init;
	
	/// \brief get gradient averaging parameter m_rho
	scalar_type rho() const{
		return m_rho;
	}

	/// \brief set gradient averaging parameter m_rho
	void setBeta2(scalar_type rho) {
		SHARK_RUNTIME_CHECK(rho > 0, "rho must be positive.");
		SHARK_RUNTIME_CHECK(rho < 1, "rho must be smaller than 1.");
		m_rho = rho;
	}
	
	/// \brief get minimum noise estimate epsilon
	scalar_type epsilon() const {
		return m_epsilon;
	}

	/// \brief set minimum noise estimate epsilon
	void setEpsilon(scalar_type epsilon) {
		SHARK_RUNTIME_CHECK(epsilon > 0, "epsilon must be positive.");
		m_epsilon = epsilon;
	}
	
	/// \brief Performs a step of the optimization.
	///
	/// First the current guess for gradient and its second moment are updated using
	/// \f[ E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) \frac{\partial}{\partial x} f(x_{t-1})^2\f]
	/// \f[ \delta x_t= - \frac{\sqrt{E[\delta x^2]_{t-1} + \epsilon }}{\sqrt{E[g^2]_t + \epsilon}} \frac{\partial}{\partial x} f(x_{t-1}) \f]
	/// \f[ E[\delta x^2]_t = \rho E[\delta x^2]_{t-1} + (1-\rho)  \delta x_t^2\f]
	///
	/// The step is then performed as
	/// \f[ x_{t} = x_{t-1} + \delta x_t \f]
	/// where a slight step correction is used to remove the bias in the first few iterations where the means are close to 0.
	void step(ObjectiveFunctionType const& objectiveFunction) {
		noalias(m_g2) = m_rho * m_g2 + (1-m_rho) * sqr(m_derivative);
		auto deltaX = - sqrt(m_x2 + m_epsilon) / sqrt(m_g2 + m_epsilon) * m_derivative;
		noalias(m_x2) = m_rho * m_x2 + (1-m_rho) * sqr(deltaX);
		
		noalias(this->m_best.point) += deltaX;
		this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
	}
	virtual void read( InArchive & archive ){
		archive>>m_x2;
		archive>>m_g2;
		archive>>m_derivative;
		archive>>this->m_best;
		
		archive>>m_rho;
		archive>>m_epsilon;
	}

	virtual void write( OutArchive & archive ) const
	{
		archive<<m_x2;
		archive<<m_g2;
		archive<<m_derivative;
		archive<<this->m_best;
		
		archive<<m_rho;
		archive<<m_epsilon;
	}

private:
	SearchPointType m_x2;
	SearchPointType m_g2;
	SearchPointType m_derivative;
	
	scalar_type m_rho;
	scalar_type m_epsilon;
};

}
#endif

