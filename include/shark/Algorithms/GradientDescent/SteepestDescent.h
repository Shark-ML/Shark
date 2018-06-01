//===========================================================================
/*!
 * 
 *
 * \brief       SteepestDescent
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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
#ifndef SHARK_ML_OPTIMIZER_STEEPESTDESCENT_H
#define SHARK_ML_OPTIMIZER_STEEPESTDESCENT_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

namespace shark{

///@brief Standard steepest descent.
/// \ingroup gradientopt
template<class SearchPointType = RealVector>
class SteepestDescent : public AbstractSingleObjectiveOptimizer<SearchPointType>
{
public:
	typedef AbstractObjectiveFunction<SearchPointType,double> ObjectiveFunctionType;
	SteepestDescent() {
		this->m_features |= this->REQUIRES_FIRST_DERIVATIVE;

		m_learningRate = 0.1;
		m_momentum = 0.0;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SteepestDescent"; }

	void init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint) {
		this->checkFeatures(objectiveFunction);
		SHARK_RUNTIME_CHECK(startingPoint.size() == objectiveFunction.numberOfVariables(), "Initial starting point and dimensionality of function do not agree");
		
		m_path.resize(startingPoint.size());
		m_path.clear();
		this->m_best.point = startingPoint;
		this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
	}
	using AbstractSingleObjectiveOptimizer<SearchPointType >::init;

	/*!
	 *  \brief get learning rate
	 */
	double learningRate() const {
		return m_learningRate;
	}

	/*!
	 *  \brief set learning rate
	 */
	void setLearningRate(double learningRate) {
		m_learningRate = learningRate;
	}

	/*!
	 *  \brief get momentum parameter
	 */
	double momentum() const {
		return m_momentum;
	}

	/*!
	 *  \brief set momentum parameter
	 */
	void setMomentum(double momentum) {
		m_momentum = momentum;
	}
	/*!
	 *  \brief updates searchdirection and then does simple gradient descent
	 */
	void step(ObjectiveFunctionType const& objectiveFunction) {
		m_path = -m_learningRate * m_derivative + m_momentum * m_path;
		this->m_best.point+=m_path;
		this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
	}
	virtual void read( InArchive & archive )
	{
		archive>>m_path;
		archive>>m_learningRate;
		archive>>m_momentum;
	}

	virtual void write( OutArchive & archive ) const
	{
		archive<<m_path;
		archive<<m_learningRate;
		archive<<m_momentum;
	}

private:
	SearchPointType m_path;
	SearchPointType m_derivative;
	double m_learningRate;
	double m_momentum;
};

}
#endif

