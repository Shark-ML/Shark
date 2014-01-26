/*!
 * 
 *
 * \brief       Quickprop
 * 
 * 
 *
 * \author      O. Krause
 * \date        2013
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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
#include <shark/Algorithms/GradientDescent/Quickprop.h>
 #include <boost/math/special_functions/sign.hpp>
 
 using namespace shark;

Quickprop::Quickprop()
{
	m_learningRate = 1.5;
	m_maxIncrease = 1.75;
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
}
void Quickprop::configure( const PropertyTree & node ) {
	m_learningRate=node.get("learningRate",1.5);
	m_maxIncrease=node.get("maxIncrease",1.75);
}

void Quickprop::read( InArchive & archive )
{
	archive>>m_deltaw;
	archive>>m_oldDerivative;
	archive>>m_learningRate;
	archive>>m_maxIncrease;
	archive>>m_parameterSize;
	archive>>m_best.point;
	archive>>m_best.value;
}

void Quickprop::write( OutArchive & archive ) const
{
	archive<<m_deltaw;
	archive<<m_oldDerivative;
	archive<<m_learningRate;
	archive<<m_maxIncrease;
	archive<<m_parameterSize;
	archive<<m_best.point;
	archive<<m_best.value;
}

void Quickprop::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	checkFeatures(objectiveFunction);

	m_parameterSize = startingPoint.size();
	m_deltaw.resize(m_parameterSize);
	m_oldDerivative.resize(m_parameterSize);

	m_oldDerivative.clear();
	m_deltaw.clear();
	m_best.point=startingPoint;
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}

void Quickprop::step(const ObjectiveFunctionType& objectiveFunction) {
	for (size_t i = 0; i < m_parameterSize; i++)
	{
		// avoid division by zero
		if (m_oldDerivative(i) == m_derivative(i)) break;

		// quadratic part
		double delta = m_derivative(i) / (m_oldDerivative(i) - m_derivative(i)) * m_deltaw(i);

		// add gradient term "unless the current slope is opposite in sign from
		// the previous slope"
		// initial step is steepest descent
		if (m_derivative(i) * m_oldDerivative(i) >= 0)
		{
			delta -= m_learningRate * m_derivative(i);
		}

		// limit growth factor if slopes have the same sign
		if ((m_derivative(i) * m_oldDerivative(i) > 0) && (std::abs(delta) > std::abs(m_maxIncrease * m_deltaw(i))))
		{
			delta = boost::math::sign(delta) * std::abs(m_maxIncrease * m_deltaw(i));
		}

		m_deltaw(i) = delta;
		m_best.point(i) += m_deltaw(i);
	}
	m_oldDerivative = m_derivative;
	//evaluate new point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}

QuickpropOriginal::QuickpropOriginal()
{
	m_learningRate=1.5;
	m_maxIncrease=1.75;
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
}
void QuickpropOriginal::configure( const PropertyTree & node ) {
	m_learningRate=node.get("learningRate",1.5);
	m_maxIncrease=node.get("maxIncrease",1.75);
}

void QuickpropOriginal::read( InArchive & archive )
{
	archive>>m_deltaw;
	archive>>m_oldDerivative;
	archive>>m_learningRate;
	archive>>m_maxIncrease;
	archive>>m_parameterSize;
	archive>>m_best.point;
	archive>>m_best.value;
}

void QuickpropOriginal::write( OutArchive & archive ) const
{
	archive<<m_deltaw;
	archive<<m_oldDerivative;
	archive<<m_learningRate;
	archive<<m_maxIncrease;
	archive<<m_parameterSize;
	archive<<m_best.point;
	archive<<m_best.value;
}

void QuickpropOriginal::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	checkFeatures(objectiveFunction);

	m_parameterSize = startingPoint.size();
	m_deltaw.resize(m_parameterSize);
	m_oldDerivative.resize(m_parameterSize);

	m_oldDerivative.clear();
	m_deltaw.clear();
	m_best.point=startingPoint;
	//evaluate starting point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}

void QuickpropOriginal::step(const ObjectiveFunctionType& objectiveFunction) {

	double shrinkFactor = m_maxIncrease / (1 + m_maxIncrease);

	for (size_t i = 0; i < m_parameterSize; i++)
	{
		// avoid division by zero
		if (m_oldDerivative(i) == m_derivative(i)) break;

		double delta = 0.;

		if (m_deltaw(i) < 0.)
		{
			if (m_derivative(i) > 0.)
				// add negative gradient
				delta -= m_learningRate * m_derivative(i);
			if (m_derivative(i) > shrinkFactor * m_oldDerivative(i))
				delta += m_maxIncrease * m_deltaw(i);
			else
				// quadratic part
				delta += m_derivative(i) / (m_oldDerivative(i) - m_derivative(i)) * m_deltaw(i);
		}
		else if (m_deltaw(i) > 0.)
		{
			if (m_derivative(i) < 0.)
				// add negative gradient
				delta -= m_learningRate * m_derivative(i);
			if (m_derivative(i) < shrinkFactor * m_oldDerivative(i))
				delta += m_maxIncrease * m_deltaw(i);
			else
				// quadratic part
				delta += m_derivative(i) / (m_oldDerivative(i) - m_derivative(i)) * m_deltaw(i);
		}
		else
		{
			delta -= m_learningRate * m_derivative(i);
		}

		m_deltaw(i) = delta;
		m_best.point(i) += m_deltaw(i);
	}

	m_oldDerivative = m_derivative;
	
	//evaluate new point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}