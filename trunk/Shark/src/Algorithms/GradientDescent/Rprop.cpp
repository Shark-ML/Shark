/*!
 * 
 *
 * \brief       implements different versions of Resilient Backpropagation of error.
 * 
 * 
 * 
 *
 * \author      Oswin Krause
 * \date        2013
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
 
 #include <shark/Algorithms/GradientDescent/Rprop.h>
 
 #include <algorithm>

#include <boost/math/special_functions/sign.hpp>
#include <boost/serialization/base_object.hpp>
 
 using namespace shark;
 
 
//RPROP-MINUS>


RpropMinus::RpropMinus(){
	m_features |= REQUIRES_FIRST_DERIVATIVE;
	m_features |= CAN_SOLVE_CONSTRAINED;

	m_increaseFactor = 1.2;
	m_decreaseFactor = 0.5;
	m_maxDelta = 1e100;
	m_minDelta = 0.0;
}

void RpropMinus::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	init(objectiveFunction,startingPoint,0.01);
}
void RpropMinus::init(
	const ObjectiveFunctionType & objectiveFunction, 
	const SearchPointType& startingPoint, 
	double initDelta
) {
	checkFeatures(objectiveFunction);

	m_parameterSize = startingPoint.size();
	m_delta.resize(m_parameterSize);
	m_oldDerivative.resize(m_parameterSize);

	std::fill(m_delta.begin(),m_delta.end(),initDelta);
	m_oldDerivative.clear();
	m_best.point = startingPoint;
	//evaluate initial point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}
void RpropMinus::init(
	const ObjectiveFunctionType & objectiveFunction, 
	const SearchPointType& startingPoint, 
	const RealVector& initDelta
) {
	checkFeatures(objectiveFunction);

	m_parameterSize = startingPoint.size();
	m_delta.resize(m_parameterSize);
	m_oldDerivative.resize(m_parameterSize);

	m_delta   = initDelta;
	m_oldDerivative.clear();
	m_best.point=startingPoint;
	//evaluate initial point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}

void RpropMinus::step(const ObjectiveFunctionType& objectiveFunction) {
	for (size_t i = 0; i < m_parameterSize; i++)
	{
		double p = m_best.point(i);
		if (m_derivative(i) * m_oldDerivative(i) > 0)
		{
			m_delta(i) = std::min(m_maxDelta, m_increaseFactor * m_delta(i));
		}
		else if (m_derivative(i) * m_oldDerivative(i) < 0)
		{
			m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
		}
		m_best.point(i) -= m_delta(i) * boost::math::sign(m_derivative(i));
		if (! objectiveFunction.isFeasible(m_best.point))
		{
			m_best.point(i) = p;
			m_delta(i) *= m_decreaseFactor;
			m_oldDerivative(i) = 0.0;
		}
		else
		{
			m_oldDerivative(i) = m_derivative(i);
		}
	}
	//evaluate the new point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}

void RpropMinus::configure( const PropertyTree & node ) {
	m_increaseFactor=node.get("increaseFactor",1.2);
	m_decreaseFactor=node.get("decreaseFactor",0.5);
	m_maxDelta=node.get("maxDelta",1e100);
	m_minDelta=node.get("minDelta",0.0);
}

void RpropMinus::read( InArchive & archive )
{
	archive>>m_delta;
	archive>>m_oldDerivative;
	archive>>m_increaseFactor;
	archive>>m_decreaseFactor;
	archive>>m_maxDelta;
	archive>>m_minDelta;
	archive>>m_parameterSize;
	archive>>m_best.point;
	archive>>m_best.value;
}

void RpropMinus::write( OutArchive & archive ) const
{
	archive<<m_delta;
	archive<<m_oldDerivative;
	archive<<m_increaseFactor;
	archive<<m_decreaseFactor;
	archive<<m_maxDelta;
	archive<<m_minDelta;
	archive<<m_parameterSize;
	archive<<m_best.point;
	archive<<m_best.value;
}


//RPROP-PLUS

RpropPlus::RpropPlus()
{
}

void RpropPlus::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	init(objectiveFunction,startingPoint,0.01);
}
void RpropPlus::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, double initDelta)
{
	RpropMinus::init(objectiveFunction,startingPoint,initDelta);
	m_deltaw.resize(m_parameterSize);
	m_deltaw.clear();
}
void RpropPlus::init(
	const ObjectiveFunctionType & objectiveFunction, 
	const SearchPointType& startingPoint, 
	const RealVector& initDelta
){
	RpropMinus::init(objectiveFunction,startingPoint,initDelta);
	m_deltaw.resize(m_parameterSize);
	m_deltaw.clear();
}
void RpropPlus::step(const ObjectiveFunctionType& objectiveFunction) {
	for (size_t i = 0; i < m_parameterSize; i++)
	{
		//save the current value to ensure, that it can be restored
		double p = m_best.point(i);
		if (m_derivative(i) * m_oldDerivative(i) > 0)
		{
			m_delta(i) = std::min(m_maxDelta, m_increaseFactor * m_delta(i));
			m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
			m_best.point(i)+=m_deltaw(i);
			m_oldDerivative(i) = m_derivative(i);
		}
		else if (m_derivative(i) * m_oldDerivative(i) < 0)
		{
			m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
			m_best.point(i)-=m_deltaw(i);
			m_oldDerivative(i) = 0;
		}
		else
		{
			m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
			m_best.point(i)+=m_deltaw(i);
			m_oldDerivative(i) = m_derivative(i);
		}
		if (! objectiveFunction.isFeasible(m_best.point))
		{
			m_best.point(i)=p;
			m_delta(i) *= m_decreaseFactor;
			m_oldDerivative(i) = 0.0;
		}
	}
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}
void RpropPlus::read( InArchive & archive )
{
	archive>>boost::serialization::base_object<RpropMinus>(*this);
	archive>>m_deltaw;
}

void RpropPlus::write( OutArchive & archive ) const
{
	archive<<boost::serialization::base_object<RpropMinus>(*this);
	archive<<m_deltaw;
}

//IRpropPlus


IRpropPlus::IRpropPlus()
{
	m_features |= REQUIRES_VALUE;
	m_derivativeThreshold = 0.;
}

void IRpropPlus::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	init(objectiveFunction,startingPoint,0.01);
}
void IRpropPlus::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, double initDelta) {
	if(!(objectiveFunction.features() & ObjectiveFunctionType::HAS_VALUE))
		SHARKEXCEPTION("[IRPropPlus::init] requires the value of the function");
	RpropPlus::init(objectiveFunction,startingPoint,initDelta);
	m_oldError = std::numeric_limits<double>::max();
}
void IRpropPlus::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, const RealVector& initDelta) {
	checkFeatures(objectiveFunction);

	RpropPlus::init(objectiveFunction,startingPoint,initDelta);
	m_oldError = std::numeric_limits<double>::max();
}

void IRpropPlus::step(const ObjectiveFunctionType& objectiveFunction) {
	for (size_t i = 0; i < m_parameterSize; i++)
	{
		if(std::abs(m_derivative(i)) < m_derivativeThreshold) m_derivative(i) = 0.;
		double p = m_best.point(i);
		double direction = m_derivative(i) * m_oldDerivative(i);
		if ( direction > 0)
		{
			m_delta(i) = std::min(m_maxDelta, m_increaseFactor * m_delta(i));
			m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
			m_best.point(i) += m_deltaw(i);
			m_oldDerivative(i) = m_derivative(i);
		}
		else if (direction < 0)
		{
			m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
			if (m_best.value > m_oldError)
			{
				m_best.point(i) -= m_deltaw(i);
			}
			m_oldDerivative(i) = 0;
		}
		else
		{
			m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
			m_best.point(i) += m_deltaw(i);
			m_oldDerivative(i) = m_derivative(i);
		}
		if (! objectiveFunction.isFeasible(m_best.point))
		{
			m_best.point(i)=p;
			m_delta(i) *= m_decreaseFactor;
			m_oldDerivative(i) = 0.0;
		}
	}
	m_oldError = m_best.value;
	m_best.value = objectiveFunction.evalDerivative( m_best.point, m_derivative );
}

void IRpropPlus::read( InArchive & archive ) {
	archive>>boost::serialization::base_object<RpropPlus>(*this);
	archive>>m_oldError;
}
void IRpropPlus::write( OutArchive & archive ) const {
	archive<<boost::serialization::base_object<RpropPlus>(*this);
	archive<<m_oldError;
}

void IRpropPlus::setDerivativeThreshold(double derivativeThreshold)  {
	m_derivativeThreshold = derivativeThreshold;		
}


IRpropPlusFull::IRpropPlusFull()
{
	m_features |= REQUIRES_VALUE;
	m_derivativeThreshold = 0.;
}

void IRpropPlusFull::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	init(objectiveFunction,startingPoint,0.01);
}
void IRpropPlusFull::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, double initDelta) {
	if(!(objectiveFunction.features() & ObjectiveFunctionType::HAS_VALUE))
		SHARKEXCEPTION("[IRPropPlus::init] requires the value of the function");
	RpropPlus::init(objectiveFunction,startingPoint,initDelta);
	m_oldError = std::numeric_limits<double>::max();
}
void IRpropPlusFull::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, const RealVector& initDelta) {
	checkFeatures(objectiveFunction);

	RpropPlus::init(objectiveFunction,startingPoint,initDelta);
	m_oldError = std::numeric_limits<double>::max();
}

void IRpropPlusFull::step(const ObjectiveFunctionType& objectiveFunction) {
	if ( m_best.value < m_oldError){//accept the point as the new current one if it is better
		//step size adaptation
		for (size_t i = 0; i < m_parameterSize; i++)
		{
			if(std::abs(m_derivative(i)) < m_derivativeThreshold) m_derivative(i) = 0.;
			double direction = m_derivative(i) * m_oldDerivative(i);
			if(direction < 0){//decrease if we overstepped the optimum
				m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
			}
			if( direction > 0)//increase if we still go in the same direction
			{
				m_delta(i) = std::min(m_maxDelta, m_increaseFactor * m_delta(i));
			}
		}
		//accept the point as the new current one
		m_oldDerivative = m_derivative;
		m_oldError = m_best.value;
	}
	else{
		//do a full backtrack
		noalias(m_best.point) -= m_deltaw;
		for (size_t i = 0; i < m_parameterSize; i++)
		{
			if(std::abs(m_derivative(i)) < m_derivativeThreshold) m_derivative(i) = 0.;
			double direction = m_derivative(i) * m_oldDerivative(i);
			if(direction < 0){//this went too far...
				m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
			}
		}
	}
	
	//propose new step with updated step sizes
	for (size_t i = 0; i < m_parameterSize; i++)
	{
		m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
		m_best.point(i) += m_deltaw(i);
	}
	m_best.value = objectiveFunction.evalDerivative( m_best.point, m_derivative );
}

void IRpropPlusFull::read( InArchive & archive ) {
	archive>>boost::serialization::base_object<RpropPlus>(*this);
	archive>>m_oldError;
}
void IRpropPlusFull::write( OutArchive & archive ) const {
	archive<<boost::serialization::base_object<RpropPlus>(*this);
	archive<<m_oldError;
}

void IRpropPlusFull::setDerivativeThreshold(double derivativeThreshold)  {
	m_derivativeThreshold = derivativeThreshold;		
}


//IRpropMinus

IRpropMinus::IRpropMinus()
{
}

void IRpropMinus::step(const ObjectiveFunctionType& objectiveFunction) {
	for (size_t i = 0; i < m_parameterSize; i++)
	{
		double p = m_best.point(i);
		double direction = m_derivative(i) * m_oldDerivative(i);
		if (direction > 0)
		{
			m_delta(i) = std::min(m_maxDelta, m_increaseFactor * m_delta(i));
			m_oldDerivative(i) = m_derivative(i);
		}
		else if (direction < 0)
		{
			m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
			m_oldDerivative(i) = 0;
		}
		else
		{
			m_oldDerivative(i) = m_derivative(i);
		}
		m_best.point(i)-=m_delta(i) * boost::math::sign(m_derivative(i));
		if (! objectiveFunction.isFeasible(m_best.point))
		{
			m_best.point(i)=p;
			m_delta(i) *= m_decreaseFactor;
			m_oldDerivative(i) = 0.0;
		}
	}
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
}
