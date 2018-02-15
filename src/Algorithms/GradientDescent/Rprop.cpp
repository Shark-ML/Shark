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
#define SHARK_COMPILE_DLL
#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
 
#include <boost/math/special_functions/sign.hpp>
#include <boost/serialization/base_object.hpp>
 
namespace shark{

template<class SearchPointType>
Rprop<SearchPointType>::Rprop(){
	this->m_features |= this->REQUIRES_VALUE;
	this->m_features |= this->REQUIRES_FIRST_DERIVATIVE;
	this->m_features |= this->CAN_SOLVE_CONSTRAINED;

	m_increaseFactor = 1.2;
	m_decreaseFactor = 0.5;
	m_maxDelta = 1e100;
	m_minDelta = 0.0;
	m_useFreezing = true;
	m_useBacktracking = true;
	m_useOldValue = true;
}

template<class SearchPointType>
void Rprop<SearchPointType>::init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint) {
	init(objectiveFunction,startingPoint,0.01);
}
template<class SearchPointType>
void Rprop<SearchPointType>::init(
	ObjectiveFunctionType const& objectiveFunction, 
	SearchPointType const& startingPoint, 
	double initDelta
) {
	this->checkFeatures(objectiveFunction);
	
	m_parameterSize = startingPoint.size();
	m_delta = SearchPointType(m_parameterSize, initDelta);
	m_oldDerivative = SearchPointType(m_parameterSize, 0.0);
	m_deltaw = SearchPointType(m_parameterSize, 0.0);
	m_oldValue = std::numeric_limits<double>::max();

	this->m_best.point = startingPoint;
	//evaluate initial point
	this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
}

template<class SearchPointType>
void Rprop<SearchPointType>::step(ObjectiveFunctionType const& objectiveFunction) {
	for (size_t i = 0; i < m_parameterSize; i++){
		double p = this->m_best.point(i);
		double direction = m_derivative(i) * m_oldDerivative(i);
		m_oldDerivative(i) = m_derivative(i);
		
		if(direction > 0){
			//successful last step, increase delta und perform step
			m_delta(i) = std::min(m_maxDelta, m_increaseFactor * m_delta(i));
			m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
		}
		else if (direction < 0){
			//last step jumped over optimum. decrease step size and perform backtrakcing
			//if necessary
			m_delta(i) = std::max(m_minDelta, m_decreaseFactor * m_delta(i));
			if(m_useFreezing)
				m_oldDerivative(i) = 0;
			if(!m_useBacktracking)
				m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
			else if(!m_useOldValue || m_oldValue < this->m_best.value){
				this->m_best.point(i) -= m_deltaw(i);
				m_deltaw(i) = 0.0;//backtracking does not count as step
			}
		}else{
			//we end up here after freezing. In this case, just follow the current gradient with current delta
			m_deltaw(i) = m_delta(i) * -boost::math::sign(m_derivative(i));
		}
		//perform the step
		this->m_best.point(i) += m_deltaw(i);
		
		//if the step was not feasible, undo and freeze
		if (! objectiveFunction.isFeasible(this->m_best.point)){
			this->m_best.point(i) = p;
			m_delta(i) *= m_decreaseFactor;
			m_oldDerivative(i) = 0.0;
		}
	}
	m_oldValue = this->m_best.value; //save old error value for backtracking
	this->m_best.value = objectiveFunction.evalDerivative(this->m_best.point,m_derivative);
}

template<class SearchPointType>
void Rprop<SearchPointType>::read( InArchive & archive ){
	archive>>m_delta;
	archive>>m_deltaw;
	archive>>m_oldDerivative;
	archive>>m_oldValue;
	archive>>m_increaseFactor;
	archive>>m_decreaseFactor;
	archive>>m_maxDelta;
	archive>>m_minDelta;
	archive>>m_parameterSize;
	archive>>this->m_best.point;
	archive>>this->m_best.value;
}

template<class SearchPointType>
void Rprop<SearchPointType>::write( OutArchive & archive ) const{
	archive<<m_delta;
	archive<<m_deltaw;
	archive<<m_oldDerivative;
	archive<<m_oldValue;
	archive<<m_increaseFactor;
	archive<<m_decreaseFactor;
	archive<<m_maxDelta;
	archive<<m_minDelta;
	archive<<m_parameterSize;
	archive<<this->m_best.point;
	archive<<this->m_best.value;
}

template class SHARK_EXPORT_SYMBOL Rprop<RealVector>;
template class SHARK_EXPORT_SYMBOL Rprop<FloatVector>;
}