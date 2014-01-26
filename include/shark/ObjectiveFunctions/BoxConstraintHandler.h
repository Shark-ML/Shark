//===========================================================================
/*!
 * 
 *
 * \brief       Base class for constraints.
 * 
 *
 * \author      O.Krause
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BOXCONSTRAINTHANDLER_H
#define SHARK_OBJECTIVEFUNCTIONS_BOXCONSTRAINTHANDLER_H

#include <shark/ObjectiveFunctions/AbstractConstraintHandler.h>
#include <shark/Rng/GlobalRng.h>

namespace shark{

template<class Vector>
class BoxConstraintHandler:public AbstractConstraintHandler<Vector> {
public:
	BoxConstraintHandler(Vector const& lower, Vector const& upper)
	:m_lower(lower),m_upper(upper){
		SIZE_CHECK(lower.size() == upper.size());
		typedef AbstractConstraintHandler<Vector> base_type;
		this->m_features |= base_type::CAN_PROVIDE_CLOSEST_FEASIBLE;
		this->m_features |= base_type::IS_BOX_CONSTRAINED;
		this->m_features |= base_type::CAN_GENERATE_RANDOM_POINT;
	}
	BoxConstraintHandler(std::size_t dim, double lower, double upper)
	:m_lower(Vector(dim,lower)),m_upper(Vector(dim,upper)){
		typedef AbstractConstraintHandler<Vector> base_type;
		this->m_features |= base_type::CAN_PROVIDE_CLOSEST_FEASIBLE;
		this->m_features |= base_type::IS_BOX_CONSTRAINED;
		this->m_features |= base_type::CAN_GENERATE_RANDOM_POINT;
	}
	
	BoxConstraintHandler(){
		typedef AbstractConstraintHandler<Vector> base_type;
		this->m_features |= base_type::CAN_PROVIDE_CLOSEST_FEASIBLE;
		this->m_features |= base_type::IS_BOX_CONSTRAINED;
		this->m_features |= base_type::CAN_GENERATE_RANDOM_POINT;
	}
	
	std::size_t dimensions()const{
		return m_lower.size();
	}
	
	bool isFeasible(Vector const& point)const{
		SIZE_CHECK(point.size() == dimensions());
		for(std::size_t i = 0; i != dimensions();++i){
			if(point(i) < m_lower(i)||point(i) > m_upper(i)) 
				return false;
		}
		return true;
	}
	void closestFeasible(Vector& point )const{
		SIZE_CHECK(point.size() == dimensions());
		for(std::size_t i = 0; i != dimensions();++i){
			point(i) = std::max(point(i),m_lower(i));
			point(i) = std::min(point(i),m_upper(i));
		}
	}
	
	virtual void generateRandomPoint( Vector & startingPoint )const {
		startingPoint.resize(dimensions());
		for(std::size_t i = 0; i != dimensions(); ++i){
			startingPoint(i) = Rng::uni(m_lower(i),m_upper(i));
		}
	}
	
	
	/// \brief Sets lower and upper bounds of the box.
	void setBounds(Vector const& lower, Vector const& upper){
		SIZE_CHECK(lower.size() == upper.size());
		m_lower = lower;
		m_upper = upper;
	}
	/// \brief Sets lower and upper bounds of the box.
	void setBounds(std::size_t dimension, double lower, double upper){
		m_lower = Vector(dimension,lower);
		m_upper = Vector(dimension,upper);
	}
	/// \brief Returns the lower bound of the box.
	Vector const& lower()const{
		return m_lower;
	}
	/// \brief Returns the upper bound of the box.
	Vector const& upper()const{
		return m_upper;
	}
private:
	/// \brief Represents the lower bound of the points in the box
	Vector m_lower;
	/// \brief Represents the upper bound of the points in the box
	Vector m_upper;
};
}
#endif