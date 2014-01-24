//===========================================================================
/*!
 * 
 * \file        LineSearch.h
 *
 * \brief       LineSearch
 * 
 * 
 *
 * \author      O. Krause, S. Dahlgaard
 * \date        2010-2013
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

#ifndef SHARK_ALGORITHMS_GRADIENTDESCENT_LINESEARCH_H
#define SHARK_ALGORITHMS_GRADIENTDESCENT_LINESEARCH_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/IConfigurable.h>
#include <shark/Core/ISerializable.h>
#include "Impl/wolfecubic.inl"
#include "Impl/dlinmin.inl"
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

namespace shark {
///\brief Wrapper for the linesearch class of functions in the linear algebra library.
///
///This class is a wrapper for the linesearch class of functions of the linear algebra library.
///The class is used for example in CG or BFGS for their internal linesearch learning steps.
///It is NOT an Optimizer on its own, since it needs the Newton direction to be specified.
class LineSearch:public IConfigurable, public ISerializable {
public:
	enum LineSearchType {
	    Dlinmin,
	    WolfeCubic
	};
	typedef AbstractObjectiveFunction<VectorSpace<double>,double> ObjectiveFunction;

	///Initializes the internal variables of the class to useful default values.
	///Dlinmin is used as default
	LineSearch() {
		m_minInterval=0;
		m_maxInterval=1;
		m_lineSearchType=Dlinmin;
	}

	LineSearchType lineSearchType()const {
		return m_lineSearchType;
	}
	LineSearchType &lineSearchType() {
		return m_lineSearchType;
	}
	///minInterval sets the minimum initial bracket
	double minInterval()const {
		return m_minInterval;
	}
	///minInterval sets the minimum initial bracket
	double &minInterval() {
		return m_minInterval;
	}
	///maxInterval sets the maximum initial bracket
	double maxInterval()const {
		return m_maxInterval;
	}
	///maxInterval sets the maximum initial bracket
	double &maxInterval() {
		return m_maxInterval;
	}

	///initializes the internal state of the LineSearch class and sets the function on which the lineSearch is to be evaluated
	void init(ObjectiveFunction const& objectiveFunction) {
		m_function = &objectiveFunction;
	}

	///performs a linesearch on the objectiveFunction given the starting point, its value the newton direction and optionally the derivative at the starting point
	///@param searchPoint the point where the linesearch start
	///@param pointValue the value of the function at searchPoint
	///@param newtonDirection the search direction of the line search
	///@param derivative the derivative of the funktion at searchPoint
	///@param stepLength initial step length guess for guiding the line search
	virtual void operator()(RealVector &searchPoint,double &pointValue,RealVector const& newtonDirection, RealVector &derivative, double stepLength = 1.0)const {
		switch (m_lineSearchType) {
		case Dlinmin:
			detail::dlinmin(searchPoint, newtonDirection, pointValue, *m_function, m_minInterval, m_maxInterval);
			m_function->evalDerivative(searchPoint, derivative);
			break;
		case WolfeCubic:
			detail::wolfecubic(searchPoint, newtonDirection, pointValue, *m_function, derivative, stepLength);
			break;
		}
	}

	//IConfigurable
	void configure(PropertyTree const& node) {
		m_lineSearchType=static_cast<LineSearchType>(node.get("searchtype",(unsigned int)Dlinmin));
		m_minInterval=node.get("minInterval",0.0);
		m_maxInterval=node.get("maxInterval",1.0);
	}

	//ISerializable
	virtual void read(InArchive &archive) {
		archive>>m_minInterval;
		archive>>m_maxInterval;
		archive>>m_lineSearchType;
	}

	virtual void write(OutArchive &archive) const {
		archive<<m_minInterval;
		archive<<m_maxInterval;
		archive<<m_lineSearchType;
	}


protected:
	///initial [min,max] bracket for linesearch
	double m_minInterval;
	///initial [min,max] bracket for linesearch
	double m_maxInterval;

	LineSearchType m_lineSearchType;

	///function to optimize
	ObjectiveFunction const* m_function;
};
}

#endif
