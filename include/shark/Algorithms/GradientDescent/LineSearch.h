//===========================================================================
/*!
 * 
 *
 * \brief       LineSearch
 * 
 * 
 *
 * \author      O. Krause, S. Dahlgaard
 * \date        2010-2017
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

#ifndef SHARK_ALGORITHMS_GRADIENTDESCENT_LINESEARCH_H
#define SHARK_ALGORITHMS_GRADIENTDESCENT_LINESEARCH_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/ISerializable.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

namespace shark {
	
enum class LineSearchType {
	Dlinmin,
	WolfeCubic,
	Backtracking
};
	
///\brief Wrapper for the linesearch class of functions in the linear algebra library.
///
///This class is a wrapper for the linesearch class of functions of the linear algebra library.
///The class is used for example in CG or BFGS for their internal linesearch learning steps.
///It is NOT an Optimizer on its own, since it needs the Newton direction to be specified.
/// \ingroup gradientopt
template<class SearchPointType>
class LineSearch:public ISerializable {
public:
	
	typedef AbstractObjectiveFunction<SearchPointType,double> ObjectiveFunction;

	///Initializes the internal variables of the class to useful default values.
	///Dlinmin is used as default
	LineSearch() {
		m_minInterval=0;
		m_maxInterval=1;
		m_lineSearchType= LineSearchType::WolfeCubic;
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
	///@param derivative the derivative of the function at searchPoint
	///@param stepLength initial step length guess for guiding the line search
	void operator()(SearchPointType &searchPoint,double &pointValue,SearchPointType const& newtonDirection, SearchPointType &derivative, double stepLength = 1.0)const;

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

extern template class LineSearch<RealVector>;
extern template class LineSearch<FloatVector>;
#ifdef SHARK_USE_OPENCL
extern template class LineSearch<RealGPUVector>;
extern template class LineSearch<FloatGPUVector>;
#endif
}

#endif
