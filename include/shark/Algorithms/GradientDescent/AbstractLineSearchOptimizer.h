//===========================================================================
/*!
 * 
 *
 * \brief       Base class for Line Search Optimizer
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_GRADIENTDESCENT_ABSTRACTLINESEARCHOPTIMIZER_H
#define SHARK_ALGORITHMS_GRADIENTDESCENT_ABSTRACTLINESEARCHOPTIMIZER_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Algorithms/GradientDescent/LineSearch.h>

namespace shark {

/// \brief Basis class for line search methods.
///
/// Line Search optimizer find an iterative optimum by starting from some point, choosing a search direction and than
/// performing a line search in that direction. To choose the search direction a local model of the function is  often used.
/// This class is a base class for all line search method which implement the general behaviour of line search methods.
/// Derived classes only need to implement initModel() and computeSearchDirection() to initializee and update
/// the model and find a new line search direction. The remaining functionality is implemented by the optimizer.
///
/// Also derived classes should specialise read() and write() methods for serialization if they have additional members
/// as well as choose a name() for the optimizer.
class AbstractLineSearchOptimizer : public AbstractSingleObjectiveOptimizer< VectorSpace<double> > {
protected:
	/// \brief Initializes the internal model.
	///
	/// Line Search Methods use a Model to search for the next search direction.
	/// The model is initialized during init()
	virtual void initModel() = 0;

	/// \brief Updates the Model and computes the next search direction
	///
	/// After a step was performed, this method is called to compute the next 
	/// search direction. This usually involves updating the internal model using the 
	/// new and old step information. Afterwards m_searchDirection should contain
	/// the next search direction.
	virtual void computeSearchDirection() = 0;

public:
	AbstractLineSearchOptimizer();

	void init(const ObjectiveFunctionType &objectiveFunction, const SearchPointType &startingPoint) ;
	
	using AbstractSingleObjectiveOptimizer< VectorSpace<double> >::init;

	void step(const ObjectiveFunctionType &objectiveFunction);

	//from IConfigure
	void configure(const PropertyTree &node);

	//from ISerializable
	void read(InArchive &archive);
	void write(OutArchive &archive) const;


	//linesearch handling
	const LineSearch &lineSearch()const {
		return m_linesearch;
	}
	LineSearch &lineSearch() {
		return m_linesearch;
	}


protected: // Instance vars

	LineSearch m_linesearch; ///< used line search method.
	std::size_t m_dimension; ///< number of parameters
	double m_initialStepLength;///< Initial step length to begin with the line search.

	RealVector      m_derivative; ///< gradient of m_best.point
	RealVector      m_searchDirection;///< search direction of next step

	//information from previous step
	RealVector m_lastPoint; ///<  previous point
	RealVector m_lastDerivative; ///< gradient of the previous point
	double m_lastValue;     ///< value of the previous point
};

}
#endif
