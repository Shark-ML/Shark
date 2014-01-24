//===========================================================================
/*!
 * 
 * \file        EvaluationArchive.h
 *
 * \brief       Archive of evaluated points as an objective function wrapper.

 * 
 *
 * \author      T. Glasmachers
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_EVALUATIONARCHIVE_H
#define SHARK_OBJECTIVEFUNCTIONS_EVALUATIONARCHIVE_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

#include <set>
#include <string>
#include <sstream>


namespace shark {


///
/// \brief Objective function wrapper storing all function evaluations.
///
/// \tparam SearchSpaceT The search space the function is defined upon.
/// \tparam ResultT The objective space the function is defined upon.
///
/// \par
/// The EvaluationArchive class serves as an archive of all evaluated
/// points and the corresponding result. It can be used transparently
/// instead of the original objective function, e.g., by an optimizer.
/// Point/result pairs are added to the archive only if the specific
/// combination is not yet stored.
///
/// \par
/// For fast-to-evaluate objective functions the archive wrapper can
/// be a considerable performance killer. However, whenever function
/// evaluations are costly (and an archive makes sense) then the
/// storage and maintenance overhead should be negligible.
///
template <typename SearchSpaceT, typename ResultT>
class EvaluationArchive : public AbstractObjectiveFunction<SearchSpaceT, ResultT>
{
public:
	typedef AbstractObjectiveFunction<SearchSpaceT, ResultT> base_type;
	typedef SearchSpaceT SearchSpaceType;
	typedef typename SearchSpaceT::PointType SearchPointType;
	typedef ResultT ResultType;

	typedef SearchPointType FirstOrderDerivative;
	typedef TypedSecondOrderDerivative<SearchPointType,RealMatrix> SecondOrderDerivative;

	/// \brief Pair of point and result.
	class PointResultPairType
	{
	public:
		PointResultPairType(SearchPointType p, ResultType r)
		: point(p)
		, result(r)
		{ }

		PointResultPairType(PointResultPairType const& other)
		: point(other.point)
		, result(other.result)
		{ }

		// Comparison is based on string representation.
		// This is a hack, but it is quite generic.
		// And a generic solution is needed for std::set.
		bool operator == (PointResultPairType const& other) const
		{
			return (toString() == other.toString());
		}
		bool operator < (PointResultPairType const& other) const
		{
			return (toString() < other.toString());
		}

		SearchPointType point;
		ResultType result;

	private:
		std::string toString() const
		{
			std::stringstream ss;
			ss << point << " " << result;
			return ss.str();
		}
	};

	typedef std::set<PointResultPairType> PointResultPairContainer;
	typedef typename PointResultPairContainer::iterator PointResultPairIterator;
	typedef typename PointResultPairContainer::const_iterator PointResultPairConstIterator;

	/// \brief Constructor.
	///
	/// \par
	/// The constructor takes the objective function to be wrapped
	/// as an argument. It is assumed that the objective object's
	/// life time exceeds the life time of the present instance.
	EvaluationArchive(base_type* objective)
	: mep_objective(objective)
	{
		base_type::m_features = mep_objective->features();
		base_type::m_constraintHandler = mep_objective->hasConstraintHandler() ? &mep_objective->getConstraintHandler() : NULL;
	}


	/// \brief Access to the underlying objective function.
	base_type* objective()
	{ return mep_objective; }

	/// \brief Access to the underlying objective function.
	const base_type* objective() const
	{ return mep_objective; }

	/// \brief Wrapper function.
	void init()
	{ mep_objective->init(); }

	/// \brief Wrapper function.
	virtual std::size_t numberOfObjectives() const
	{ return mep_objective->numberOfObjectives(); }

	/// \brief Wrapper function.
	bool hasScalableObjectives() const
	{ return mep_objective->hasScalableObjectives(); }

	/// \brief Wrapper function.
	void setNumberOfObjectives(std::size_t numberOfObjectives)
	{ mep_objective->setNumberOfObjectives(numberOfObjectives); }

	/// \brief Wrapper function.
	bool isFeasible(const SearchPointType& input) const
	{ return mep_objective->isFeasible(input); }

	/// \brief Wrapper function.
	void closestFeasible(SearchPointType& input) const
	{ return mep_objective->closestFeasible(input); }

	/// \brief Wrapper function.
	void proposeStartingPoint(SearchPointType& startingPoint) const
	{ return mep_objective->proposeStartingPoint(startingPoint); }

	/// \brief Wrapper function; conditional on vector space property.
	std::size_t numberOfVariables() const
	{
		AbstractVectorSpaceObjectiveFunction* avsof = dynamic_cast<AbstractVectorSpaceObjectiveFunction*>(mep_objective);
		if (avsof) return avsof->numberOfVariables();
		else throw SHARKEXCEPTION("search space is not a vector space");
	}

	/// \brief Wrapper function storing point and result.
	ResultType eval(const SearchPointType& input) const
	{
		ResultType r = mep_objective->eval(input);
		base_type::m_evaluationCounter++;
		m_archive.insert(PointResultPairType(input, r));
		return r;
	}

	// TG: Could someone enlighten me: why do I have to copy this
	// from the super class to make the compiler find the f**king
	// operator??
	ResultType operator()( const SearchPointType & input ) const
	{ return eval(input); }

	/// \brief Wrapper function storing point and result.
	ResultType evalDerivative(const SearchPointType& input, FirstOrderDerivative& derivative) const
	{
		ResultType r = mep_objective->evalDerivative(input, derivative);
		base_type::m_evaluationCounter++;
		m_archive.insert(PointResultPairType(input, r));
		return r;
	}

	/// \brief Wrapper function storing point and result.
	ResultType evalDerivative(const SearchPointType& input, SecondOrderDerivative& derivative) const
	{
		ResultType r = mep_objective->evalDerivative(input, derivative);
		base_type::m_evaluationCounter++;
		m_archive.insert(PointResultPairType(input, r));
		return r;
	}


	////////////////////////////////////////////////////////////
	// access to the archive
	//

	/// Return the size of the archive; which is the number of point/result pairs.
	std::size_t size() const
	{ return m_archive.size(); }

	/// Begin iterator to the point/result pairs.
	PointResultPairIterator begin()
	{ return m_archive.begin(); }

	/// Begin iterator to the point/result pairs.
	PointResultPairConstIterator begin() const
	{ return m_archive.begin(); }

	/// End iterator to the point/result pairs.
	PointResultPairIterator end()
	{ return m_archive.end(); }

	/// End iterator to the point/result pairs.
	PointResultPairConstIterator end() const
	{ return m_archive.end(); }

private:
	base_type* mep_objective;                     ///< objective function to be wrapped
	mutable PointResultPairContainer m_archive;   ///< evaluated point/result pairs
};


};
#endif
