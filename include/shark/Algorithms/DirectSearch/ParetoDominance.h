/*!
 * 
 *
 * \brief       Implementation of the Pareto-Dominance relation.
 * 
 *
 * \author      T. Glasmachers (based on old version by T. Voß)
 * \date        2011-2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_PARETODOMINANCE_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_PARETODOMINANCE_H


#include <shark/LinAlg/Base.h>


namespace shark {


/// \brief Result of comparing two objective vectors w.r.t. Pareto dominance.
///
/// The values of this type should not be tested directly,
/// instead the functions prec, preceq, succ, succeq, equivalent,
/// and incomparable should be used. They are overloaded to take
/// a ParetoRelation, two objective vectors, or two individuals as
/// arguments.
enum ParetoRelation
{
	INCOMPARABLE = 0,       // LHS and RHS are incomparable
	LHS_PREC_RHS = 1,       // LHS strictly dominates RHS
	RHS_PREC_LHS = 2,       // RHS strictly dominates LHS
	EQUIVALENT = 3,         // LHS and RHS are equally good
};

/// \brief Shorthand test for strict Pareto dominance of LHS over RHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \prec \f$.
inline bool prec(ParetoRelation relation)
{ return (relation == LHS_PREC_RHS); }

/// \brief Shorthand test for weak Pareto dominance of LHS over RHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \preceq \f$.
inline bool preceq(ParetoRelation relation)
{ return (relation & LHS_PREC_RHS); }

/// \brief Shorthand test for strict Pareto dominance of RHS over LHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \succ \f$.
inline bool succ(ParetoRelation relation)
{ return (relation == RHS_PREC_LHS); }

/// \brief Shorthand test for weak Pareto dominance of RHS over LHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \succeq \f$.
inline bool succeq(ParetoRelation relation)
{ return (relation & RHS_PREC_LHS); }

/// \brief Shorthand test for equality of the objective vectors LHS and RHS.
inline bool equivalent(ParetoRelation relation)
{ return (relation == EQUIVALENT); }

/// \brief Shorthand test for (Pareto) incomparability of the objective vectors LHS and RHS.
inline bool incomparable(ParetoRelation relation)
{ return (relation == INCOMPARABLE); }

/// \brief Pareto dominance relation for two objective vectors.
inline ParetoRelation dominance(RealVector const& lhs, RealVector const& rhs)
{
	SHARK_ASSERT(lhs.size() == rhs.size());
	std::size_t l = 0, r = 0;
	for (std::size_t i=0; i<lhs.size(); i++)
	{
		if (lhs[i] < rhs[i]) l++;
		else if (lhs[i] > rhs[i]) r++;
	}

	if (l > 0)
	{
		if (r > 0) return INCOMPARABLE;
		else return LHS_PREC_RHS;
	}
	else
	{
		if (r > 0) return RHS_PREC_LHS;
		else return EQUIVALENT;
	}
}

/// \brief Shorthand test for strict Pareto dominance of LHS over RHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \prec \f$.
inline bool prec(RealVector const& lhs, RealVector const& rhs)
{ return (prec(dominance(lhs, rhs))); }

/// \brief Shorthand test for weak Pareto dominance of LHS over RHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \preceq \f$.
inline bool preceq(RealVector const& lhs, RealVector const& rhs)
{ return (preceq(dominance(lhs, rhs))); }

/// \brief Shorthand test for strict Pareto dominance of RHS over LHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \succ \f$.
inline bool succ(RealVector const& lhs, RealVector const& rhs)
{ return (succ(dominance(lhs, rhs))); }

/// \brief Shorthand test for weak Pareto dominance of RHS over LHS.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \succeq \f$.
inline bool succeq(RealVector const& lhs, RealVector const& rhs)
{ return (succeq(dominance(lhs, rhs))); }

/// \brief Shorthand test for equality of the objective vectors LHS and RHS.
inline bool equivalent(RealVector const& lhs, RealVector const& rhs)
{ return (equivalent(dominance(lhs, rhs))); }

/// \brief Shorthand test for (Pareto) incomparability of the objective vectors LHS and RHS.
inline bool incomparable(RealVector const& lhs, RealVector const& rhs)
{ return (incomparable(dominance(lhs, rhs))); }

/// \brief Pareto dominance relation for two individuals.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
template <typename IndividualType, typename Extractor>
ParetoRelation dominance(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return dominance(e(lhs), e(rhs));
}

/// \brief Shorthand test for strict Pareto dominance of LHS over RHS.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \prec \f$.
template <typename IndividualType, typename Extractor>
bool prec(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return prec(e(lhs), e(rhs));
}

/// \brief Shorthand test for weak Pareto dominance of LHS over RHS.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \preceq \f$.
template <typename IndividualType, typename Extractor>
bool preceq(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return preceq(e(lhs), e(rhs));
}

/// \brief Shorthand test for strict Pareto dominance of RHS over LHS.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \succ \f$.
template <typename IndividualType, typename Extractor>
bool succ(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return succ(e(lhs), e(rhs));
}

/// \brief Shorthand test for weak Pareto dominance of RHS over LHS.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
///
/// Note: The function name is derived from the LaTeX symbol \f$ \succeq \f$.
template <typename IndividualType, typename Extractor>
bool succeq(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return succeq(e(lhs), e(rhs));
}

/// \brief Shorthand test for (Pareto) equivalence of the individuals LHS and RHS.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
template <typename IndividualType, typename Extractor>
bool equivalent(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return equivalent(e(lhs), e(rhs));
}

/// \brief Shorthand test for (Pareto) incomparability of the individuals LHS and RHS.
/// \tparam  IndividualType  Individual carrying an objective vector.
/// \tparam  Extractor       Extractor returning the objective vector of an individual.
template <typename IndividualType, typename Extractor>
bool incomparable(IndividualType const& lhs, IndividualType const& rhs)
{
	Extractor e;
	return incomparable(e(lhs), e(rhs));
}


};  // namespace shark
#endif
