//===========================================================================
/*!
 * 
 *
 * \brief       LBFGS
 * 
 * The Limited-Memory Broyden, Fletcher, Goldfarb, Shannon (BFGS) algorithm
 * is a quasi-Newton method for unconstrained real-valued optimization.
 * See: http://en.wikipedia.org/wiki/LBFGS for details.
 * 
 * 
 *
 * \author      S. Dahlgaard, O.Krause
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
//===========================================================================


#ifndef SHARK_ML_OPTIMIZER_LBFGS_H
#define SHARK_ML_OPTIMIZER_LBFGS_H

#include <shark/Algorithms/GradientDescent/AbstractLineSearchOptimizer.h>
#include <deque>

namespace shark {

/// \brief Limited-Memory Broyden, Fletcher, Goldfarb, Shannon algorithm.
///
/// BFGS is one of the best performing quasi-newton methods. However for large scale
/// optimization, storing the full hessian approximation is infeasible due to O(n^2) memory requirement.
/// The L-BFGS algorithm does not store the full hessian approximation but only stores the
/// data used for updating in the last steps. The matrix itself is then regenerated on-the-fly in
/// an implicit matrix scheme. This brings runtime and memory rquirements
/// of a single step down to O(n*hist_size).
///
/// The number of steps stored can be set with setHistCount. This is 100 as default.
///
/// The algorithm is implemented for unconstrained and constrained optimization
/// in the case of box constraints. When box constraints are present and the algorithm
/// encounters a constraint, a dog-leg style algorithm is used:
///
/// first, all variables with active constraints (e.g. x_i = l_i and g_i > 0)
/// are fixed, i.e. p_i = 0. For the remaining variables, the unconstrained optimization
/// problem is solved. If the solution does not statisfy the box constraints, in the next step
/// the cauchy point is computed. If the cauchy point is feasible, we search the point
/// along the line between unconstrained optimum and cauchy point that lies exactly on the constraint.
/// This is the point with smallest value along the path.
/// This does not find the true optimal step in the unconstrained problem, however a cheap and reasonably good optimum
/// which often improves over naive coordinate descent.
/// \ingroup gradientopt
template<class SearchPointType = RealVector>
class LBFGS : public AbstractLineSearchOptimizer<SearchPointType>{
public:
	typedef typename AbstractLineSearchOptimizer<SearchPointType>::ObjectiveFunctionType ObjectiveFunctionType;

	LBFGS() :m_numHist(100){
		this->m_features |= this->CAN_SOLVE_CONSTRAINED;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LBFGS"; }
	
	///  \brief Specify the amount of steps to be memorized and used to find the L-BFGS direction.
	///
	///\param numhist The amount of steps to use.
	void setHistCount(unsigned int numhist) {
		SHARK_RUNTIME_CHECK(numhist > 0, "An empty history is not allowed");
		m_numHist = numhist;
	}

	//from ISerializable
	void read(InArchive &archive);
	void write(OutArchive &archive) const;
protected: // Methods inherited from AbstractLineSearchOptimizer
	void initModel();
	void computeSearchDirection(ObjectiveFunctionType const&);
private:
	///\brief Stores another step and searchDirection, discarding the oldest on if necessary.
	///
	/// \param step Last performed step
	/// \param y difference in gradients
	void updateHist(SearchPointType& y, SearchPointType &step);
	/// \brief Compute B^{-1}x
	///
	/// The history is used to define B which is easy to invert
	void multBInv(SearchPointType& searchDirection)const;

	/// \brief Compute Bx
	void multB(SearchPointType& searchDirection)const;

	/// \brief Get the box-constrained LBFGS direction. 
	///
	/// Approximately solves the constrained optimization problem
	/// min_p 1/2 p^TBp +g^Tp
	/// s.t. l_i <= x_i + p_i <= u_i
	/// This is done using a constrained dogleg approach.
	///
	/// first, all variables with active constraints (e.g. x_i = l_i and g_i > 0)
	/// are fixed, i.e. p_i = 0. For the remaining variables, the unconstrained optimization
	/// problem is solved. If the solution does not statisfy the box constraints, in the next step
	/// the cauchy point is computed. If the cauchy point is feasible, we search the point
	/// along the line between unconstrained optimum and cauchy point that lies exactly on the constraint.
	/// This is the point with smallest value along the path.
	void getBoxConstrainedDirection(
		SearchPointType& searchDirection,
		SearchPointType const& lower,
		SearchPointType const& upper
	)const;

	double m_updThres;///<Threshold for when to update history.
	unsigned int m_numHist; ///< Number of steps to use for LBFGS.
	// Initial Hessian approximation. We use a diagonal matrix, where each element is
	// the same, so we only need to store one double.
	double          m_bdiag;

	// Saved steps for creating the approximation.
	// Use deque as it gives fast pop.front, push.back and access. Supposedly.
	// steps holds the values x_(k+1) - x_k
	// gradientDifferences holds the values g_(k+1) - g_k
	std::deque<SearchPointType> m_steps;
	std::deque<SearchPointType> m_gradientDifferences;	
};

//implementation is included in the library
extern template class LBFGS<RealVector>;
extern template class LBFGS<FloatVector>;

}
#endif
