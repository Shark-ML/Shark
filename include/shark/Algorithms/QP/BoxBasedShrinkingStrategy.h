/*!
 * 
 *
 * \brief       Shrinking strategy based on box constraints
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2017
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
#ifndef SHARK_ALGORITHMS_QP_BOXBASEDSHRINKINGSTRATEGY_H
#define SHARK_ALGORITHMS_QP_BOXBASEDSHRINKINGSTRATEGY_H

namespace shark{

	
/// \brief Takes q boxx constrained QP-type problem and implements shrinking on it
///
/// Given a QP-type-Problem, implements a strategy which allows to efficiently shrink
/// and unshrink the problem. If a value of the QP has an active box constraint,
/// it is shrinked from the problem when currently there is no possible step 
/// using that variable that leads to a gain. This is problem dependent as
/// this might involve consideration of additional constraints.
/// Therefore, every Problem must implement the method testShrinkVariable.
template<class Problem>
struct BoxBasedShrinkingStrategy : public Problem{
public:
	typedef typename Problem::QpFloatType QpFloatType;
	typedef typename Problem::MatrixType MatrixType;
	typedef typename Problem::PreferedSelectionStrategy PreferedSelectionStrategy;

	template<class ProblemT>
	BoxBasedShrinkingStrategy(ProblemT& problem, bool shrink=true)
	: Problem(problem)
	, m_isUnshrinked(false)
	, m_shrink(shrink)
	, m_gradientEdge(problem.linear)
	{ }

	using Problem::alpha;
	using Problem::gradient;
	using Problem::linear;
	using Problem::active;
	using Problem::dimensions;
	using Problem::quadratic;
	using Problem::isLowerBound;
	using Problem::isUpperBound;
	using Problem::boxMin;
	using Problem::boxMax;
	using Problem::setInitialSolution;

	virtual void updateSMO(std::size_t i, std::size_t j){
		double aiOld = alpha(i);
		double ajOld = alpha(j);
		//call base class to do the step
		Problem::updateSMO(i,j);
		double ai = alpha(i);
		double aj = alpha(j);
		
		// update the gradient edge data structure to keep up with changes
		updateGradientEdge(i,aiOld,ai);
		updateGradientEdge(j,ajOld,aj);
	}

	bool shrink(double epsilon){
		if(!m_shrink) return false;

		double largestUp;
		double smallestDown;
		getMaxKKTViolations(largestUp,smallestDown,active());

		// check whether unshrinking is necessary at this accuracy level
		if (!m_isUnshrinked  && (largestUp - smallestDown < 10.0 * epsilon))
		{
			unshrink();
			//recalculate maximum KKT violation for immediate re-shrinking
			getMaxKKTViolations(largestUp, smallestDown, dimensions());
		}
		//shrink
		auto& active = this->m_active;
		for (std::size_t a = active; a > 0; --a){
			std::size_t i = a-1;
			if(Problem::testShrinkVariable(i, largestUp, smallestDown)){
				Problem::flipCoordinates(i,active-1);
				std::swap( m_gradientEdge[i], m_gradientEdge[active-1]);
				--active;
			}
		}
		return true;
	}

	///\brief Unshrink the problem
	void unshrink(){
		if (active() == dimensions()) return;
		m_isUnshrinked = true;
		
		// Recompute the gradient of the whole problem.
		// We assume here that all shrinked variables are on the border of the problem.
		// The gradient of the active components is already correct and
		// we store the gradient of the subset of variables which are on the
		// borders of the box for the whole set.
		// Thus we only have to recompute the part of the gradient which is
		// based on variables in the active set which are not on the border.
		for (std::size_t a = active(); a < dimensions(); a++)
			this->m_gradient(a) = m_gradientEdge(a);

		for (std::size_t i = 0; i < active(); i++){
			//check whether alpha value is already stored in gradientEdge
			if (isUpperBound(i) || isLowerBound(i)) continue;
			
			QpFloatType* q = quadratic().row(i, 0, dimensions());
			for (std::size_t a = active(); a < dimensions(); a++) 
				this->m_gradient(a) -= alpha(i) * q[a];
		}

		this->m_active = dimensions();
	}

	void setShrinking(bool shrinking){
		m_shrink = shrinking;
		if(!shrinking)
			unshrink();
	}

	/// \brief Define the initial solution for the iterative solver.
	///
	/// This method can be used to warm-start the solver. It requires a
	/// feasible solution (alpha) and the corresponding gradient of the
	/// dual objective function.
	void setInitialSolution(RealVector const& alpha, RealVector const& gradient, RealVector const& gradientEdge){
		Problem::setInitialSolution(alpha,gradient);
		std::size_t n = dimensions();
		SIZE_CHECK(gradientEdge.size() == n);
		for (std::size_t i=0; i<n; i++){
			std::size_t j = this->permutation(i);
			m_gradientEdge(i) = gradientEdge(j);
		}
	}

	/// \brief Define the initial solution for the iterative solver.
	///
	/// This method can be used to warm-start the solver. It requires a
	/// feasible solution (alpha), for which it computes the gradient of
	/// the dual objective function. Note that this is a quadratic time
	/// operation in the number of non-zero coefficients.
	void setInitialSolution(RealVector const& alpha){
		std::size_t n = dimensions();
		SIZE_CHECK(alpha.size() == n);
		RealVector gradient = this->m_problem.linear;
		RealVector gradientEdge = this->m_problem.linear;
		blas::vector<QpFloatType> q(n);
		std::vector<std::size_t> inverse(n);
		for (std::size_t i=0; i<n; i++) 
			inverse[this->permutation(i)] = i;
		for (std::size_t i=0; i<n; i++)
		{
			double a = alpha(i);
			if (a == 0.0) continue;
			this->m_problem.quadratic.row(i, 0, n, q.raw_storage().values);
			noalias(gradient) -= a * q;
			std::size_t j = inverse[i];
			if (a == boxMin(j) || a == boxMax(j)) gradientEdge -= a * q;
		}
		setInitialSolution(alpha, gradient, gradientEdge);
	}
	
	///\brief Remove the i-th example from the problem.
	void deactivateVariable(std::size_t i){
		SIZE_CHECK(i < dimensions());
		RealVector alpha_old(dimensions);
		for(std::size_t i = 0; i != dimensions; ++i){
			updateGradientEdge(i,alpha_old(i), alpha(i));
		}
	}

	/// \brief Scales all box constraints by a constant factor and adapts the solution by scaling it by the same factor.
	void scaleBoxConstraints(double factor, double variableScalingFactor){
		Problem::scaleBoxConstraints(factor,variableScalingFactor);
		if(factor != variableScalingFactor){
			for(std::size_t i = 0; i != dimensions(); ++i){
				m_gradientEdge(i) = linear(i);
			}	
		}
		else{
			for(std::size_t i = 0; i != dimensions(); ++i){
				m_gradientEdge(i) -= linear(i);
				m_gradientEdge(i) *= factor;
				m_gradientEdge(i) += linear(i);
			}
		}
	}
	
	/// \brief adapts the linear part of the problem and updates the internal data structures accordingly.
	virtual void setLinear(std::size_t i, double newValue){
		m_gradientEdge(i) -= linear(i);
		m_gradientEdge(i) += newValue;
		Problem::setLinear(i,newValue);
	}
	
	///\brief swap indizes (i,j)
	void flipCoordinates(std::size_t i,std::size_t j){
		Problem::flipCoordinates(i,j);
		std::swap( m_gradientEdge[i], m_gradientEdge[j]);
	}
	
private:
	///\brief Updates the edge-part of the gradient when an alpha value was changed
	///
	/// This function overwites the base class method and is called whenever
	/// an alpha value is changed.
	void updateGradientEdge(std::size_t i, double oldAlpha, double newAlpha){
		SIZE_CHECK(i < active());
		if(!m_shrink || oldAlpha==newAlpha) return;
		bool isInsideOld = oldAlpha > boxMin(i) && oldAlpha < boxMax(i);
		bool isInsideNew = newAlpha > boxMin(i) && newAlpha < boxMax(i);
		//check if variable is relevant at all, that means that old and new alpha value are inside
		//or old alpha is 0 and new alpha inside
		if((oldAlpha == 0 || isInsideOld) && isInsideNew)
			return;

		//compute change to the gradient
		double diff = 0;
		if(!isInsideOld)//the value was on a border, so remove it's old influence on the gradient
			diff -=oldAlpha;
		if(!isInsideNew){//variable entered boundary or changed from one boundary to another
			diff += newAlpha;
		}

		QpFloatType* q = quadratic().row(i, 0, dimensions());
		for(std::size_t a = 0; a != dimensions(); ++a){
			m_gradientEdge(a) -= diff*q[a];
		}
	}

	void getMaxKKTViolations(double& largestUp, double& smallestDown, std::size_t maxIndex){
		largestUp = -1e100;
		smallestDown = 1e100;
		for (std::size_t a = 0; a < maxIndex; a++){
			if (!isLowerBound(a))
				smallestDown = std::min(smallestDown,gradient(a));
			if (!isUpperBound(a))
				largestUp = std::max(largestUp,gradient(a));
		}
	}
	
	bool m_isUnshrinked;
	
	///\brief true if shrinking is to be used.
	bool m_shrink;

	///\brief Stores the gradient of the alpha dimensions which are either 0 or C
	RealVector m_gradientEdge;
};

}
#endif

