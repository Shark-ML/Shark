/*!
 * 
 *
 * \brief       Quadratic program definitions.
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause
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
#ifndef SHARK_ALGORITHMS_QP_BOXCONSTRAINEDPROBLEMS_H
#define SHARK_ALGORITHMS_QP_BOXCONSTRAINEDPROBLEMS_H
 
#include <shark/Algorithms/QP/QpSolver.h>
#include <shark/Algorithms/QP/Impl/AnalyticProblems.h>
#include <shark/Algorithms/QP/BoxBasedShrinkingStrategy.h>

namespace shark {

/// \brief Working set selection by maximization of the projected gradient.
///
/// This selection operator picks the largest and second largest variable index if possible.
struct WS2MaximumGradientCriterion{
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j){
		i = 0;
		j = 0;
		double largestGradient = 0;
		double secondLargestGradient = 0;

		for (std::size_t a = 0; a < problem.active(); a++){
			double g = problem.gradient(a);
			if (!problem.isUpperBound(a) && g > secondLargestGradient){
				secondLargestGradient = g;
				j = a;
			}
			if (!problem.isLowerBound(a) && -g > secondLargestGradient){
				secondLargestGradient = -g;
				j = a;
			}
			if(secondLargestGradient > largestGradient){
				std::swap(secondLargestGradient,largestGradient);
				std::swap(i,j);
			}
		}
		if(secondLargestGradient == 0)
			j = i;
		return largestGradient;
	}

	void reset(){}
};

/// \brief Working set selection by maximization of the projected gradient.
///
/// This selection operator picks a single variable index.
struct MaximumGradientCriterion{
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j){
		WS2MaximumGradientCriterion criterion;
		double value = criterion(problem, i,j);
		j = i; //we just use one variable here
		return value;
	}

	void reset(){}
};

/// \brief Working set selection by maximization of the dual objective gain.
struct MaximumGainCriterion{
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j){
		//choose first variable by first order criterion
		MaximumGradientCriterion firstOrder;
		double maxGrad = firstOrder(problem,i,j);
		if (maxGrad == 0.0)
			return maxGrad;

		double gi = problem.gradient(i);
		typename Problem::QpFloatType* q = problem.quadratic().row(i, 0, problem.active());
		double Qii = problem.diagonal(i);

		// select second variable j with second order method
		double maxGain = 0.0;
		for (std::size_t a=0; a<problem.active(); a++)
		{
			if (a == i) continue;
			double ga = problem.gradient(a);
			if (
				(!problem.isLowerBound(a) && ga < 0.0) 
				|| (!problem.isUpperBound(a) && ga > 0.0)
			){
				double Qia = q[a];
				double Qaa = problem.diagonal(a);
				double gain = detail::maximumGainQuadratic2D(Qii,Qaa,Qia,gi,ga);
				if (gain > maxGain)
				{
					maxGain = gain;
					j = a;
				}
			}
		}

		return maxGrad;		// solution is not optimal
	}

	void reset(){}
};

/// \brief Quadratic program with box constraints.
///
/// \par
/// An instance of this class represents a quadratic program of the type
/// TODO: write documentation!
///
template<class SVMProblem>
class BoxConstrainedProblem{
public:
	typedef typename SVMProblem::QpFloatType QpFloatType;
	typedef typename SVMProblem::MatrixType MatrixType;
	typedef MaximumGainCriterion PreferedSelectionStrategy;
	//~ typedef MaximumGradientCriterion PreferedSelectionStrategy;

	BoxConstrainedProblem(SVMProblem& problem)
	: m_problem(problem)
	, m_gradient(problem.linear)
	, m_active (problem.dimensions())
	, m_alphaStatus(problem.dimensions(),AlphaFree){
		//compute the gradient if alpha != 0
		for (std::size_t i=0; i != dimensions(); i++){
			double v = alpha(i);
			if (v != 0.0){
				QpFloatType* q = quadratic().row(i, 0, dimensions());
				for (std::size_t a=0; a < dimensions(); a++) 
					m_gradient(a) -= q[a] * v;
			}
			updateAlphaStatus(i);
		}
	}
	std::size_t dimensions()const{
		return m_problem.dimensions();
	}

	std::size_t active()const{
		return m_active;
	}

	double boxMin(std::size_t i)const{
		return m_alphaStatus[i]==AlphaDeactivated? alpha(i): m_problem.boxMin(i);
	}
	double boxMax(std::size_t i)const{
		return m_alphaStatus[i]==AlphaDeactivated? alpha(i): m_problem.boxMax(i);
	}
	bool isLowerBound(std::size_t i)const{
		return m_alphaStatus[i] & AlphaLowerBound;
	}
	bool isUpperBound(std::size_t i)const{
		return m_alphaStatus[i] & AlphaUpperBound;
	}
	bool isDeactivated(std::size_t i)const{
		return isUpperBound(i) && isLowerBound(i);
	}

	/// representation of the quadratic part of the objective function
	MatrixType& quadratic(){
		return m_problem.quadratic;
	}

	double linear(std::size_t i)const{
		return m_problem.linear(i);
	}

	double alpha(std::size_t i)const{
		return m_problem.alpha(i);
	}

	double diagonal(std::size_t i)const{
		return m_problem.diagonal(i);
	}

	double gradient(std::size_t i)const{
		return m_gradient(i);
	}
	
	std::size_t permutation(std::size_t i)const{
		return m_problem.permutation[i];
	}

	RealVector getUnpermutedAlpha()const{
		RealVector alpha(dimensions());
		for (std::size_t i=0; i<dimensions(); i++) 
			alpha(m_problem.permutation[i]) = m_problem.alpha(i);
		return alpha;
	}

	///\brief Does an update of SMO given a working set with indices i and j.
	virtual void updateSMO(std::size_t i, std::size_t j){
		SIZE_CHECK(i < active());
		SIZE_CHECK(j < active());
		if(i == j){//both variables are identical, thus solve the 1-d problem.
			// get the matrix row corresponding to the working set
			QpFloatType* q = quadratic().row(i, 0, active());

			// update alpha, that is, solve the sub-problem defined by i
			// and compute the stepsize mu of the step
			double mu = -alpha(i);
			detail::solveQuadraticEdge(m_problem.alpha(i),gradient(i),diagonal(i),boxMin(i),boxMax(i));
			mu+=alpha(i);
			
			// update the internal states
			for (std::size_t a = 0; a < active(); a++) 
				m_gradient(a) -= mu * q[a];
			
			updateAlphaStatus(i);
			return;
		}
		
		double Li = boxMin(i);
		double Ui = boxMax(i);
		double Lj = boxMin(j);
		double Uj = boxMax(j);

		// get the matrix rows corresponding to the working set
		QpFloatType* qi = quadratic().row(i, 0, active());
		QpFloatType* qj = quadratic().row(j, 0, active());

		// solve the 2D sub-problem imposed by the two chosen variables
		// and compute the stepsizes mu
		double mui = -alpha(i);
		double muj = -alpha(j);
		detail::solveQuadratic2DBox(m_problem.alpha(i), m_problem.alpha(j),
			m_gradient(i), m_gradient(j),
			diagonal(i), qi[j], diagonal(j),
			Li, Ui, Lj, Uj
		);
		mui += alpha(i);
		muj += alpha(j);

		// update the internal states
		for (std::size_t a = 0; a < active(); a++) 
			m_gradient(a) -= mui * qi[a] + muj * qj[a];
			
		updateAlphaStatus(i);
		updateAlphaStatus(j);
	}

	///\brief Returns the current function value of the problem.
	double functionValue()const{
		return 0.5*inner_prod(m_gradient+m_problem.linear,m_problem.alpha);
	}

	bool shrink(double){return false;}
	void reshrink(){}
	void unshrink(){}

	/// \brief Define the initial solution for the iterative solver.
	///
	/// This method can be used to warm-start the solver. It requires a
	/// feasible solution (alpha) and the corresponding gradient of the
	/// dual objective function.
	void setInitialSolution(RealVector const& alpha, RealVector const& gradient)
	{
		std::size_t n = dimensions();
		SIZE_CHECK(alpha.size() == n);
		SIZE_CHECK(gradient.size() == n);
		for (std::size_t i=0; i<n; i++)
		{
			std::size_t j = permutation(i);
			SHARK_ASSERT(alpha(j) >= boxMin(j) && alpha(j) <= boxMax(j));
			m_problem.alpha(i) = alpha(j);
			m_gradient(i) = gradient(j);
			updateAlphaStatus(i);
		}
	}

	/// \brief Define the initial solution for the iterative solver.
	///
	/// This method can be used to warm-start the solver. It requires a
	/// feasible solution (alpha), for which it computes the gradient of
	/// the dual objective function. Note that this is a quadratic time
	/// operation in the number of non-zero coefficients.
	void setInitialSolution(RealVector const& alpha)
	{
		std::size_t n = dimensions();
		SIZE_CHECK(alpha.size() == n);
		RealVector gradient = m_problem.linear;
		blas::vector<QpFloatType> q(n);
		for (std::size_t i=0; i<n; i++)
		{
			double a = alpha(i);
			if (a == 0.0) continue;
			m_problem.quadratic.row(i, 0, n, q.storage());
			noalias(gradient) -= a * q;
		}
		setInitialSolution(alpha, gradient);
	}
	
	///\brief Remove the i-th example from the problem.
	void deactivateVariable(std::size_t i){
		SIZE_CHECK(i < dimensions());
		double alphai = alpha(i);
		m_problem.alpha(i) = 0;
		//update the internal state
		QpFloatType* qi = quadratic().row(i, 0, active());
		for (std::size_t a = 0; a < active(); a++) 
			m_gradient(a) += alphai * qi[a];
		m_alphaStatus[i] = AlphaDeactivated;
	}
	///\brief Reactivate an previously deactivated variable.
	void activateVariable(std::size_t i){
		SIZE_CHECK(i < dimensions());
		updateAlphaStatus(i);
	}
	
	/// exchange two variables via the permutation
	void flipCoordinates(std::size_t i, std::size_t j)
	{
		SIZE_CHECK(i < dimensions());
		SIZE_CHECK(j < dimensions());
		if (i == j) return;

		m_problem.flipCoordinates(i, j);
		std::swap( m_gradient[i], m_gradient[j]);
		std::swap( m_alphaStatus[i], m_alphaStatus[j]);
	}
	
	/// \brief adapts the linear part of the problem and updates the internal data structures accordingly.
	virtual void setLinear(std::size_t i, double newValue){
		m_gradient(i) -= linear(i);
		m_gradient(i) += newValue;
		m_problem.linear(i) = newValue;
	}
	
	double checkKKT()const{
		double maxViolation = 0.0;
		for(std::size_t i = 0; i != dimensions(); ++i){
			if(isDeactivated(i)) continue;
			if(!isUpperBound(i)){
				maxViolation = std::max(maxViolation, gradient(i));
			}
			if(!isLowerBound(i)){
				maxViolation = std::max(maxViolation, -gradient(i));
			}
		}
		return maxViolation;
	}

protected:
	SVMProblem& m_problem;

	/// gradient of the objective function at the current alpha
	RealVector m_gradient;	

	std::size_t m_active;

	std::vector<char> m_alphaStatus;

	void updateAlphaStatus(std::size_t i){
		SIZE_CHECK(i < dimensions());
		m_alphaStatus[i] = AlphaFree;
		if(m_problem.alpha(i) == boxMax(i))
			m_alphaStatus[i] |= AlphaUpperBound;
		if(m_problem.alpha(i) == boxMin(i))
			m_alphaStatus[i] |= AlphaLowerBound;
	}
	
	bool testShrinkVariable(std::size_t a, double largestUp, double smallestDown)const{
		smallestDown = std::min(smallestDown, 0.0);
		largestUp = std::max(largestUp, 0.0);
		if (
			( isLowerBound(a) && gradient(a) < smallestDown)
			|| ( isUpperBound(a) && gradient(a) >largestUp)
		){
			// In this moment no feasible step including this variable
			// can improve the objective. Thus deactivate the variable.
			return true;
		}
		return false;
	}
};

template<class Problem>
class BoxConstrainedShrinkingProblem: public BoxBasedShrinkingStrategy<BoxConstrainedProblem<Problem> >{
public:
	BoxConstrainedShrinkingProblem(Problem& problem, bool shrink=true)
	:BoxBasedShrinkingStrategy<BoxConstrainedProblem<Problem> >(problem,shrink){}
};

}
#endif
