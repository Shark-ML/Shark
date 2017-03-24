/*!
 * 
 *
 * \brief       -
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
#ifndef SHARK_ALGORITHMS_QP_SVMPROBLEMS_H
#define SHARK_ALGORITHMS_QP_SVMPROBLEMS_H

#include <shark/Algorithms/QP/BoxConstrainedProblems.h>

namespace shark{
 
// Working-Set-Selection-Criteria are applied as follows:
// Criterium crit;
// value = crit(problem, i, j);
	
	
///\brief Computes the most violating pair of the problem
///
/// The MVP is defined as the set of indices (i,j)
/// such that g(i) - g(j) is maximal and alpha_i < max_j
/// and alpha_j > min_j.
struct MVPSelectionCriterion{
	/// \brief Select the most violating pair (MVP)
	///
	/// \return maximal KKT violation
	/// \param problem the SVM problem to select the working set for
	/// \param i  first working set component
	/// \param j  second working set component
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j)
	{
		double largestUp = -1e100;
		double smallestDown = 1e100;

		for (std::size_t a=0; a < problem.active(); a++)
		{
			double ga = problem.gradient(a);
			if (!problem.isUpperBound(a))
			{
				if (ga > largestUp)
				{
					largestUp = ga;
					i = a;
				}
			}
			if (!problem.isLowerBound(a))
			{
				if (ga < smallestDown)
				{
					smallestDown = ga;
					j = a;
				}
			}
		}

		// MVP stopping condition
		return largestUp - smallestDown;
	}
	
	void reset(){}
};


///\brief Computes the maximum gian solution
///
/// The first index is chosen based on the maximum gradient (first index of MVP)
/// and the second index is chosen such that the step of the corresponding alphas 
/// produces the largest gain.
struct LibSVMSelectionCriterion{
	
	/// \brief Select a working set according to the second order algorithm of LIBSVM 2.8
	///
	/// \return maximal KKT vioation
	/// \param problem the svm problem to select the working set for
	/// \param i  first working set component
	/// \param j  second working set component
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j)
	{
		i = 0;
		j = 1;

		double smallestDown = 1e100;
		double largestUp = -1e100;

		for (std::size_t a=0; a < problem.active(); a++)
		{
			double ga = problem.gradient(a);
			//if (aa < problem.boxMax(a))
			if (!problem.isUpperBound(a))
			{
				if (ga > largestUp)
				{
					largestUp = ga;
					i = a;
				}
			}
		}
		if (largestUp == -1e100) return 0.0;

		// find the second index using second order information
		typename Problem::QpFloatType* q = problem.quadratic().row(i, 0, problem.active());
		double best = 0.0;
		for (std::size_t a = 0; a < problem.active(); a++){
			double ga = problem.gradient(a);
			if (!problem.isLowerBound(a))
			{
				smallestDown=std::min(smallestDown,ga);
				
				double gain = detail::maximumGainQuadratic2DOnLine(
					problem.diagonal(i),
					problem.diagonal(a),
					q[a],
					largestUp,ga
				);
				if (gain> best)
				{
					best = gain;
					j = a;
				}
			}
		}

		if (best == 0.0) return 0.0;		// numerical accuracy reached :(

		// MVP stopping condition
		return largestUp - smallestDown;
	}
	
	void reset(){}
};

class HMGSelectionCriterion{
public:
	HMGSelectionCriterion():useLibSVM(true),smallProblem(false){}

	/// \brief Select a working set according to the hybrid maximum gain (HMG) algorithm
	///
	/// \return maximal KKT vioation
	/// \param problem the svm problem to select the working set for
	/// \param i  first working set component
	/// \param j  second working set component
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j)
	{
		if (smallProblem || useLibSVM || isInCorner(problem))
		{
			useLibSVM = false;
			if(!smallProblem && sqr(problem.active()) < problem.quadratic().getMaxCacheSize())
				smallProblem = true;
			LibSVMSelectionCriterion libSVMSelection;
			double value = libSVMSelection(problem,i, j);
			last_i = i;
			last_j = j;
			return value;
		}
		//old HMG
		MGStep besti = selectMGVariable(problem,last_i);
		if(besti.violation == 0.0)
			return 0;
		MGStep bestj = selectMGVariable(problem,last_j);
		
		if(bestj.gain > besti.gain){
			i = last_j;
			j = bestj.index;
		}else{
			i = last_i;
			j = besti.index;
		}
		if (problem.gradient(i) < problem.gradient(j)) 
			std::swap(i, j);
		last_i = i;
		last_j = j;
		return besti.violation;
	}
	
	void reset(){
		useLibSVM = true;
		smallProblem = false;
	}
	
private:
	template<class Problem>
	bool isInCorner(Problem const& problem)const{
		double Li = problem.boxMin(last_i);
		double Ui = problem.boxMax(last_i);
		double Lj = problem.boxMin(last_j);
		double Uj = problem.boxMax(last_j);
		double eps_i = 1e-8 * (Ui - Li);
		double eps_j = 1e-8 * (Uj - Lj);
		
		if ((problem.alpha(last_i) <= Li + eps_i || problem.alpha(last_i) >= Ui - eps_i)
		&& ((problem.alpha(last_j) <= Lj + eps_j || problem.alpha(last_j) >= Uj - eps_j)))
		{
			return true;
		}
		return false;
	}
	struct MGStep{
		std::size_t index;//index of variable
		double violation;//computed gradientValue
		double gain;
	};
	template<class Problem>
	MGStep selectMGVariable(Problem& problem,std::size_t i) const{
		
		//best variable pair found so far
		std::size_t bestIndex = 0;//index of variable
		double bestGain = 0;
		
		double largestUp = -1e100;
		double smallestDown = 1e100;

		// try combinations with b = old_i
		typename Problem::QpFloatType* q = problem.quadratic().row(i, 0, problem.active());
		double ab = problem.alpha(i);
		double db = problem.diagonal(i);
		double Lb = problem.boxMin(i);
		double Ub = problem.boxMax(i);
		double gb = problem.gradient(i);
		for (std::size_t a = 0; a < problem.active(); a++)
		{
			double ga = problem.gradient(a);
			
			if (!problem.isUpperBound(a))
				largestUp = std::max(largestUp,ga);
			if (!problem.isLowerBound(a))
				smallestDown = std::min(smallestDown,ga);
			
			if (a == i) continue;
			//get maximum unconstrained step length
			double denominator = (problem.diagonal(a) + db - 2.0 * q[a]);
			double mu_max = (ga - gb) / denominator;
			
			//check whether a step > 0 is possible at all
			//~ if( mu_max > 0 && ( problem.isUpperBound(a) || problem.isLowerBound(b)))continue;
			//~ if( mu_max < 0 && ( problem.isLowerBound(a) || problem.isUpperBound(b)))continue;
			
			//constraint step to box
			double aa = problem.alpha(a);
			double La = problem.boxMin(a);
			double Ua = problem.boxMax(a);
			double mu_star = mu_max;
			if (aa + mu_star < La) mu_star = La - aa;
			else if (mu_star + aa > Ua) mu_star = Ua - aa;
			if (ab - mu_star < Lb) mu_star = ab - Lb;
			else if (ab - mu_star > Ub) mu_star = ab - Ub;

			double gain = mu_star * (2.0 * mu_max - mu_star) * denominator;
			
			// select the largest gain
			if (gain > bestGain)
			{
				bestGain = gain;
				bestIndex = a;
			}
		}
		MGStep step;
		step.violation= largestUp-smallestDown;
		step.index = bestIndex;
		step.gain=bestGain;
		return step;
		
	}
	
	std::size_t last_i;
	std::size_t last_j;
	
	bool useLibSVM;
	bool smallProblem;
};


template<class Problem>
class SvmProblem{
public:
	typedef typename Problem::QpFloatType QpFloatType;
	typedef typename Problem::MatrixType MatrixType;
	typedef LibSVMSelectionCriterion PreferedSelectionStrategy;

	SvmProblem(Problem& problem)
	: m_problem(problem)
	, m_gradient(problem.linear)
	, m_active(problem.dimensions())
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
	void updateSMO(std::size_t i, std::size_t j){
		SIZE_CHECK(i < active());
		SIZE_CHECK(j < active());
		// get the matrix row corresponding to the first variable of the working set
		QpFloatType* qi = quadratic().row(i, 0, active());

		// solve the sub-problem defined by i and j
		double numerator = gradient(i) - gradient(j);
		double denominator = diagonal(i) + diagonal(j) - 2.0 * qi[j];
		denominator =  std::max(denominator,1.e-12);
		double step = numerator/denominator;
			
		//update alpha in a numerically stable way
		// do the update of the alpha values carefully - avoid numerical problems
		double Ui = boxMax(i);
		double Lj = boxMin(j);
		double aiOld = m_problem.alpha(i);
		double ajOld = m_problem.alpha(j);
		double& ai = m_problem.alpha(i);
		double& aj = m_problem.alpha(j);
		if (step >= std::min(Ui - ai, aj - Lj))
		{
			if (Ui - ai > aj - Lj)
			{
				step = aj - Lj;
				ai += step;
				aj = Lj;
			}
			else if (Ui - ai < aj - Lj)
			{
				step = Ui - ai;
				ai = Ui;
				aj -= step;
			}
			else
			{
				step = Ui - ai;
				ai = Ui;
				aj = Lj;
			}
		}
		else
		{
			ai += step;
			aj -= step;
		}
		
		if(ai == aiOld && aj == ajOld)return;
		
		//Update internal data structures (gradient and alpha status)
		QpFloatType* qj = quadratic().row(j, 0, active());
		for (std::size_t a = 0; a < active(); a++) 
			m_gradient(a) -= step * qi[a] - step * qj[a];
		
		//update boundary status
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
	
	///\brief Remove the i-th example from the problem while taking the equality constraint into account.
	///
	/// The i-th element is first set to zero and as well as an unspecified set corrected so
	/// that the constraint is fulfilled.
	void deactivateVariable(std::size_t i){
		SIZE_CHECK(i < dimensions());
		//we need to correct for the equality constraint
		//that means we have to move enough variables to satisfy the constraint again.
		for (std::size_t j=0; j<dimensions(); j++){
			if (j == i || m_alphaStatus[j] == AlphaDeactivated) continue;
			//propose the maximum step possible and let applyStep cut it down.
			applyStep(i,j, -alpha(i));
			if(alpha(i) == 0.0) break;
		}
		m_alphaStatus[i] = AlphaDeactivated;
	}
	///\brief Reactivate an previously deactivated variable.
	void activateVariable(std::size_t i){
		SIZE_CHECK(i < dimensions());
		m_alphaStatus[i] = AlphaFree;
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
	
	/// \brief Scales all box constraints by a constant factor and adapts the solution using a separate scaling
	void scaleBoxConstraints(double factor, double variableScalingFactor){
		m_problem.scaleBoxConstraints(factor,variableScalingFactor);
		for(std::size_t i = 0; i != this->dimensions(); ++i){
			//don't change deactivated variables
			if(m_alphaStatus[i] == AlphaDeactivated) continue;
			m_gradient(i) -= linear(i);
			m_gradient(i) *= variableScalingFactor;
			m_gradient(i) += linear(i);
			updateAlphaStatus(i);
		}
	}
	
	/// \brief adapts the linear part of the problem and updates the internal data structures accordingly.
	virtual void setLinear(std::size_t i, double newValue){
		m_gradient(i) -= linear(i);
		m_gradient(i) += newValue;
		m_problem.linear(i) = newValue;
	}
	
	double checkKKT()const{
		double largestUp = -1e100;
		double smallestDown = 1e100;
		for (std::size_t a = 0; a < active(); a++){
			if (!isLowerBound(a))
				smallestDown = std::min(smallestDown,gradient(a));
			if (!isUpperBound(a))
				largestUp = std::max(largestUp,gradient(a));
		}
		return largestUp - smallestDown;
	}

protected:
	Problem m_problem;

	/// gradient of the objective function at the current alpha
	RealVector m_gradient;	

	std::size_t m_active; 

	/// \brief Stores the status, whther alpha is on the lower or upper bound, or whether it is free.
	std::vector<char> m_alphaStatus;

	///\brief Update the problem by a proposed step i taking the box constraints into account.
	///
	/// A step length 0<=lambda<=1 is found so that 
	/// boxMin(i) <= alpha(i)+lambda*step <= boxMax(i) 
	/// and
	/// boxMin(j) <= alpha(j)-lambda*step <= boxMax(j)
	/// the update is performed in a numerically stable way and the internal data structures
	/// are also updated.
	virtual void applyStep(std::size_t i, std::size_t j, double step){
	        SIZE_CHECK(i < active());
	        SIZE_CHECK(j < active());
	        // do the update of the alpha values carefully - avoid numerical problems
	        double Ui = boxMax(i);
	        double Lj = boxMin(j);
	        double aiOld = m_problem.alpha(i);
	        double ajOld = m_problem.alpha(j);
	        double& ai = m_problem.alpha(i);
	        double& aj = m_problem.alpha(j);
	        if (step >= std::min(Ui - ai, aj - Lj))
	        {
	                if (Ui - ai > aj - Lj)
	                {
	                        step = aj - Lj;
	                        ai += step;
	                        aj = Lj;
	                }
	                else if (Ui - ai < aj - Lj)
	                {
	                        step = Ui - ai;
	                        ai = Ui;
	                        aj -= step;
	                }
	                else
	                {
	                        step = Ui - ai;
	                        ai = Ui;
	                        aj = Lj;
	                }
	        }
	        else
	        {
	                ai += step;
	                aj -= step;
	        }
	        
	        if(ai == aiOld && aj == ajOld)return;
	        
	        //Update internal data structures (gradient and alpha status)
	        QpFloatType* qi = quadratic().row(i, 0, active());
	        QpFloatType* qj = quadratic().row(j, 0, active());
	        for (std::size_t a = 0; a < active(); a++) 
	                m_gradient(a) -= step * qi[a] - step * qj[a];
	        
	        //update boundary status
	        updateAlphaStatus(i);
	        updateAlphaStatus(j);
	}

	
	void updateAlphaStatus(std::size_t i){
		SIZE_CHECK(i < dimensions());
		m_alphaStatus[i] = AlphaFree;
		if(m_problem.alpha(i) == boxMax(i))
			m_alphaStatus[i] |= AlphaUpperBound;
		if(m_problem.alpha(i) == boxMin(i))
			m_alphaStatus[i] |= AlphaLowerBound;
	}
	
	bool testShrinkVariable(std::size_t a, double largestUp, double smallestDown)const{
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
class SvmShrinkingProblem: public BoxBasedShrinkingStrategy<SvmProblem<Problem> >{
public:
	SvmShrinkingProblem(Problem& problem, bool shrink=true)
	:BoxBasedShrinkingStrategy<SvmProblem<Problem> >(problem,shrink){}
};

}
#endif
