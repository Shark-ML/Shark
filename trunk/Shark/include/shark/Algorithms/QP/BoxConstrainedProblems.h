/*!
 *
 *  \author  T. Glasmachers, O.Krause
 *  \date    2013
 *
 *  \par Copyright (c) 1999-2012:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
 #ifndef SHARK_ALGORITHMS_QP_BOXCONSTRAINEDPROBLEMS_H
#define SHARK_ALGORITHMS_QP_BOXCONSTRAINEDPROBLEMS_H
 
#include <shark/Algorithms/QP/QpSolver.h>
 
 namespace shark{
 
struct MaximumGradientCriterium{
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j){
		double largestGradient = 0;

		for (std::size_t a = 0; a < problem.active(); a++){
			double v = problem.alpha(a);
			double g = problem.gradient(a);
			if (v < problem.boxMax(a)){
				if (g > largestGradient){
					largestGradient = g;
					i = a;
				}
			}
			if (v > problem.boxMin(a)){
				if (-g > largestGradient){
					largestGradient = -g;
					i = a;
				}
			}
		}
		j=i;//we choose only a i-d working set

		return largestGradient;
	}
	
	void reset(){}
};


struct MaximumGainCriterium{
	template<class Problem>
	double operator()(Problem& problem, std::size_t& i, std::size_t& j){
		//choose first variable by first order criterium
		MaximumGradientCriterium firstOrder;
		double maxGrad = firstOrder(problem,i,j);
		if (maxGrad == 0.0) return maxGrad;

		double gi = problem.gradient(i);
		typename Problem::QpFloatType* q = problem.quadratic().row(i, 0, problem.active());
		double Qii = problem.diagonal(i);

		// select second variable j with second order method
		double maxGain = 0.0;
		for (std::size_t a=0; a<problem.active(); a++)
		{
			if (a == i) continue;
			double aa = problem.alpha(a);
			double ga = problem.gradient(a);
			if ((aa > problem.boxMin(a) && ga < 0.0) 
			|| (aa < problem.boxMax(a) && ga > 0.0)){
				double Qia = q[a];
				double Qaa = problem.diagonal(a);

				double QD = Qii * Qaa;
				double detQ = QD - Qia * Qia;
				if (detQ < 1e-10 * QD)
				{
					if (Qii == 0.0 && Qaa == 0.0)
					{
						// Q has rank zero
						if (gi != 0.0 || ga != 0.0)
						{
							j = a;
							return maxGrad;		// infinite gain, return immediately
						}
					}
					else
					{
						// Q has rank one
						if (Qii * ga - Qia * gi != 0.0)
						{
							j = a;
							return maxGrad;		// infinite gain, return immediately
						}
						else
						{
							double g2 = ga*ga + gi*gi;
							double gain = (g2*g2) / (ga*ga*Qaa + 2.0*ga*gi*Qia + gi*gi*Qii);
							if (gain > maxGain)
							{
								maxGain = gain;
								j = a;
							}
						}
					}
				}
				else
				{
					// Q has rank two
					double gain = (ga*ga*Qii - 2.0*ga*gi*Qia + gi*gi*Qaa) / detQ;
					if (gain > maxGain)
					{
						maxGain = gain;
						j = a;
					}
				}
			}
		}

		return maxGrad;		// solution is not optimal
	}
	
	void reset(){}
};

template<class SVMProblem>
class BoxConstrainedProblem{
public:
	typedef typename SVMProblem::QpFloatType QpFloatType;
	typedef typename SVMProblem::MatrixType MatrixType;
	typedef MaximumGainCriterium PreferedSelectionStrategy;

	BoxConstrainedProblem(SVMProblem& problem)
	: m_problem(problem)
	, m_gradient(problem.linear)
	, m_active (problem.dimensions()){
		//compute the gradient if alpha != 0
		for (std::size_t i=0; i != dimensions(); i++){
			double v = alpha(i);
			if (v != 0.0){
				QpFloatType* q = quadratic().row(i, 0, dimensions());
				for (std::size_t a=0; a < dimensions(); a++) 
					m_gradient(a) -= q[a] * v;
			}
		}
	}
	std::size_t dimensions()const{
		return m_problem.dimensions();
	}
	
	std::size_t active()const{
		return m_active;
	}
	
	double boxMin(std::size_t i)const{
		return m_problem.boxMin(i);
	}
	double boxMax(std::size_t i)const{
		return m_problem.boxMax(i);
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
	
	RealVector getUnpermutedAlpha()const{
		RealVector alpha(dimensions());
		for (std::size_t i=0; i<dimensions(); i++) 
			alpha(m_problem.permutation[i]) = m_problem.alpha(i);
		return alpha;
	}
	
	///\brief Does an update of SMO given a working set with indices i and j.
	void updateSMO(std::size_t i, std::size_t j){
		
		if(i == j){//both variables are identical, thus solve the 1-d problem.
			double Li = boxMin(i);
			double Ui = boxMax(i);

			// get the matrix row corresponding to the working set
			QpFloatType* q = quadratic().row(i, 0, active());

			// update alpha, that is, solve the sub-problem defined by i
			double numerator = gradient(i);
			double denominator = diagonal(i);
			double mu = numerator / denominator;

			mu = boundedUpdate(m_problem.alpha(i),mu,Li,Ui);
			// update the gradient
			for (std::size_t a = 0; a < active(); a++) 
				m_gradient(a) -= mu * q[a];
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
		double mu_i = 0.0;
		double mu_j = 0.0;
		solve2DBox(alpha(i), alpha(j),
			m_gradient(i), m_gradient(j),
			diagonal(i), qi[j], diagonal(j),
			Li, Ui, Lj, Uj,
			mu_i, mu_j
		);
		
		mu_i= boundedUpdate(m_problem.alpha(i),mu_i,Li,Ui);
		mu_j= boundedUpdate(m_problem.alpha(j),mu_j,Lj,Uj);

		// update the gradient
		for (std::size_t a = 0; a < active(); a++) 
			m_gradient(a) -= mu_i * qi[a] + mu_j * qj[a];
	}
	
	///\brief Returns the current function value of the problem.
	double functionValue()const{
		return 0.5*inner_prod(m_gradient+m_problem.linear,m_problem.alpha);
	}
	
	bool shrink(double){return false;}
	void reshrink(){}
	void unshrink(){}
		
	void modifyStep(std::size_t i, std::size_t j, double diff){
		SIZE_CHECK(i < dimensions());
		SIZE_CHECK(i == j );
		RANGE_CHECK(alpha(i)+diff >= boxMin(i)-1.e-14*(boxMax(i)-boxMin(i)));//we allow a bit of numeric error
		RANGE_CHECK(alpha(i)+diff <= boxMax(i)+1.e-14*(boxMax(i)-boxMin(i)));
		if(diff == 0) return;

		boundedUpdate(m_problem.alpha(i),diff,boxMin(i),boxMax(i));
		//update gradient
		QpFloatType* q = quadratic().row(i, 0, active());
		for (std::size_t a = 0; a < active(); a++) 
			m_gradient(a) -=diff * q[a];
	}
	
protected:
	SVMProblem& m_problem;

	/// gradient of the objective function at the current alpha
	RealVector m_gradient;	

	std::size_t m_active;

	/// Internally used by Solve2D;
	/// computes the solution of a
	/// one-dimensional sub-problem.
	double stepEdge(double alpha, double g, double Q, double L, double U, double& mu)
	{
		// compute the optimal unconstrained step
		double muHat = g / Q;

		// check for numerical problems
		if( !boost::math::isnormal( muHat ) )
		{
			if (g > 0.0) mu = 1e100;
			else mu = -1e100;
			return 1e100;
		}

		// compute the optimal constrained step
		double mu_g;
		if (muHat <= L - alpha)
		{
			mu_g = L - alpha;
			mu = -1e100;
		}
		else if (muHat >= U - alpha)
		{
			mu_g = U - alpha;
			mu = 1e100;
		}
		else
		{
			mu_g = muHat;
			mu = muHat;
		}

		// compute (twice) the gain
		double deltaMu = muHat - mu_g;
		return (muHat * muHat - deltaMu * deltaMu) * Q;
	}

	/// Exact solver for a two-dimensional sub-problem.
	/// If the optimal solution is on the edge, then the
	/// corresponding mu-value is set to either +1e100
	/// or -1e100 as an indication.
	void solve2DBox(
		double alphai, double alphaj,
		double gi, double gj,
		double Qii, double Qij, double Qjj,
		double Li, double Ui, double Lj, double Uj,
		double& mui, double& muj
	){
		double QD = Qii * Qjj;
		double detQ = QD - Qij * Qij;
		if (detQ < 1e-10 * QD)
		{
			if (Qii == 0.0 && Qjj == 0.0)
			{
				// Q has rank zero (is the zero matrix)
				// just follow the gradient
				if (gi > 0.0) mui = 1e100;
				else if (gi < 0.0) mui = -1e100;
				else mui = 0.0;
				if (gj > 0.0) muj = 1e100;
				else if (gj < 0.0) muj = -1e100;
				else muj = 0.0;
			}
			else
			{
				// Q has rank one
				double gamma = Qii * gj - Qij * gi;
				double edgei_mui = 0.0, edgei_muj = 0.0, edgei_gain = 0.0;
				double edgej_mui = 0.0, edgej_muj = 0.0, edgej_gain = 0.0;

				// edge with fixed mu_i
				if (Qij * gamma > 0.0)
				{
					edgei_mui = -1e100;
					edgei_gain = stepEdge(alphaj, gj - Qij * (Li - alphai), Qjj, Lj, Uj, edgei_muj);
				}
				else if (Qij * gamma < 0.0)
				{
					edgei_mui = 1e100;
					edgei_gain = stepEdge(alphaj, gj - Qij * (Ui - alphai), Qjj, Lj, Uj, edgei_muj);
				}

				// edge with fixed mu_j
				if (Qii * gamma < 0.0)
				{
					edgej_muj = -1e100;
					edgej_gain = stepEdge(alphai, gi - Qij * (Lj - alphaj), Qii, Li, Ui, edgej_mui);
				}
				else if (Qii * gamma > 0.0)
				{
					edgej_muj = 1e100;
					edgej_gain = stepEdge(alphai, gi - Qij * (Uj - alphaj), Qii, Li, Ui, edgej_mui);
				}

				// keep the better edge point
				if (edgei_gain > edgej_gain)
				{
					mui = edgei_mui;
					muj = edgei_muj;
				}
				else
				{
					mui = edgej_mui;
					muj = edgej_muj;
				}
			}
		}
		else
		{
			// Q has full rank of two, thus it is invertible
			double muiHat = (Qjj * gi - Qij * gj) / detQ;
			double mujHat = (Qii * gj - Qij * gi) / detQ;
			double edgei_mui = 0.0, edgei_muj = 0.0, edgei_gain = 0.0;
			double edgej_mui = 0.0, edgej_muj = 0.0, edgej_gain = 0.0;

			// edge with fixed mu_i
			if (muiHat < Li - alphai)
			{
				edgei_mui = -1e100;
				edgei_gain = stepEdge(alphaj, gj - Qij * (Li - alphai), Qjj, Lj, Uj, edgei_muj);
			}
			else if (muiHat > Ui - alphai)
			{
				edgei_mui = 1e100;
				edgei_gain = stepEdge(alphaj, gj - Qij * (Ui - alphai), Qjj, Lj, Uj, edgei_muj);
			}

			// edge with fixed mu_j
			if (mujHat < Lj - alphaj)
			{
				edgej_muj = -1e100;
				edgej_gain = stepEdge(alphai, gi - Qij * (Lj - alphaj), Qii, Li, Ui, edgej_mui);
			}
			else if (mujHat > Uj - alphaj)
			{
				edgej_muj = 1e100;
				edgej_gain = stepEdge(alphai, gi - Qij * (Uj - alphaj), Qii, Li, Ui, edgej_mui);
			}

			// keep the unconstrained optimum or the better edge point
			if (edgei_gain == 0.0 && edgej_gain == 0.0)
			{
				mui = muiHat;
				muj = mujHat;
			}
			else if (edgei_gain > edgej_gain)
			{
				mui = edgei_mui;
				muj = edgei_muj;
			}
			else
			{
				mui = edgej_mui;
				muj = edgej_muj;
			}
		}
	}
};

template<class SVMProblem>
struct BoxConstrainedShrinkingProblem
: public BaseShrinkingProblem<BoxConstrainedProblem<SVMProblem> >{
	typedef BaseShrinkingProblem<BoxConstrainedProblem<SVMProblem> > base_type;
	static const std::size_t IterationsBetweenShrinking;
	
	BoxConstrainedShrinkingProblem(SVMProblem& problem, bool shrink = true)
	: base_type(problem,shrink)
	, m_isUnshrinked(false)
	, m_shrinkCounter(std::min(this->dimensions(),IterationsBetweenShrinking)){}
	
protected:
	void doShrink(double epsilon){
		//check if shrinking is necessary
		--m_shrinkCounter;
		if(m_shrinkCounter != 0) return;
		m_shrinkCounter = std::min(this->active(),IterationsBetweenShrinking);
		
		double largestUp;
		double smallestDown;
		getMaxKKTViolations(largestUp,smallestDown,this->active());

		// check whether unshrinking is necessary at this accuracy level
		// to prevent that a shrinking error invalidates
		// the fine grained late optimization steps
		if (! m_isUnshrinked && (largestUp - smallestDown < 10.0 * epsilon))
		{
			m_isUnshrinked = true;
			this->reshrink();
			return;
		}
		doShrink(largestUp,smallestDown);
	}

	/// \brief Unshrink the problem and immdiately reshrink it.
	void doReshrink(){
		if (this->active() == this->dimensions()) return;
		
		this->unshrink();
		
		// shrink directly again
		double largestUp;
		double smallestDown;
		getMaxKKTViolations(largestUp,smallestDown,this->dimensions());
		doShrink(largestUp,smallestDown);
		
		m_shrinkCounter = std::min(this->active(),IterationsBetweenShrinking);
	}

private:
	void doShrink(double largestUp, double smallestDown){
		for (int a = this->active()-1; a >= 0; --a){
			if(testShrinkVariable(a,largestUp,smallestDown))
				this->shrinkVariable(a);
		}
	}

	bool testShrinkVariable(std::size_t a, double largestUp, double smallestDown)const{
		double v = this->alpha(a);
		double g = this->gradient(a);

		if (
			( g <= smallestDown && v == this->boxMin(a))
			|| ( g >=largestUp && v == this->boxMax(a))
		){
			// In this moment no feasible step including this variable
			// can improve the objective. Thus deactivate the variable.
			return true;
		}
		return false;
	}
	
	void getMaxKKTViolations(double& largestUp, double& smallestDown, std::size_t maxIndex){
		largestUp = -1e100;
		smallestDown = 1e100;
		for (std::size_t a = 0; a < maxIndex; a++)
		{
			double v = this->alpha(a);
			double g = this->gradient(a);
			if (v > this->boxMin(a))
				smallestDown = std::min(smallestDown,g);
			if (v < this->boxMax(a))
				largestUp = std::max(largestUp,g);
		}
	}
	
	/// true if the problem has already been unshrinked
	bool m_isUnshrinked;

	std::size_t m_shrinkCounter;
};
template<class SVMProblem>
const std::size_t BoxConstrainedShrinkingProblem<SVMProblem>::IterationsBetweenShrinking = 1000;


}
#endif