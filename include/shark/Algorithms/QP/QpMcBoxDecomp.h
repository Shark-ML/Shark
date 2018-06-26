//===========================================================================
/*!
 * 
 *
 * \brief       Quadratic programming m_problem for multi-class SVMs
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2007-2012
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


#ifndef SHARK_ALGORITHMS_QP_QPMCBOXDECOMP_H
#define SHARK_ALGORITHMS_QP_QPMCBOXDECOMP_H

#include <shark/Algorithms/QP/QpSolver.h>
#include <shark/Algorithms/QP/QpSparseArray.h>
#include <shark/Algorithms/QP/Impl/AnalyticProblems.h>
#include <shark/Core/Timer.h>
#include <shark/Data/Dataset.h>


namespace shark {

template <class Matrix>
class QpMcBoxDecomp
{
public:
	typedef typename Matrix::QpFloatType QpFloatType;
	/// \brief Working set selection eturning th S2DO working set
	///
	/// This selection operator picks the first variable by maximum gradient, 
	/// the second by maximum unconstrained gain.
	struct PreferedSelectionStrategy{
		template<class Problem>
		double operator()(Problem& problem, std::size_t& i, std::size_t& j){
			//todo move implementation here
			return problem.selectWorkingSet(i,j);
		}

		void reset(){}
	};

	///Constructor
	///\param  kernel               kernel matrix - cache or pre-computed matrix
	///\param  M                   kernel modifiers in the format \f$ M_(y_i, p, y_j, q) = _M(classes*(y_i*|P|+p_i)+y_j, q) \f$
	///\param  target the target labels for the variables
	///\param linearMat the linear part of the problem
	///\param C upper bound for all box variables, lower bound is 0.
	QpMcBoxDecomp(
		Matrix& kernel,
		QpSparseArray<QpFloatType> const& M,
		Data<unsigned int> const& target,
		RealMatrix const& linearMat,
		double C
	)
	: bUnshrinked(false)
	, m_kernelMatrix(kernel)
	, m_M(M)
	, m_C(C)
	, m_classes(numberOfClasses(target))
	, m_cardP(linearMat.size2())
	, m_numExamples(kernel.size())
	, m_numVariables(m_cardP * m_numExamples)
	, m_linear(m_numVariables)
	, m_alpha(m_numVariables,0.0)
	, m_gradient(m_numVariables)
	, m_examples(m_numExamples)
	, m_variables(m_numVariables)
	, m_storage1(m_numVariables)
	, m_storage2(m_numVariables)
	, m_useShrinking(true)
	{
		SHARK_RUNTIME_CHECK(target.numberOfElements() == kernel.size(), "Size of kernel matrix and target vector do not agree.");
		SHARK_RUNTIME_CHECK(kernel.size() == linearMat.size1(), "Size of kernel matrix and linear factor to not agree.");
		
		// prepare m_problem internal variables
		m_activeEx = m_numExamples;
		m_activeVar = m_numVariables;
		std::size_t i = 0;
		std::size_t v = 0;
		for(auto y: elements(target)){
			m_examples[i].index = i;
			m_examples[i].y = y;
			m_examples[i].active = m_cardP;
			m_examples[i].var = &m_storage1[m_cardP * i];
			m_examples[i].avar = &m_storage2[m_cardP * i];
			double k = m_kernelMatrix.entry(i, i);
			for (unsigned int p=0; p<m_cardP; p++, v++)
			{
				m_variables[v].i = i;
				m_variables[v].p = p;
				m_variables[v].index = p;
				double Q = m_M(m_classes * (y * m_cardP + p) + y, p) * k;
				m_variables[v].diagonal = Q;
				m_storage1[v] = v;
				m_storage2[v] = v;
				
				m_linear(v) = m_gradient(v) = linearMat(i,p);
			}
			++i;
		}
	}
	
	///enable/disable shrinking
	void setShrinking(bool shrinking = true)
	{
		m_useShrinking = shrinking;
	}
	
	/// \brief Return the solution found.
	RealMatrix solution() const{
		RealMatrix solutionMatrix(m_numVariables,m_cardP,0);
		for (std::size_t v=0; v<m_numVariables; v++)
		{
			solutionMatrix(originalIndex(v),m_variables[v].p) = m_alpha(v);
		}
		return solutionMatrix;
	}
	
	double alpha(std::size_t i, std::size_t p)const{
		return m_alpha(m_cardP * i + p);
	}
	/// \brief Return the gradient of the solution.
	RealMatrix solutionGradient() const{
		RealMatrix solutionGradientMatrix(m_numVariables,m_cardP,0);
		for (std::size_t v=0; v<m_numVariables; v++)
		{
			solutionGradientMatrix(originalIndex(v),m_variables[v].p) = m_gradient(v);
		}
		return solutionGradientMatrix;
	}
	
	/// \brief Compute the objective value of the current solution.
	double functionValue()const{
		return 0.5*inner_prod(m_gradient+m_linear,m_alpha);
	}
	
	unsigned int label(std::size_t i){
		return m_examples[i].y;
	}
	
	std::size_t dimensions()const{
		return m_numVariables;
	}
	std::size_t cardP()const{
		return m_cardP;
	}
	
	std::size_t getNumExamples()const{
		return m_numExamples;
	}
	
	///return the largest KKT violation
	double checkKKT()const
	{
		double maxViolation = 0.0;
		for (std::size_t v=0; v<m_activeVar; v++)
		{
			double a = m_alpha(v);
			double g = m_gradient(v);
			if (a < m_C)
			{
				maxViolation = std::max(maxViolation,g);
			}
			if (a > 0.0)
			{
				maxViolation = std::max(maxViolation,-g);
			}
		}
		return maxViolation;
	}
	
	/// \brief change the linear part of the problem by some delta
	void addDeltaLinear(RealMatrix const& deltaLinear){
		SIZE_CHECK(deltaLinear.size1() == m_numExamples);
		SIZE_CHECK(deltaLinear.size2() == m_cardP);
		for (std::size_t v=0; v<m_numVariables; v++)
		{
			
			std::size_t p = m_variables[v].p;
			m_gradient(v) += deltaLinear(originalIndex(v),p); 
			m_linear(v) += deltaLinear(originalIndex(v),p);
		}
	}
	
	void updateSMO(std::size_t v, std::size_t w){
		SIZE_CHECK(v < m_activeVar);
		SIZE_CHECK(w < m_activeVar);
		// update
		if (v == w)
		{
			// Limit case of a single variable;
			// this means that there is only one
			// non-optimal variables left.
			std::size_t i = m_variables[v].i;
			SHARK_ASSERT(i < m_activeEx);
			unsigned int p = m_variables[v].p;
			unsigned int y = m_examples[i].y;
			std::size_t r = m_cardP * y + p;
			QpFloatType* q = m_kernelMatrix.row(i, 0, m_activeEx);
			double Qvv = m_variables[v].diagonal;
			double mu = -m_alpha(v);
			detail::solveQuadraticEdge(m_alpha(v),m_gradient(v),Qvv,0,m_C);
			mu+=m_alpha(v);
			gradientUpdate(r, mu, q);
		}
		else
		{
			// S2DO
			std::size_t iv = m_variables[v].i;
			SHARK_ASSERT(iv < m_activeEx);
			unsigned int pv = m_variables[v].p;
			unsigned int yv = m_examples[iv].y;

			std::size_t iw = m_variables[w].i;
			SHARK_ASSERT(iw < m_activeEx);
			unsigned int pw = m_variables[w].p;
			unsigned int yw = m_examples[iw].y;

			// get the matrix rows corresponding to the working set
			QpFloatType* qv = m_kernelMatrix.row(iv, 0, m_activeEx);
			QpFloatType* qw = m_kernelMatrix.row(iw, 0, m_activeEx);
			std::size_t rv = m_cardP*yv+pv;
			std::size_t rw = m_cardP*yw+pw;

			// get the Q-matrix restricted to the working set
			double Qvv = m_variables[v].diagonal;
			double Qww = m_variables[w].diagonal;
			double Qvw = m_M(m_classes * rv + yw, pw) * qv[iw];

			// solve the sub-problem and update the gradient using the step sizes mu			
			double mu_v = -m_alpha(v);
			double mu_w = -m_alpha(w);
			detail::solveQuadratic2DBox(m_alpha(v), m_alpha(w),
				m_gradient(v), m_gradient(w),
				Qvv, Qvw, Qww,
				0, m_C, 0, m_C
			);
			mu_v += m_alpha(v);
			mu_w += m_alpha(w);
			
			gradientUpdate(rv, mu_v, qv);
			gradientUpdate(rw, mu_w, qw);
		}
	}
	
	///Shrink the problem
	bool shrink(double epsilon)
	{
		if(! m_useShrinking)
			return false;
		if (! bUnshrinked)
		{
			double largest = 0.0;
			for (std::size_t a = 0; a < m_activeVar; a++)
			{
				if (m_alpha(a) < m_C)
				{
					largest = std::max(largest,m_gradient(a));
				}
				if (m_alpha(a) > 0.0)
				{
					largest = std::max(largest,-m_gradient(a));
				}
			}
			if (largest < 10.0 * epsilon)
			{
				// unshrink the problem at this accuracy level
				unshrink();
				bUnshrinked = true;
			}
		}

		// shrink variables
		bool se = false;
		for (int a= (int)m_activeVar-1; a >= 0; a--)
		{
			double v = m_alpha(a);
			double g = m_gradient(a);

			if ((v == 0.0 && g <= 0.0) || (v == m_C && g >= 0.0))
			{
				// In this moment no feasible step including this variables
				// can improve the objective. Thus deactivate the variables.
				std::size_t e = m_variables[a].i;
				deactivateVariable(a);
				if (m_examples[e].active == 0)
				{
					se = true;
				}
			}
		}

		if (se)
		{
			// exchange examples such that shrinked examples
			// are moved to the ends of the lists
			for (int a = (int)m_activeEx - 1; a >= 0; a--)
			{
				if (m_examples[a].active == 0) 
					deactivateExample(a);
			}
		}
		return true;
	}

	///Activate all variables
	void unshrink()
	{
		if (m_activeVar == m_numVariables) return;

		// compute the inactive m_gradient components (quadratic time complexity)
		subrange(m_gradient, m_activeVar, m_numVariables) = subrange(m_linear, m_activeVar, m_numVariables);
		for (std::size_t v = 0; v != m_numVariables; v++)
		{
			double mu = m_alpha(v);
			if (mu == 0.0) continue;

			std::size_t iv = m_variables[v].i;
			unsigned int pv = m_variables[v].p;
			unsigned int yv = m_examples[iv].y;
			std::size_t r = m_cardP * yv + pv;
			std::vector<QpFloatType> q(m_numExamples);
			m_kernelMatrix.row(iv, 0, m_numExamples, &q[0]);

			for (std::size_t a = 0; a != m_numExamples; a++)
			{
				double k = q[a];
				Example& ex = m_examples[a];
				typename QpSparseArray<QpFloatType>::Row const& row = m_M.row(m_classes * r + ex.y);
				QpFloatType def = row.defaultvalue;
				for (std::size_t b=0; b<row.size; b++)
				{
					std::size_t f = ex.var[row.entry[b].index];
					if (f >= m_activeVar) 
						m_gradient(f) -= mu * (row.entry[b].value - def) * k;
				}
				if (def != 0.0)
				{
					double upd = mu * def * k;
					for (std::size_t  b=ex.active; b<m_cardP; b++)
					{
						std::size_t f = ex.avar[b];
						SHARK_ASSERT(f >= m_activeVar);
						m_gradient(f) -= upd;
					}
				}
			}
		}

		for (std::size_t  i=0; i<m_numExamples; i++) 
			m_examples[i].active = m_cardP;
		m_activeEx = m_numExamples;
		m_activeVar = m_numVariables;
	}
	
	//!
	///\brief select the working set
	//!
	///Select one or two numVariables for the sub-problem
	///and return the maximal KKT violation. The method
	///MAY select the same index for i and j. In that
	///case the working set consists of a single variables.
	///The working set may be invalid if the method reports
	///a KKT violation of zero, indicating optimality.
	double selectWorkingSet(std::size_t& i, std::size_t& j)
	{
		// box case
		double maxViolation = 0.0;

		// first order selection
		for (std::size_t a=0; a<m_activeVar; a++)
		{
			double aa = m_alpha(a);
			double ga = m_gradient(a);
			if (ga >maxViolation && aa < m_C)
			{
				maxViolation = ga;
				i = a;
			}
			else if (-ga > maxViolation && aa > 0.0)
			{
				maxViolation = -ga;
				i = a;
			}
		}
		if (maxViolation == 0.0) return maxViolation;

		// second order selection
		Variable& vari = m_variables[i];
		std::size_t ii = vari.i;
		SHARK_ASSERT(ii < m_activeEx);
		unsigned int pi = vari.p;
		unsigned int yi = m_examples[ii].y;
		double di = vari.diagonal;
		double gi = m_gradient(i);
		QpFloatType* k = m_kernelMatrix.row(ii, 0, m_activeEx);
		
		j = i;
		double bestgain = gi * gi / di;
		
		for (std::size_t a=0; a<m_activeEx; a++)
		{
			Example const& exa = m_examples[a];
			unsigned int ya = exa.y;
			typename QpSparseArray<QpFloatType>::Row const& row = m_M.row(m_classes * (yi * m_cardP + pi) + ya);
			QpFloatType def = row.defaultvalue;
			
			for (std::size_t pf=0, b=0; pf < m_cardP; pf++)
			{
				std::size_t f = exa.var[pf];
				double qif = def * k[a];
				//check whether we are at an existing element of the sparse row
				if( b != row.size && pf == row.entry[b].index){
					qif = row.entry[b].value * k[a];
					++b;//move to next element
				}
				if(f >= m_activeVar || f == i)
					continue;
				
				double af = m_alpha(f);
				double gf = m_gradient(f);
				double df = m_variables[f].diagonal;
				
				//check whether a step is possible at all.
				if (!(af > 0.0 && gf < 0.0) && !(af < m_C && gf > 0.0))
					continue;
				
				double gain = detail::maximumGainQuadratic2D(di,df,qif,di,gi,gf);
				if( gain > bestgain){
					j = f;
					bestgain = gain;
				}
			}
		}

		return maxViolation;
	}
	
protected:
	
	void gradientUpdate(std::size_t r, double mu, QpFloatType* q)
	{
		for ( std::size_t a= 0; a< m_activeEx; a++)
		{
			double k = q[a];
			Example& ex = m_examples[a];
			typename QpSparseArray<QpFloatType>::Row const& row = m_M.row(m_classes * r + ex.y);
			QpFloatType def = row.defaultvalue;
			for (std::size_t b=0; b<row.size; b++){
				std::size_t p = row.entry[b].index;
				m_gradient(ex.var[p]) -= mu * (row.entry[b].value - def) * k;
			}
			if (def != 0.0){
				double upd = mu* def * k;
				for (std::size_t b=0; b<ex.active; b++) 
					m_gradient(ex.avar[b]) -= upd;
			}
		}
	}

	///true if the problem has already been unshrinked
	bool bUnshrinked;
	
	///shrink a variable
	void deactivateVariable(std::size_t v)
	{
		std::size_t ev = m_variables[v].i;
		std::size_t iv = m_variables[v].index;
		unsigned int pv = m_variables[v].p;
		Example* exv = &m_examples[ev];

		std::size_t ih = exv->active - 1;
		std::size_t h = exv->avar[ih];
		m_variables[v].index = ih;
		m_variables[h].index = iv;
		std::swap(exv->avar[iv], exv->avar[ih]);
		iv = ih;
		exv->active--;

		std::size_t j = m_activeVar - 1;
		std::size_t ej = m_variables[j].i;
		std::size_t ij = m_variables[j].index;
		unsigned int pj = m_variables[j].p;
		Example* exj = &m_examples[ej];

		// exchange entries in the lists
		std::swap(m_alpha(v), m_alpha(j));
		std::swap(m_gradient(v), m_gradient(j));
		std::swap(m_linear(v), m_linear(j));
		std::swap(m_variables[v], m_variables[j]);

		m_variables[exv->avar[iv]].index = ij;
		m_variables[exj->avar[ij]].index = iv;
		exv->avar[iv] = j;
		exv->var[pv] = j;
		exj->avar[ij] = v;
		exj->var[pj] = v;

		m_activeVar--;
	}

	///shrink an m_examples
	void deactivateExample(std::size_t e)
	{
		SHARK_ASSERT(e < m_activeEx);
		std::size_t j = m_activeEx - 1;
		m_activeEx--;
		if(e == j) return;

		std::swap(m_examples[e], m_examples[j]);

		std::size_t* pe = m_examples[e].var;
		std::size_t* pj = m_examples[j].var;
		for (std::size_t v = 0; v < m_cardP; v++)
		{
			SHARK_ASSERT(pj[v] >= m_activeVar);
			m_variables[pe[v]].i = e;
			m_variables[pj[v]].i = j;
		}

		m_kernelMatrix.flipColumnsAndRows(e, j);
	}
	
	/// \brief Returns the original index of the example of a variable in the dataset before optimization.
	///
	/// Shrinking is an internal detail so the communication with the outside world uses the original indizes.
	std::size_t originalIndex(std::size_t v)const{
		std::size_t i = m_variables[v].i;
		return m_examples[i].index;//i before shrinking
	}

	/// data structure describing one m_variables of the problem
	struct Variable
	{
		///index into the example list
		std::size_t i;
		/// constraint corresponding to this m_variables
		unsigned int p;
		/// index into example->m_numVariables
		std::size_t index;
		/// diagonal entry of the big Q-matrix
		double diagonal;
	};

	/// data structure describing one training example
	struct Example
	{
		/// example index in the dataset, not the example vector!
		std::size_t index;
		/// label of this example
		unsigned int y;
		/// number of active m_numVariables
		std::size_t active;
		/// list of all m_cardP m_numVariables, in order of the p-index
		std::size_t* var;
		/// list of active m_numVariables
		std::size_t* avar;            
	};

	///kernel matrix (precomputed matrix or matrix cache)
	Matrix& m_kernelMatrix;

	///kernel modifiers
	QpSparseArray<QpFloatType> const& m_M;			// M(|P|*y_i+p, y_j, q)

	///complexity constant; upper bound on all variabless
	double m_C;
	
	///number of m_classes in the problem
	unsigned int m_classes;
	
	///number of dual m_numVariables per example
	std::size_t m_cardP;
	
	///number of m_examples in the problem (size of the kernel matrix)
	std::size_t m_numExamples;

	///number of m_numVariables in the problem = m_examples times m_cardP
	std::size_t m_numVariables;
	
	///m_linear part of the objective function
	RealVector m_linear;
	
	///solution candidate
	RealVector m_alpha;
	
	///m_gradient of the objective function
	///The m_gradient array is of fixed size and not subject to shrinking.
	RealVector m_gradient;

	///information about each training example
	std::vector<Example> m_examples;

	///information about each m_variables of the problem
	std::vector<Variable> m_variables;

	///space for the example[i].var pointers
	std::vector<std::size_t> m_storage1;

	///space for the example[i].avar pointers
	std::vector<std::size_t> m_storage2;

	///number of currently active m_examples
	std::size_t m_activeEx;

	///number of currently active variabless
	std::size_t m_activeVar;

	///should the m_problem use the shrinking heuristics?
	bool m_useShrinking;
};


template<class Matrix>
class BiasSolver{
public:
	typedef typename Matrix::QpFloatType QpFloatType;
	BiasSolver(QpMcBoxDecomp<Matrix>* problem) : m_problem(problem){}
		
	void solve(
		RealVector& bias,
		QpStoppingCondition& stop,
		QpSparseArray<QpFloatType> const& nu,
		bool sumToZero,
		QpSolutionProperties* prop = NULL
	){
		std::size_t classes = bias.size();
		std::size_t numExamples = m_problem->getNumExamples();
		std::size_t cardP = m_problem->cardP();
		RealVector stepsize(classes, 0.01);
		RealVector prev(classes,0);
		RealVector step(classes);
		
		double start_time = Timer::now();
		unsigned long long iterations = 0;
		
		do{
			QpSolutionProperties propInner;
			QpSolver<QpMcBoxDecomp<Matrix> > solver(*m_problem);
			solver.solve(stop, &propInner);
			iterations += propInner.iterations;

			// Rprop loop to update the bias
			while (true)
			{
				RealMatrix dualGradient = m_problem->solutionGradient();
				// compute the primal m_gradient w.r.t. bias
				RealVector grad(classes,0);

				for (std::size_t i=0; i<numExamples; i++){
					for (std::size_t p=0; p<cardP; p++){
						double g = dualGradient(i,p);
						if (g > 0.0)
						{
							unsigned int y = m_problem->label(i);
							typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP + p);
							for (std::size_t b=0; b<row.size; b++) 
								grad(row.entry[b].index) -= row.entry[b].value;
						}
					}
				}
				
				//~ for (std::size_t i=0; i<numExamples; i++){
					//~ unsigned int y = m_problem->label(i);
					//~ for (std::size_t p=0; p<cardP; p++){
						//~ double a = m_problem->alpha(i,p);
						//~ if(a == 0) continue;
						//~ typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP + p);
						//~ for (std::size_t b=0; b<row.size; b++) 
							//~ grad(row.entry[b].index) -= row.entry[b].value * a;
					//~ }
				//~ }

				if (sumToZero)
				{
					// project the gradient
					grad -= sum(grad) / classes;
				}

				// Rprop
				for (std::size_t c=0; c<classes; c++)
				{
					double g = grad(c);
					if (g > 0.0) 
						step(c) = -stepsize(c);
					else if (g < 0.0) 
						step(c) = stepsize(c);

					double gg = prev(c) * grad(c);
					if (gg > 0.0) 
						stepsize(c) *= 1.2;
					else 
						stepsize(c) *= 0.5;
				}
				prev = grad;

				if (sumToZero)
				{
					// project the step
					step -= sum(step) / classes;
				}

				// update the solution and the dual m_gradient
				bias += step;
				performBiasUpdate(step,nu);
				
				if (max(stepsize) < 0.01 * stop.minAccuracy) break;
			}
		}while(m_problem->checkKKT()> stop.minAccuracy);
		
		if (prop != NULL)
		{
			double finish_time = Timer::now();

			prop->accuracy = m_problem->checkKKT();
			prop->value = m_problem->functionValue();
			prop->iterations = iterations;
			prop->seconds = finish_time - start_time;
		}
	}
private:
	void performBiasUpdate(
		RealVector const& step, QpSparseArray<QpFloatType> const& nu
	){
		std::size_t numExamples = m_problem->getNumExamples();
		std::size_t cardP = m_problem->cardP();
		RealMatrix deltaLinear(numExamples,cardP,0.0);
		for (std::size_t i=0; i<numExamples; i++){
			for (std::size_t p=0; p<cardP; p++){
				unsigned int y = m_problem->label(i);
				typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP +p);
				for (std::size_t b=0; b<row.size; b++)
				{
					deltaLinear(i,p) -= row.entry[b].value * step(row.entry[b].index);
				}
			}
		}
		m_problem->addDeltaLinear(deltaLinear);
		
	}
	QpMcBoxDecomp<Matrix>* m_problem;
};


}
#endif
