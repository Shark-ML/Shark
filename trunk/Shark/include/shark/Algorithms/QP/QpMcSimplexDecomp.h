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


#ifndef SHARK_ALGORITHMS_QP_QPMCSIMPLEXDECOMP_H
#define SHARK_ALGORITHMS_QP_QPMCSIMPLEXDECOMP_H

#include <shark/Algorithms/QP/QpSolver.h>
#include <shark/Algorithms/QP/QpSparseArray.h>
#include <shark/Algorithms/QP/Impl/AnalyticProblems.h>
#include <shark/Core/Timer.h>
#include <shark/Data/Dataset.h>


namespace shark {

template <class Matrix>
class QpMcSimplexDecomp
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

	/// Constructor
	/// \param  kernel               kernel matrix - cache or pre-computed matrix
	/// \param  M                   kernel modifiers in the format \f$ M_(y_i, p, y_j, q) = _M(classes*(y_i*|P|+p_i)+y_j, q) \f$
	/// \param  target the target labels for the variables
	/// \param linearMat the linear part of the problem
	/// \param C upper bound for all box variables, lower bound is 0.
	QpMcSimplexDecomp(
		Matrix& kernel,
		QpSparseArray<QpFloatType> const& M,
		Data<unsigned int> const& target,
		RealMatrix const& linearMat,
		double C
	)
	: m_kernelMatrix(kernel)
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

		SHARK_CHECK(
			target.numberOfElements() == m_numExamples 
			&& linearMat.size1() == kernel.size(),
			"[QpMcDecomp::QpMcDecomp] dimension conflict"
		);
		
		// prepare problem internal variables
		m_activeEx = m_numExamples;
		m_activeVar = m_numVariables;
		for (std::size_t v=0, i=0; i<m_numExamples; i++)
		{
			unsigned int y = target.element(i);
			m_examples[i].index = i;
			m_examples[i].y = y;
			m_examples[i].active = m_cardP;
			m_examples[i].var = &m_storage1[m_cardP * i];
			m_examples[i].avar = &m_storage2[m_cardP * i];
			m_examples[i].varsum = 0;
			double k = m_kernelMatrix.entry(i, i);
			m_examples[i].diagonal = k;
			for (std::size_t p=0; p<m_cardP; p++, v++)
			{
				m_variables[v].example = i;
				m_variables[v].p = p;
				m_variables[v].index = p;
				double Q = m_M(m_classes * (y * m_cardP + p) + y, p) * k;
				m_variables[v].diagonal = Q;
				m_storage1[v] = v;
				m_storage2[v] = v;
				
				m_linear(v) = m_gradient(v) = linearMat(i,p);
			}
		}
		// initialize unshrinking to make valgrind happy.
		bUnshrinked = false;
	}
	
	/// enable/disable shrinking
	void setShrinking(bool shrinking = true)
	{
		m_useShrinking = shrinking; 
	}
	
	/// \brief Returns the solution found.
	RealMatrix solution() const{
		RealMatrix solutionMatrix(m_numVariables,m_cardP,0);
		for (std::size_t v=0; v<m_numVariables; v++)
		{
			solutionMatrix(originalIndex(v),m_variables[v].p) = m_alpha(v);
		}
		return solutionMatrix;
	}
	/// \brief Returns the gradient of the solution.
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
			std::size_t i = m_variables[v].example;
			SHARK_ASSERT(i < m_activeEx);
			unsigned int p = m_variables[v].p;
			unsigned int y = m_examples[i].y;
			unsigned int r = m_cardP * y + p;
			double& varsum = m_examples[i].varsum;
			//the upper bound depends on the values of the variables of the other classes.
			double upperBound = m_C-varsum+m_alpha(v);
			
			QpFloatType* q = m_kernelMatrix.row(i, 0, m_activeEx);
			double Qvv = m_variables[v].diagonal;
			double mu = -m_alpha(v);
			detail::solveQuadraticEdge(m_alpha(v),m_gradient(v),Qvv,0,upperBound);
			mu += m_alpha(v);
			updateVarsum(i,mu);
			gradientUpdate(r, mu, q);
		}
		else
		{
			// S2DO
			std::size_t iv = m_variables[v].example;
			SHARK_ASSERT(iv < m_activeEx);
			unsigned int pv = m_variables[v].p;
			unsigned int yv = m_examples[iv].y;
			double& varsumv = m_examples[iv].varsum;

			std::size_t iw = m_variables[w].example;
			SHARK_ASSERT(iw < m_activeEx);
			unsigned int pw = m_variables[w].p;
			unsigned int yw = m_examples[iw].y;
			double& varsumw = m_examples[iw].varsum;

			// get the matrix rows corresponding to the working set
			QpFloatType* qv = m_kernelMatrix.row(iv, 0, m_activeEx);
			QpFloatType* qw = m_kernelMatrix.row(iw, 0, m_activeEx);
			unsigned int rv = m_cardP*yv+pv;
			unsigned int rw = m_cardP*yw+pw;

			// get the Q-matrix restricted to the working set
			double Qvv = m_variables[v].diagonal;
			double Qww = m_variables[w].diagonal;
			double Qvw = m_M(m_classes * rv + yw, pw) * qv[iw];
			
			//same sample - simplex case
			double mu_v = -m_alpha(v);
			double mu_w = -m_alpha(w);
			if(iv == iw){
				
				double upperBound = m_C-varsumv+m_alpha(v)+m_alpha(w);
				// solve the sub-problem and update the gradient using the step sizes mu
				detail::solveQuadratic2DTriangle(m_alpha(v), m_alpha(w),
					m_gradient(v), m_gradient(w),
					Qvv, Qvw, Qww,
					upperBound
				);
				mu_v += m_alpha(v);
				mu_w += m_alpha(w);
				updateVarsum(iv,mu_v+mu_w);
			}
			else{
				double Uv = m_C-varsumv+m_alpha(v);
				double Uw = m_C-varsumw+m_alpha(w);
				// solve the sub-problem and update the gradient using the step sizes mu
				detail::solveQuadratic2DBox(m_alpha(v), m_alpha(w),
					m_gradient(v), m_gradient(w),
					Qvv, Qvw, Qww,
					0, Uv, 0, Uw
				);
				mu_v += m_alpha(v);
				mu_w += m_alpha(w);
				updateVarsum(iv,mu_v);
				updateVarsum(iw,mu_w);
			}
			
			double varsumvo = 0;
			for(std::size_t p = 0; p != m_cardP; ++p){
				std::size_t varIndex = m_examples[iv].var[p];
				varsumvo += m_alpha[varIndex];
			}
			double varsumwo = 0;
			for(std::size_t p = 0; p != m_cardP; ++p){
				std::size_t varIndex = m_examples[iw].var[p];
				varsumwo += m_alpha[varIndex];
			}
			gradientUpdate(rv, mu_v, qv);
			gradientUpdate(rw, mu_w, qw);
		}
	}
	
	/// Shrink the problem
	bool shrink(double epsilon)
	{
		if(! m_useShrinking)
			return false;
		if (! bUnshrinked)
		{
			if (checkKKT() < 10.0 * epsilon)
			{
				// unshrink the problem at this accuracy level
				unshrink();
				bUnshrinked = true;
			}
		}
		
		//iterate through all simplices.
		for (int i= m_activeEx-1; i >= 0; i--){
			Example const& ex = m_examples[i];
			std::pair<std::pair<double,std::size_t>,std::pair<double,std::size_t> > pair = getSimplexMVP(ex);
			double up = pair.first.first;
			double down = pair.second.first;
			
			//check the simplex for possible search directions
			//case 1:  simplex is bounded and stays at the bound, in this case proceed as in MVP
			if(down > 0 && ex.varsum == m_C && up-down > 0){
				int pc = ex.active; 
				for(int p = pc-1; p >= 0; --p){
					double a = m_alpha(ex.avar[p]);
					double g = m_gradient(ex.avar[p]);
					//if we can't do a step along the simplex, we can shrink the variable.
					if(a == 0 && g-down < 0){
						deactivateVariable(ex.avar[p]);
					}
					else if (a == m_C && up-g < 0){
						//shrinking this variable means, that the whole simplex can't move,
						//so shrink every variable, even the ones that previously couldn't
						//be shrinked
						for(int q = ex.active; q >= 0; --q){
							deactivateVariable(ex.avar[q]);
						}
						p = 0;
					}
				}
			}
			//case 2: all variables are zero and pushed against the boundary
			// -> shrink the example
			else if(ex.varsum == 0 && up < 0){
				int pc = ex.active; 
				for(int p = pc-1; p >= 0; --p){
					deactivateVariable(ex.avar[p]);
				}
			}
			//case 3: the simplex is not bounded and there are free variables. 
			//in this case we currently do not shrink
			//as a variable might get bounded at some point which means that all variables
			//can be important again.
			//else{
			//}
			
		}
//		std::cout<<"shrunk. remaining: "<<m_activeEx<<","<<m_activeVar<<std::endl;
		return true;
	}

	/// Activate all m_numVariables
	void unshrink()
	{
		if (m_activeVar == m_numVariables) return;

		// compute the inactive m_gradient components (quadratic time complexity)
		subrange(m_gradient, m_activeVar, m_numVariables) = subrange(m_linear, m_activeVar, m_numVariables);
		for (std::size_t v = 0; v != m_numVariables; v++)
		{
			double mu = m_alpha(v);
			if (mu == 0.0) continue;

			std::size_t iv = m_variables[v].example;
			unsigned int pv = m_variables[v].p;
			unsigned int yv = m_examples[iv].y;
			unsigned int r = m_cardP * yv + pv;
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
	
	///
	/// \brief select the working set
	///
	/// Select one or two numVariables for the sub-problem
	/// and return the maximal KKT violation. The method
	/// MAY select the same index for i and j. In that
	/// case the working set consists of a single variables.
	/// The working set may be invalid if the method reports
	/// a KKT violation of zero, indicating optimality.	
	double selectWorkingSet(std::size_t& i, std::size_t& j)
	{
		
		//first order selection
		//we select the best variable along the box constraint (for a step between samples)
		//and the best gradient and example index for a step along the simplex (step inside a sample)
		double maxGradient = 0;//max gradient for variables between samples (box constraints)
		double maxSimplexGradient = 0;//max gradient along the simplex constraints
		std::size_t maxSimplexExample = 0;//example with the maximum simplex constraint
		i = j = 0;
		// first order selection
		for (std::size_t e=0; e< m_activeEx; e++)
		{
			Example& ex = m_examples[e];
			bool canGrow = ex.varsum < m_C;
			
			//calculate the maximum violationg pair for the example.
			std::pair<std::pair<double,std::size_t>,std::pair<double,std::size_t> > pair = getSimplexMVP(ex);
			double up = pair.first.first;
			double down = pair.second.first;
			
			if(!canGrow && up - down > maxSimplexGradient){
				maxSimplexGradient = up-down;
				maxSimplexExample = e;
			}
			if (canGrow && up > maxGradient) {
				maxGradient = up;
				i = pair.first.second;
			}
			if (-down > maxGradient) {
				maxGradient = -down;
				i = pair.second.second;
			}
		}
		
		//find the best possible partner of the variable
		//by searching every other sample
		std::pair<std::pair<std::size_t,std::size_t> ,double > boxPair = maxGainBox(i);
		double bestGain = boxPair.second;
		std::pair<std::size_t, std::size_t> best = boxPair.first;
		
		//always search the simplex of the variable
		std::pair<std::pair<std::size_t,std::size_t> ,double > simplexPairi = maxGainSimplex(m_variables[i].example);
		if(simplexPairi.second > bestGain){
			best = simplexPairi.first;
			bestGain = simplexPairi.second;
		}
		//finally search also in the best simplex
		if(maxSimplexGradient > 0){
			std::pair<std::pair<std::size_t,std::size_t> ,double > simplexPairBest = maxGainSimplex(maxSimplexExample);
			if(simplexPairBest.second > bestGain){
				best = simplexPairBest.first;
				bestGain = simplexPairBest.second;
			}
		}
		i = best.first;
		j = best.second;
		//return the mvp gradient
		return std::max(maxGradient,maxSimplexGradient);
	}
	
	/// return the largest KKT violation
	double checkKKT()const
	{
		double ret = 0.0;
		for (std::size_t i=0; i<m_activeEx; i++){
			Example const& ex = m_examples[i];
			std::pair<std::pair<double,std::size_t>,std::pair<double,std::size_t> > pair = getSimplexMVP(ex);
			double up = pair.first.first;
			double down = pair.second.first;
			
			//check all search directions
			//case 1:  decreasing the value of a variable
			ret = std::max(-down, ret);
			//case 2: if we are not at the boundary increasing the variable
			if(ex.varsum < m_C)
				ret = std::max(up, ret);
			//case 3: along the plane \sum_i alpha_i = m_C
			if(ex.varsum == m_C)
				ret = std::max(up-down, ret);
		}
		return ret;
	}
	
protected:
	
	/// data structure describing one variable of the problem
	struct Variable
	{
		///index into the example list
		std::size_t example;
		/// constraint corresponding to this variable
		unsigned int p;
		/// index into example->m_numVariables
		unsigned int index;
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
		/// number of active variables
		unsigned int active;
		/// list of all m_cardP variables, in order of the p-index
		std::size_t* var;
		/// list of active variables
		std::size_t* avar;
		/// total sum of all variables of this example
		double varsum;
		/// diagonal entry of the kernel matrix k(x,x);
		double diagonal;
	};
	
	/// \brief Finds the second variable of a working set using maximum gain and returns the pair and gain
	///
	/// The variable is searched in-between samples. And not inside the simplex of i.
	/// It returns the best pair (i,j) as well as the gain. If the first variable
	/// can't make a step, gain 0 is returned with pair(i,i).
	std::pair<std::pair<std::size_t,std::size_t>,double> maxGainBox(std::size_t i)const{
		std::size_t e = m_variables[i].example;
		SHARK_ASSERT(e < m_activeEx);
		unsigned int pi = m_variables[i].p;
		unsigned int yi = m_examples[e].y;
		double Qii = m_variables[i].diagonal;
		double gi = m_gradient(i);
		if(m_examples[e].varsum == m_C && gi > 0)
			return std::make_pair(std::make_pair(i,i),0.0);
		
		
		QpFloatType* k = m_kernelMatrix.row(e, 0, m_activeEx);
		
		std::size_t bestj = i;
		double bestGain = gi * gi / Qii;
		
		for (std::size_t a=0; a<m_activeEx; a++)
		{
			//don't search the simplex of the first variable
			if(a == e) continue;
			
			Example const& exa = m_examples[a];
			unsigned int ya = exa.y;
			bool canGrow = exa.varsum != m_C;
			
			typename QpSparseArray<QpFloatType>::Row const& row = m_M.row(m_classes * (yi * m_cardP + pi) + ya);
			QpFloatType def = row.defaultvalue;
			
			for (std::size_t p=0, b=0; p < m_cardP; p++)
			{
				std::size_t j = exa.var[p];
				
				std::size_t Qjj = m_variables[j].diagonal;
				double gj = m_gradient(j);
				double Qij = def * k[a];
				//check whether we are at an existing element of the sparse row
				if( b != row.size && p == row.entry[b].index){
					Qij = row.entry[b].value * k[a];
					++b;//move to next element
				}
				
				//don't check variables which are shrinked or bounded
				if(j >= m_activeVar || (m_alpha(j) == 0 && gj <= 0)|| (!canGrow && gj >= 0))
					continue;
				
				double gain = detail::maximumGainQuadratic2D(Qii, Qjj, Qij, gi,gj);
				if( bestGain < gain){
					bestj = j;
					bestGain = gain;
				}
			}
		}
		return std::make_pair(std::make_pair(i,bestj),bestGain);
	}
	
	///\brief Returns the best variable pair (i,j) and gain for a given example.
	///
	/// For a given example all possible pairs of variables are checkd and the one giving
	/// the maximum gain is returned. This method has a special handling for the
	/// simplex case.
	std::pair<std::pair<std::size_t,std::size_t>,double> maxGainSimplex(std::size_t e)const{
		Example const& ex = m_examples[e];
		std::size_t pc = ex.active;//number of active variables for this example
		unsigned int y = ex.y;//label of the example
		bool canGrow = ex.varsum < m_C; //are we inside the simplex?
		double Qee = m_examples[e].diagonal; //kernel entry of the example
		
		double bestGain = -1e100;
		std::size_t besti = 0;
		std::size_t bestj = 0;
		
		//search through all possible variable pairs
		//for every pair we will build the quadratic subproblem 
		//and than decide whether we can do 
		// 1.a valid step in the inside of the simplex
		// that is canGrow==true or the gradients of both variables point inside
		// 2. a valid step along the simplex constraint, 
		// that is cangrow == true and both variables point outside)
		// 3. a 1-D step
		// that is canGrow == true or alpha(i) > 0 & gradient(i) < 0
		
		//iterate over the active ones as the first variable
		for(std::size_t p1 = 0; p1 != pc; ++p1){
			std::size_t i = ex.avar[p1];
			double gi = m_gradient(i);
			double ai = m_alpha(i);
			double Qii = m_variables[i].diagonal;
			
			//check whether a 1-D gain is possible
			if((gi < 0 && m_alpha(i) > 0.0) || (gi > 0 && canGrow)){
				double gain = gi * gi / Qii;
				if(gain > bestGain){
					bestGain= gain;
					besti = i;
					bestj = i;
				}
			}
			
			//now create the 2D problem for all possible variable pairs
			//and than check for possible gain steps
			//find first the row of coefficients for M(y,y,i,j) for all j
			//question: is p1 == m_variables[ex.avar[p1]].p?
			//otherwise: is p1 == m_variables[ex.var[p1]].p for *all* variables?
			typename QpSparseArray<QpFloatType>::Row const& row = m_M.row(m_classes * (y * m_cardP + m_variables[i].p) + y);
			QpFloatType def = row.defaultvalue;
			
			//we need to iterate over all vars instead of only the active variables to be in sync with the matrix row
			//we will still overstep all inactive variables
			for(std::size_t p2 = 0, b=0; p2 != m_cardP; ++p2){
				std::size_t j = ex.var[p2];
				double gj = m_gradient(j);
				double aj = m_alpha(j);
				double Qjj = m_variables[j].diagonal;
				
				//create the offdiagonal element of the 2D problem
				double Qij = def * Qee;
				//check whether we are at an existing element of the sparse row
				if( b != row.size && p2 == row.entry[b].index){
					Qij = row.entry[b].value * Qee;
					++b;//move to next element
				}
				
				//ignore inactive variables or variables we already checked
				if(j >= m_activeVar || j <= i ){
					continue;
				}
				
				double gain = 0;
				//check whether we can move along the simplex constraint
				if(!canGrow && gi > 0 && gj > 0){
					double gainUp = 0;
					double gainDown = 0;
					//we need to check both search directions for ai
					if(aj > 0 && gi-gj > 0){
						gainUp = detail::maximumGainQuadratic2DOnLine(Qii, Qjj, Qij, gi,gj);
					}
					//also check whether a line search in the other direction works
					if (ai > 0 &&gj-gi > 0){
						gainDown = detail::maximumGainQuadratic2DOnLine(Qjj, Qii, Qij, gj,gi);
					}
					gain = std::max(gainUp,gainDown);
				}
				//else we are inside the simplex
				//in this case only check that both variables can shrink if needed
				else if(!(gi <= 0 && ai == 0) && !(gj<= 0  && aj == 0)){
					gain = detail::maximumGainQuadratic2D(Qii, Qjj, Qij, gi,gj);
				}
				
				//accept only maximum gain
				if(gain > bestGain){
					bestGain= gain;
					besti = i;
					bestj = j;
				}
				
			}
		}
		//return best pair and possible gain
		return std::make_pair(std::make_pair(besti,bestj),bestGain);
	}
	
	/// \brief For a given simplex returns the MVP indicies (max_up,min_down)
	std::pair<
		std::pair<double,std::size_t>,
		std::pair<double,std::size_t> 
	> getSimplexMVP(Example const& ex)const{
		unsigned int pc = ex.active;
		double up = -1e100;
		double down = 1e100;
		std::size_t maxUp = ex.avar[0];
		std::size_t minDown = ex.avar[0];
		for (std::size_t p = 0; p != pc; p++)
		{
			std::size_t v = ex.avar[p];
			SHARK_ASSERT(v < m_activeVar);
			double a = m_alpha(v);
			double g = m_gradient(v);
			if (g > up) { 
				maxUp = v;
				up = g;
			}
			if (a > 0.0 && g < down){
				minDown = v;
				down = g;
			}
		}
		return std::make_pair(std::make_pair(up,maxUp),std::make_pair(down,minDown));
	}
	
	void updateVarsum(std::size_t exampleId, double mu){
		double& varsum = m_examples[exampleId].varsum;
		varsum += mu;
		if(varsum > 1.e-12 && m_C-varsum > 1.e-12*m_C)
			return;
		//recompute for numerical accuracy
		
		varsum = 0;
		for(std::size_t p = 0; p != m_cardP; ++p){
			std::size_t varIndex = m_examples[exampleId].var[p];
			varsum += m_alpha[varIndex];
		}
		
		if(varsum < 1.e-14)
			varsum = 0;
		if(m_C-varsum < 1.e-14*m_C)
			varsum = m_C;
	}
	
	void gradientUpdate(std::size_t r, double mu, float* q)
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
	
	/// shrink a variable
	void deactivateVariable(std::size_t v)
	{
		std::size_t ev = m_variables[v].example;
		unsigned int iv = m_variables[v].index;
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
		std::size_t ej = m_variables[j].example;
		unsigned int ij = m_variables[j].index;
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
		
		//finally check if the example is needed anymore
		if(exv->active == 0)
			deactivateExample(ev);
	}

	/// shrink an example
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
			m_variables[pe[v]].example = e;
			m_variables[pj[v]].example = j;
		}
		m_kernelMatrix.flipColumnsAndRows(e, j);
	}
	
	/// \brief Returns the original index of the example of a variable in the dataset before optimization.
	///
	/// Shrinking is an internal detail so the communication with the outside world uses the original indizes.
	std::size_t originalIndex(std::size_t v)const{
		std::size_t i = m_variables[v].example;
		return m_examples[i].index;//i before shrinking
	}

	/// kernel matrix (precomputed matrix or matrix cache)
	Matrix& m_kernelMatrix;

	/// kernel modifiers
	QpSparseArray<QpFloatType> const& m_M;			// M(|P|*y_i+p, y_j, q)

	/// complexity constant; upper bound on all variabless
	double m_C;
	
	/// number of classes in the problem
	unsigned int m_classes;
	
	/// number of dual variables per example
	unsigned int m_cardP;
	
	/// number of examples in the problem (size of the kernel matrix)
	std::size_t m_numExamples;

	/// number of variables in the problem = m_numExamples * m_cardP
	std::size_t m_numVariables;
	
	/// linear part of the objective function
	RealVector m_linear;
	
	/// solution candidate
	RealVector m_alpha;
	
	/// gradient of the objective function
	/// The m_gradient array is of fixed size and not subject to shrinking.
	RealVector m_gradient;

	/// information about each training example
	std::vector<Example> m_examples;

	/// information about each variable of the problem
	std::vector<Variable> m_variables;

	/// space for the example[i].var pointers
	std::vector<std::size_t> m_storage1;

	/// space for the example[i].avar pointers
	std::vector<std::size_t> m_storage2;

	/// number of currently active examples
	std::size_t m_activeEx;

	/// number of currently active variables
	std::size_t m_activeVar;

	/// should the m_problem use the shrinking heuristics?
	bool m_useShrinking;
	
	/// true if the problem has already been unshrinked
	bool bUnshrinked;
};


template<class Matrix>
class BiasSolverSimplex{
public:
	typedef typename Matrix::QpFloatType QpFloatType;
	BiasSolverSimplex(QpMcSimplexDecomp<Matrix>* problem) : m_problem(problem){}
		
	void solve(
		RealVector& bias,
		QpStoppingCondition& stop,
		QpSparseArray<QpFloatType> const& nu,
		bool sumToZero = false
	){
		std::size_t classes = bias.size();
		std::size_t numExamples = m_problem->getNumExamples();
		std::size_t cardP = m_problem->cardP();
		RealVector stepsize(classes, 0.01);
		RealVector prev(classes,0);
		RealVector step(classes);
		
		do{
			QpSolver<QpMcSimplexDecomp<Matrix> > solver(*m_problem);
			solver.solve(stop);

			// Rprop loop to update the bias
			while (true)
			{
				RealMatrix dualGradient = m_problem->solutionGradient();
				// compute the primal m_gradient w.r.t. bias
				RealVector grad(classes,0);
				for (std::size_t i=0; i<numExamples; i++)
				{
					unsigned int largestP = cardP;
					double largest_value = 0.0;
					for (std::size_t p=0; p<cardP; p++)
					{
						if (dualGradient(i,p) > largest_value)
						{
							largest_value = dualGradient(i,p);
							largestP = p;
						}
					}
					if (largestP < cardP)
					{
						unsigned int y = m_problem->label(i);
						typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP + largestP);
						for (std::size_t b=0; b != row.size; b++) 
							grad(row.entry[b].index) -= row.entry[b].value;
					}
				}

				if (sumToZero)
				{
					// project the m_gradient
					double mean = sum(grad) / (double)classes;
					grad -= blas::repeat(mean,classes);
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
					double mean = sum(step) / (double)classes;
					step -= blas::repeat(mean,classes);
				}

				// update the solution and the dual m_gradient
				bias += step;
				performBiasUpdate(step,nu);
				
				if (max(stepsize) < 0.01 * stop.minAccuracy) break;
			}
		}while(m_problem->checkKKT()> stop.minAccuracy);
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
				// delta = \sum_m \nu_{m,p,y_i} \Delta b(m)
				typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP +p);
				for (std::size_t b=0; b<row.size; b++)
				{
					deltaLinear(i,p) -= row.entry[b].value * step(row.entry[b].index);
				}
			}
		}
		m_problem->addDeltaLinear(deltaLinear);
		
	}
	QpMcSimplexDecomp<Matrix>* m_problem;
};


}
#endif
