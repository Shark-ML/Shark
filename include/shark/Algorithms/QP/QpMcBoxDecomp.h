//===========================================================================
/*!
 *  \brief Quadratic programming solver for multi-class SVMs
 *
 *
 *  \author  T. Glasmachers
 *  \date    2007-2012
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_QP_QPMCBOXDECOMP_H
#define SHARK_ALGORITHMS_QP_QPMCBOXDECOMP_H

#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Algorithms/QP/QpSparseArray.h>
#include <shark/Algorithms/QP/Impl/AnalyticProblems.h>
#include <shark/Core/Timer.h>
#include <shark/Data/Dataset.h>


namespace shark {


#define ITERATIONS_BETWEEN_SHRINKING 1000

template <class Matrix>
class QpMcBoxDecomp
{
public:
	typedef typename Matrix::QpFloatType QpFloatType;

	//! Constructor
	//! \param  kernel               kernel matrix - cache or pre-computed matrix
	//! \param  _M                   kernel modifiers in the format \f$ M_(y_i, p, y_j, q) = _M(classes*(y_i*|P|+p_i)+y_j, q) \f$
	QpMcBoxDecomp(
		Matrix& kernel,
		QpSparseArray<QpFloatType> const& M,
		Data<unsigned int> const& target,
		RealMatrix const& linearMat,
		double C
	)
	: kernelMatrix(kernel)
	, M(M)
	, C(C)
	, classes(numberOfClasses(target))
	, cardP(linearMat.size2())
	, numExamples(kernel.size())
	, numVariables(cardP * numExamples)
	, linear(numVariables)
	, alpha(numVariables,0.0)
	, gradient(numVariables)
	, examples(numExamples)
	, variables(numVariables)
	, storage1(numVariables)
	, storage2(numVariables)
	, useShrinking(true)
	{

		SHARK_CHECK(
			target.numberOfElements() == numExamples 
			&& linearMat.size1() == kernel.size(),
			"[QpMcDecomp::QpMcDecomp] dimension conflict"
		);
		
		// prepare solver internal variables
		activeEx = numExamples;
		activeVar = numVariables;
		for (std::size_t v=0, i=0; i<numExamples; i++)
		{
			unsigned int y = target.element(i);
			examples[i].index = i;
			examples[i].y = y;
			examples[i].active = cardP;
			examples[i].var = &storage1[cardP * i];
			examples[i].avar = &storage2[cardP * i];
			double k = kernelMatrix.entry(i, i);
			for (std::size_t p=0; p<cardP; p++, v++)
			{
				variables[v].i = i;
				variables[v].p = p;
				variables[v].index = p;
				double Q = M(classes * (y * cardP + p) + y, p) * k;
				variables[v].diagonal = Q;
				storage1[v] = v;
				storage2[v] = v;
				
				linear(v) = gradient(v) = linearMat(i,p);
			}
		}
	}
	
	//! enable/disable shrinking
	void setShrinking(bool shrinking = true)
	{
		useShrinking = shrinking; 
	}

	//!
	//! \brief solve the quadratic program
	//! \param  stop           stopping condition(s)
	//! \param  prop           solution properties (may be NULL)
	void solve(
		QpStoppingCondition& stop,
		QpSolutionProperties* prop = NULL
	){
		double start_time = Timer::now();

		bUnshrinked = false;
		std::size_t checkCounter = 0;//force initial shrinking in the first iteration

		// decomposition loop
		unsigned long long iter = 0;
		double acc = 0;
		while (iter != stop.maxIterations)
		{
			if (checkCounter == 0)
			{
				// shrink the problem
				if (useShrinking) 
					shrink(stop.minAccuracy);

				checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;
				
				//check time as stopping criterion
				if (stop.maxSeconds < 1e100)
				{
					double current_time = Timer::now();
					if (current_time - start_time > stop.maxSeconds)
					{
						if (prop != NULL) prop->type = QpTimeout;
						break;
					}
				}
			}
			// select a working set and check for optimality
			std::size_t v=0, w=0;
			acc = selectWorkingSet(v, w);
			//unshrink the problem if optimal
			//and select a new working set
			//this is costly
			if(activeVar != numVariables && acc < stop.minAccuracy){
				unshrink(stop.minAccuracy);
				acc = selectWorkingSet(v, w);
				if(acc >= stop.minAccuracy){//shrink again if not optimal
					shrink(stop.minAccuracy);
					checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;
				}
			}
			//if shrinking did not fail, this is the actual stopping condition
			if(acc < stop.minAccuracy){
				if (prop != NULL) prop->type = QpAccuracyReached;
				break;
			}
			
			//now do a simple SMO loop
			updateSMO(v,w);

			checkCounter--;
			iter++;
		}

		if (iter == stop.maxIterations)
		{
			if (prop != NULL) prop->type = QpMaxIterationsReached;
		}

		//compute the objective value
		double objective = 0.5*inner_prod(gradient+linear,alpha);
		
		double finish_time = Timer::now();

		if (prop != NULL)
		{
			prop->accuracy = acc;
			prop->value = objective;
			prop->iterations = iter;
			prop->seconds = finish_time - start_time;
		}
	}
	
	/// \brief Return the solution found.
	RealMatrix solution() const{
		RealMatrix solutionMatrix(numVariables,cardP,0);
		for (std::size_t v=0; v<numVariables; v++)
		{
			solutionMatrix(variables[v].i,variables[v].p) = alpha(v);
		}
		return solutionMatrix;
	}
	/// \brief Return the gradient of the solution.
	RealMatrix solutionGradient() const{
		RealMatrix solutionGradientMatrix(numVariables,cardP,0);
		for (std::size_t v=0; v<numVariables; v++)
		{
			solutionGradientMatrix(variables[v].i,variables[v].p) = gradient(v);
		}
		return solutionGradientMatrix;
	}
	
	unsigned int label(std::size_t i){
		return examples[i].y;
	}
	
	std::size_t getCardP()const{
		return cardP;
	}
	
	std::size_t getNumExamples()const{
		return numExamples;
	}
	
	//! return the largest KKT violation
	double checkKKT()const
	{
		double maxViolation = 0.0;
		for (std::size_t v=0; v<activeVar; v++)
		{
			double a = alpha(v);
			double g = gradient(v);
			if (a < C)
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
		SIZE_CHECK(deltaLinear.size1() == numExamples);
		SIZE_CHECK(deltaLinear.size2() == cardP);
		for (std::size_t v=0; v<numVariables; v++)
		{
			std::size_t i = variables[v].i;
			std::size_t p = variables[v].p;
			gradient(v) += deltaLinear(i,p); 
			linear(v) += deltaLinear(i,p);
		}
	}
	
protected:
	void updateSMO(std::size_t v, std::size_t w){
		// update
		if (v == w)
		{
			// Limit case of a single variables;
			// this means that there is only one
			// non-optimal variables left.
			std::size_t i = variables[v].i;
			unsigned int p = variables[v].p;
			unsigned int y = examples[i].y;
			unsigned int r = cardP * y + p;
			QpFloatType* q = kernelMatrix.row(i, 0, activeEx);
			double Qvv = variables[v].diagonal;
			double mu = gradient(v) / Qvv;
			if (mu < 0.0)
			{
				if (mu <= -alpha(v))
				{
					mu = -alpha(v);
					alpha(v) = 0.0;
				}
				else alpha(v) += mu;
			}
			else
			{
				if (mu >= C - alpha(v))
				{
					mu = C - alpha(v);
					alpha(v) = C;
				}
				else alpha(v) += mu;
			}
			gradientUpdate(r, mu, q);
		}
		else
		{
			// S2DO
			std::size_t iv = variables[v].i;
			unsigned int pv = variables[v].p;
			unsigned int yv = examples[iv].y;

			std::size_t iw = variables[w].i;
			unsigned int pw = variables[w].p;
			unsigned int yw = examples[iw].y;

			// get the matrix rows corresponding to the working set
			QpFloatType* qv = kernelMatrix.row(iv, 0, activeEx);
			QpFloatType* qw = kernelMatrix.row(iw, 0, activeEx);
			unsigned int rv = cardP*yv+pv;
			unsigned int rw = cardP*yw+pw;

			// get the Q-matrix restricted to the working set
			double Qvv = variables[v].diagonal;
			double Qww = variables[w].diagonal;
			double Qvw = M(classes * rv + yw, pw) * qv[iw];

			// solve the sub-problem and update the gradient using the step sizes mu
			double mu_v = 0.0;
			double mu_w = 0.0;
			
			solve2D_box(alpha(v), alpha(w),
				gradient(v), gradient(w),
				Qvv, Qvw, Qww,
				C, C,
				mu_v, mu_w
			);
			gradientUpdate(rv, mu_v, qv);
			gradientUpdate(rw, mu_w, qw);
		}
	}
	
	
	void gradientUpdate(std::size_t r, double mu, float* q)
	{
		for ( std::size_t a= 0; a< activeEx; a++)
		{
			double k = q[a];
			tExample& ex = examples[a];
			typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * r + ex.y);
			QpFloatType def = row.defaultvalue;
			if (def == 0.0)
			{
				for (std::size_t b=0; b<row.size; b++)
				{
					std::size_t p = row.entry[b].index;
					gradient(ex.var[p]) -= mu * row.entry[b].value * k;
				}
			}
			else
			{
				for (std::size_t b=0; b<row.size; b++)
				{
					std::size_t p = row.entry[b].index;
					gradient(ex.var[p]) -= mu * (row.entry[b].value - def) * k;
				}
				double upd = mu* def * k;
				for (std::size_t b=0; b<ex.active; b++) 
					gradient(ex.avar[b]) -= upd;
			}
		}
	}

	//!
	//! \brief select the working set
	//!
	//! Select one or two numVariables for the sub-problem
	//! and return the maximal KKT violation. The method
	//! MAY select the same index for i and j. In that
	//! case the working set consists of a single variables.
	//! The working set may be invalid if the method reports
	//! a KKT violation of zero, indicating optimality.
	double selectWorkingSet(std::size_t& i, std::size_t& j)
	{
		// box case
		double maxViolation = 0.0;

		// first order selection
		for (std::size_t a=0; a<activeVar; a++)
		{
			double aa = alpha(a);
			double ga = gradient(a);
			if (ga >maxViolation && aa < C)
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
		tVariable& vari = variables[i];
		std::size_t ii = vari.i;
		unsigned int pi = vari.p;
		unsigned int yi = examples[ii].y;
		double di = vari.diagonal;
		double gi = gradient(i);
		QpFloatType* k = kernelMatrix.row(ii, 0, activeEx);
		j = i;
		double gain_i = gi * gi / di;
		double bestgain = gain_i;
		for (std::size_t a=0; a<activeEx; a++)
		{
			tExample const& exa = examples[a];
			unsigned int ya = exa.y;
			typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * (yi * cardP + pi) + ya);
			QpFloatType def = row.defaultvalue;
			
			for (std::size_t pf=0, b=0; pf < cardP; pf++)
			{
				std::size_t f = exa.var[pf];
				double qif = def * k[a];
				//chck whether we are at an existing element of the sparse row
				if( b != row.size && pf == row.entry[b].index){
					qif = row.entry[b].value * k[a];
					++b;//move to next element
				}
				double gain = calculateGain(f,qif,di,gi, gain_i);
				if( bestgain < gain){
					j = f;
					bestgain = gain;
					if(bestgain >= 1e100)
						break;
				}
			}
		}

		return maxViolation;
	}
	
	double calculateGain(std::size_t f,double qif, double di, double gi, double gain_i)const{
		double af = alpha(f);
		double gf = gradient(f);
		double df = variables[f].diagonal;
		if (qif == 0.0)
		{
			if ((af > 0.0 && gf < 0.0) || (af < C && gf > 0.0))
			{
				return gain_i + gf * gf / df;
			}
		}else{
			if ((af > 0.0 && gf < 0.0) || (af < C && gf > 0.0))
			{
				double diag_q = di * df;
				double det_q = diag_q - qif * qif;
				if (det_q < 1e-12 * diag_q)
				{
					if ((di == 0.0 && df == 0.0) || di * gf - df * gi != 0.0)
					{ 
						return 1.e100;//infty
					}
					else
					{
						double g2 = gf*gf + gi*gi;
						return (g2*g2) / (gf*gf*df + 2.0*gf*gi*qif + gi*gi*di);
					}
				}
				else
				{
					return (gf*gf*di - 2.0*gf*gi*qif + gi*gi*df) / det_q;
				}
			}
		}
		return 0;
	}

	//! Shrink the problem
	void shrink(double epsilon)
	{
		if (! bUnshrinked)
		{
			double largest = 0.0;
			for (std::size_t a = 0; a < activeVar; a++)
			{
				if (alpha(a) < C)
				{
					if (gradient(a) > largest)
						largest = gradient(a);
				}
				if (alpha(a) > 0.0)
				{
					if (-gradient(a) > largest) 
						largest = -gradient(a);
				}
			}
			if (largest < 10.0 * epsilon)
			{
				// unshrink the problem at this accuracy level
				unshrink(epsilon);
				bUnshrinked = true;
			}
		}

		// shrink numVariables
		bool se = false;
		for (std::size_t i = activeVar; i > 0; i--)
		{
			std::size_t a = i-1;
			double v = alpha(a);
			double g = gradient(a);

			if ((v == 0.0 && g <= 0.0) || (v == C && g >= 0.0))
			{
				// In this moment no feasible step including this variables
				// can improve the objective. Thus deactivate the variables.
				std::size_t e = variables[a].i;
				deactivateVariable(a);
				if (examples[e].active == 0)
				{
					se = true;
				}
			}
		}

		if (se)
		{
			// exchange examples such that shrinked examples
			// are moved to the ends of the lists
			for (std::size_t a = activeEx - 1; a >= 0; a--)
			{
				if (examples[a].active == 0) deactivateExample(a);
			}
			kernelMatrix.setMaxCachedIndex(activeEx);
		}
	}

	//! Activate all numVariables
	void unshrink(double epsilon)
	{
		if (activeVar == numVariables) return;

		// compute the inactive gradient components (quadratic time complexity)
		subrange(gradient, activeVar, numVariables) = subrange(linear, activeVar, numVariables);
		for (std::size_t v=0; v<numVariables; v++)
		{
			double mu = alpha(v);
			if (mu == 0.0) continue;

			std::size_t iv = variables[v].i;
			unsigned int pv = variables[v].p;
			unsigned int yv = examples[iv].y;
			unsigned int r = cardP * yv + pv;
			std::vector<QpFloatType> q(numExamples);
			kernelMatrix.row(iv, 0, numExamples, &q[0]);

			for (std::size_t a=0; a<numExamples; a++)
			{
				double k = (q)[a];
				tExample& ex = examples[a];
				typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * r + ex.y);
				QpFloatType def = row.defaultvalue;
				if (def == 0.0)
				{
					for (std::size_t  b=0; b<row.size; b++)
					{
						std::size_t f = ex.var[row.entry[b].index];
						if (f >= activeVar) gradient(f) -= mu * row.entry[b].value * k;
					}
				}
				else
				{
					for (std::size_t b=0; b<row.size; b++)
					{
						std::size_t f = ex.var[row.entry[b].index];
						if (f >= activeVar) gradient(f) -= mu * (row.entry[b].value - def) * k;
					}
					double upd = (mu) * def * (k);
					for (std::size_t  b=ex.active; b<cardP; b++)
					{
						std::size_t f = ex.avar[b];
						SHARK_ASSERT(f >= activeVar);
						gradient(f) -= upd;
					}
				}
			}
		}

		for (std::size_t  i=0; i<numExamples; i++) 
			examples[i].active = cardP;
		activeEx = numExamples;
		activeVar = numVariables;
		//todo: mt: activate line below (new unshrink action) -> verify & test
		//kernelMatrix.setTruncationIndex( activeEx ); //disable cache truncation again
	}

	//! true if the problem has already been unshrinked
	bool bUnshrinked;

	//! shrink a variables
	void deactivateVariable(std::size_t v)
	{
		std::size_t ev = variables[v].i;
		unsigned int iv = variables[v].index;
		unsigned int pv = variables[v].p;
		tExample* exv = &examples[ev];

		std::size_t ih = exv->active - 1;
		std::size_t h = exv->avar[ih];
		variables[v].index = ih;
		variables[h].index = iv;
		std::swap(exv->avar[iv], exv->avar[ih]);
		iv = ih;
		exv->active--;

		std::size_t j = activeVar - 1;
		std::size_t ej = variables[j].i;
		unsigned int ij = variables[j].index;
		unsigned int pj = variables[j].p;
		tExample* exj = &examples[ej];

		// exchange entries in the lists
		std::swap(alpha(v), alpha(j));
		std::swap(gradient(v), gradient(j));
		std::swap(linear(v), linear(j));
		std::swap(variables[v], variables[j]);

		variables[exv->avar[iv]].index = ij;
		variables[exj->avar[ij]].index = iv;
		exv->avar[iv] = j;
		exv->var[pv] = j;
		exj->avar[ij] = v;
		exj->var[pj] = v;

		activeVar--;
	}

	//! shrink an examples
	void deactivateExample(std::size_t e)
	{
		SHARK_ASSERT(e < activeEx);
		std::size_t j = activeEx - 1;

		std::swap(examples[e], examples[j]);

		std::size_t v;
		std::size_t* pe = examples[e].var;
		std::size_t* pj = examples[j].var;
		for (v = 0; v < cardP; v++)
		{
			SHARK_ASSERT(pj[v] >= activeVar);
			variables[pe[v]].i = e;
			variables[pj[v]].i = j;
		}

		// notify the matrix cache
		//kernelMatrix.cacheRowRelease(e);
		//todo: mt: new shrinking action. test & verify, then delete line above
		//kernelMatrix.cacheRedeclareOldest(e);
		kernelMatrix.flipColumnsAndRows(e, j);

		activeEx--;
	}

	//! data structure describing one variables of the problem
	struct tVariable
	{
		std::size_t i;				// index into the example list
		unsigned int p;			// constraint corresponding to this variables
		unsigned int index;		// index into example->numVariables
		double diagonal;			// diagonal entry of the big Q-matrix
	};

	//! data structure describing one training example
	struct tExample
	{
		std::size_t index;			// example index in the dataset, not the example vector!
		unsigned int y;			// label of this example
		unsigned int active;		// number of active numVariables
		std::size_t* var;			// list of all cardP numVariables, in order of the p-index
		std::size_t* avar;			// list of active numVariables
	};

	//! kernel matrix (precomputed matrix or matrix cache)
	Matrix& kernelMatrix;

	//! kernel modifiers
	QpSparseArray<QpFloatType> const& M;			// M(|P|*y_i+p, y_j, q)

	//! complexity constant; upper bound on all variabless
	double C;
	
	//! number of classes in the problem
	unsigned int classes;
	
	//! number of dual numVariables per example
	unsigned int cardP;
	
	//! number of examples in the problem (size of the kernel matrix)
	std::size_t numExamples;

	//! number of numVariables in the problem = examples times cardP
	std::size_t numVariables;
	
	//! linear part of the objective function
	RealVector linear;
	
	//! solution candidate
	RealVector alpha;
	
	//! gradient of the objective function
	//! The gradient array is of fixed size and not subject to shrinking.
	RealVector gradient;

	//! information about each training example
	std::vector<tExample> examples;

	//! information about each variables of the problem
	std::vector<tVariable> variables;

	//! space for the example[i].var pointers
	std::vector<std::size_t> storage1;

	//! space for the example[i].avar pointers
	std::vector<std::size_t> storage2;

	//! number of currently active examples
	std::size_t activeEx;

	//! number of currently active variabless
	std::size_t activeVar;

	//! should the solver use the shrinking heuristics?
	bool useShrinking;
	
	void solveEdge(double& alpha, double g, double Q, double U, double& mu)
	{
		mu = g / Q;
		if (! boost::math::isnormal(mu))
		{
			if (g > 0.0)
			{
				mu = U - alpha;
				alpha = U;
			}
			else
			{
				mu = -alpha;
				alpha = 0.0;
			}
		}
		else
		{
			double a = alpha + mu;
			if (a <= 0.0)
			{
				mu = -alpha;
				alpha = 0.0;
			}
			else if (a >= U)
			{
				mu = U - alpha;
				alpha = U;
			}
			else
			{
				alpha = a;
			}
		}
	}

	//! Exact solver for the S2DO problem for the case
	//! that the sub-problem has box-constraints. The
	//! method updates alpha and in addition returns
	//! the step mu.
	void solve2D_box(double& alphai, double& alphaj,
					double gi, double gj,
					double Qii, double Qij, double Qjj,
					double Ui, double Uj,
					double& mui, double& muj)
	{
		// try the free solution first
		double detQ = Qii * Qjj - Qij * Qij;
		mui = (Qjj * gi - Qij * gj) / detQ;
		muj = (Qii * gj - Qij * gi) / detQ;
		double opti = alphai + mui;
		double optj = alphaj + muj;
		if (boost::math::isnormal(opti) && boost::math::isnormal(optj) && opti > 0.0 && optj > 0.0 && opti < Ui && optj < Uj)
		{
			alphai = opti;
			alphaj = optj;
			return;
		}

		// otherwise process all edges
		struct tEdgeSolution
		{
			double alphai;
			double alphaj;
			double mui;
			double muj;
		};
		tEdgeSolution solution[4];
		tEdgeSolution* sol = solution;
		tEdgeSolution* best = solution;
		double gain, bestgain = 0.0;
		// edge \alpha_1 = 0
		{
			sol->alphai = 0.0;
			sol->alphaj = alphaj;
			sol->mui = -alphai;
			solveEdge(sol->alphaj, gj + Qij * alphai, Qjj, Uj, sol->muj);
			gain = sol->mui * (gi - 0.5 * (Qii*sol->mui + Qij*sol->muj))
					+ sol->muj * (gj - 0.5 * (Qij*sol->mui + Qjj*sol->muj));
			if (gain > bestgain) { bestgain = gain; best = sol; }
			sol++;
		}
		// edge \alpha_2 = 0
		{
			sol->alphai = alphai;
			sol->alphaj = 0.0;
			sol->muj = -alphaj;
			solveEdge(sol->alphai, gi + Qij * alphaj, Qii, Ui, sol->mui);
			gain = sol->mui * (gi - 0.5 * (Qii*sol->mui + Qij*sol->muj))
					+ sol->muj * (gj - 0.5 * (Qij*sol->mui + Qjj*sol->muj));
			if (gain > bestgain) { bestgain = gain; best = sol; }
			sol++;
		}
		// edge \alpha_1 = U_1
		{
			sol->alphai = Ui;
			sol->alphaj = alphaj;
			sol->mui = Ui - alphai;
			solveEdge(sol->alphaj, gj - Qij * sol->mui, Qjj, Uj, sol->muj);
			gain = sol->mui * (gi - 0.5 * (Qii*sol->mui + Qij*sol->muj))
					+ sol->muj * (gj - 0.5 * (Qij*sol->mui + Qjj*sol->muj));
			if (gain > bestgain) { bestgain = gain; best = sol; }
			sol++;
		}
		// edge \alpha_2 = U_2
		{
			sol->alphai = alphai;
			sol->alphaj = Uj;
			sol->muj = Uj - alphaj;
			solveEdge(sol->alphai, gi - Qij * sol->muj, Qii, Ui, sol->mui);
			gain = sol->mui * (gi - 0.5 * (Qii*sol->mui + Qij*sol->muj))
					+ sol->muj * (gj - 0.5 * (Qij*sol->mui + Qjj*sol->muj));
			if (gain > bestgain) { bestgain = gain; best = sol; }
			sol++;
		}
		alphai = best->alphai;
		alphaj = best->alphaj;
		mui = best->mui;
		muj = best->muj;
	}
};
#undef ITERATIONS_BETWEEN_SHRINKING

template<class Matrix>
class BiasSolver{
public:
	typedef typename Matrix::QpFloatType QpFloatType;
	BiasSolver(QpMcBoxDecomp<Matrix>* solver) : solver(solver){}
		
	void solve(
		RealVector& bias,
		QpStoppingCondition& stop,
		QpSparseArray<QpFloatType> const& nu,
		bool sumToZero = false
	){
		std::size_t classes = bias.size();
		std::size_t numExamples = solver->getNumExamples();
		std::size_t cardP = solver->getCardP();
		RealVector stepsize(classes, 0.01);
		RealVector prev(classes,0);
		RealVector step(classes);
		
		do{
			solver->solve(stop);

			// Rprop loop to update the bias
			while (true)
			{
				RealMatrix dualGradient = solver->solutionGradient();
				// compute the primal gradient w.r.t. bias
				RealVector grad(classes,0);

				for (std::size_t i=0; i<numExamples; i++){
					for (std::size_t p=0; p<cardP; p++){
						double g = dualGradient(i,p);
						if (g > 0.0)
						{
							unsigned int y = solver->label(i);
							typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP + p);
							for (std::size_t b=0; b<row.size; b++) 
								grad(row.entry[b].index) -= row.entry[b].value;
						}
					}
				}

				if (sumToZero)
				{
					// project the gradient
					double mean = sum(grad) / (double)classes;
					grad -= RealScalarVector(classes, mean);
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
					step -= RealScalarVector(classes, mean);
				}

				// update the solution and the dual gradient
				bias += step;
				performBiasUpdate(step,nu);
				//~ std::cout<<grad<<" "<<solver->checkKKT()<<" "<<stepsize<<" "<<bias<<std::endl;
			
				
				// stopping criterion
				if (max(stepsize) < 0.01 * stop.minAccuracy) break;
			}
		}while(solver->checkKKT()> stop.minAccuracy);
	}
private:
	void performBiasUpdate(
		RealVector const& step, QpSparseArray<QpFloatType> const& nu
	){
		std::size_t numExamples = solver->getNumExamples();
		std::size_t cardP = solver->getCardP();
		RealMatrix deltaLinear(numExamples,cardP,0.0);
		for (std::size_t i=0; i<numExamples; i++){
			for (std::size_t p=0; p<cardP; p++){
				unsigned int y = solver->label(i);
				// delta = \sum_m \nu_{m,p,y_i} \Delta b(m)
				typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP +p);
				for (std::size_t b=0; b<row.size; b++)
				{
					deltaLinear(i,p) -= row.entry[b].value * step(row.entry[b].index);
				}
			}
		}
		solver->addDeltaLinear(deltaLinear);
		
	}
	QpMcBoxDecomp<Matrix>* solver;
};


}
#endif
