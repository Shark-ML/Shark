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


#ifndef SHARK_ALGORITHMS_QP_QPMCDECOMP_H
#define SHARK_ALGORITHMS_QP_QPMCDECOMP_H

#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Algorithms/QP/QpSparseArray.h>
// #include <shark/Algorithms/LP/GLPK.h>
#include <shark/Core/Timer.h>
#include <shark/Data/Dataset.h>


namespace shark {


#define ITERATIONS_BETWEEN_SHRINKING 1000

//todo: O.K.: inline fucntions?!? 
#define GRADIENT_UPDATE(r, from, to, mu, q) \
{ \
	std::size_t a, b, p; \
	for (a=(from); a<(to); a++) \
	{ \
		double k = (q)[a]; \
		tExample& ex = example[a]; \
		typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * (r) + ex.y); \
		QpFloatType def = row.defaultvalue; \
		if (def == 0.0) \
		{ \
			for (b=0; b<row.size; b++) \
			{ \
				p = row.entry[b].index; \
				gradient(ex.var[p]) -= (mu) * row.entry[b].value * k; \
			} \
		} \
		else \
		{ \
			for (b=0; b<row.size; b++) \
			{ \
				p = row.entry[b].index; \
				gradient(ex.var[p]) -= (mu) * (row.entry[b].value - def) * k; \
			} \
			double upd = (mu) * def * k; \
			for (b=0; b<ex.active; b++) gradient(ex.avar[b]) -= upd; \
		} \
	} \
}

#define GAIN_SELECTION_BOX(qif) \
	f = exa.var[pf]; \
	if (f >= activeVar) continue; \
	double gf = gradient(f); \
	double af = alpha(f); \
	if ((af > 0.0 && gf < 0.0) || (af < C && gf > 0.0)) \
	{ \
		double df = variable[f].diagonal; \
		double diag_q = di * df; \
		double det_q = diag_q - qif * qif; \
		if (det_q < 1e-12 * diag_q) \
		{ \
			if (di == 0.0 && df == 0.0) \
			{ if (f != i) { j = f; return ret; } } \
			else \
			{ \
				if (di * gf - df * gi != 0.0) \
				{ if (f != i) { j = f; return ret; } } \
				else \
				{ \
					double g2 = gf*gf + gi*gi; \
					double gain = (g2*g2) / (gf*gf*df + 2.0*gf*gi*qif + gi*gi*di); \
					if (gain > bestgain) { if (f != i) { bestgain = gain; j = f; } } \
				} \
			} \
		} \
		else \
		{ \
			double gain = (gf*gf*di - 2.0*gf*gi*qif + gi*gi*df) / det_q; \
			if (gain > bestgain) { if (f != i) { bestgain = gain; j = f; } } \
		} \
	}

#define GAIN_SELECTION_TRIANGLE(qif) \
	f = exa.var[pf]; \
	if (f >= activeVar) continue; \
	double gf = gradient(f); \
	double af = alpha(f); \
	if ((af > 0.0 && gf < 0.0) || (varsum < C && gf > 0.0)) \
	{ \
		double df = variable[f].diagonal; \
		double diag_q = di * df; \
		double det_q = diag_q - qif * qif; \
		if (det_q < 1e-12 * diag_q) \
		{ \
			if (di == 0.0 && df == 0.0) \
			{ if (f != i) { j = f; return ret; } } \
			else \
			{ \
				if (di * gf - df * gi != 0.0) \
				{ if (f != i) { j = f; return ret; } } \
				else \
				{ \
					double g2 = gf*gf + gi*gi; \
					double gain = (g2*g2) / (gf*gf*df + 2.0*gf*gi*qif + gi*gi*di); \
					if (gain > bestgain) { if (f != i) { bestgain = gain; j = f; } } \
				} \
			} \
		} \
		else \
		{ \
			double gain = (gf*gf*di - 2.0*gf*gi*qif + gi*gi*df) / det_q; \
			if (gain > bestgain) { if (f != i) { bestgain = gain; j = f; } } \
		} \
	}


//!
//! \brief Quadratic program solver for multi class SVM problems
//!
//! \par
//! This quadratic program solver solves the following primal SVM problem:<br><br>
//! \f$ \min_{w, b, \xi} \quad \frac{1}{2} \sum_c \|w_c\|^2 + C \cdot \sum_{i,r} \xi_{i,r} \f$ <br>
//! \f$ \text{s.t.} \quad \forall i, p: \quad \sum_c \nu_{y_i,p,c} \big( \langle w_c, \phi(x_i) \rangle + b_c \big) \geq \gamma_{p,y_i} - \xi_{i,\rho(p)} \f$ <br>
//! \f$ \text{and}  \quad \forall i, r: \quad \xi_{i,r} \geq 0 \f$ <br>
//! \f$ \text{and}  \quad \left[ \sum_c \langle w_c, \phi(\cdot) \rangle + b_c = 0 \right] \f$
//!
//! \par
//! The last so-called sum-to-zero constraint in square brackets is optinal,
//! and so is the presence of the bias variables b. The index i runs over
//! all training examples, p runs over the constraint index set P, r runs
//! over the slack variables index set R, and c runs over the classes. The
//! coefficients \f$ \nu_{y,p,c} \f$ define the margin functions (absolute
//! or relative), \f$ \gamma_{y,p} \f$ are target margins for these functions,
//! \f$ \phi \f$ is a feature map for the kernel function, and
//! \f$ \rho : P \to R \f$ connects constraints and slack variables, and
//! depends on the surrogate loss function. Currently, the solver supports
//! only the cases \f$ \rho = \operatorname{id} \f$ (sum-loss) and
//! \f$ |R| = 1 \f$ (max-loss).
//!
//! \par
//! The solver applies a mixed strategy between solving the dual w.r.t. the
//! variables w for fixed b, and solving the primal for b with fixed w. This
//! strategy allows for the application of efficient decomposition algorithms
//! even to seemingly involved cases, like the multi-class SVM by
//! Crammer&amp;Singer with bias parameter.
//!
//! \par
//! The implementation of this solver is relatively efficient. It exploits
//! the sparsity of the quadratic term of the dual objective function for
//! relative margins, and it provides caching and shrinking techniques on the
//! level of training examples (and the kernel matrix) and variables
//! (for gradient updates).
//!
template <class Matrix>
class QpMcDecomp
{
public:
	//////////////////////////////////////////////////////////////////
	// The types below define the type used for caching kernel values. The default is float,
	// since this type offers sufficient accuracy in the vast majority of cases, at a memory
	// cost of only four bytes. However, the type definition makes it easy to use double instead
	// (e.g., in case high accuracy training is needed).
	typedef typename Matrix::QpFloatType QpFloatType;
	typedef blas::matrix<QpFloatType> QpMatrixType;
	typedef blas::matrix_row<QpMatrixType> QpMatrixRowType;
	typedef blas::matrix_column<QpMatrixType> QpMatrixColumnType;

	//! Constructor
	//! \param  kernel               kernel matrix - cache or pre-computed matrix
	//! \param  _gamma               _gamma(y, p) is the target margin of constraint p for examples of class y
	//! \param  _rho                 mapping from constraints to slack variables
	//! \param  _nu                  margin coefficients in the format \f$ \nu_{y,p,c} = _\nu(|P|*y+p, c) \f$
	//! \param  _M                   kernel modifiers in the format \f$ M_(y_i, p, y_j, q) = _M(classes*(y_i*|P|+p_i)+y_j, q) \f$
	//! \param  sumToZeroConstraint  enable or disable the sum-to-zero constraint
	QpMcDecomp(Matrix& kernel,
			RealMatrix const& _gamma,
			UIntVector const& _rho,
			QpSparseArray<QpFloatType> const& _nu,
			QpSparseArray<QpFloatType> const& _M,
			bool sumToZeroConstraint)
	: kernelMatrix(kernel)
	, gamma(_gamma)
	, rho(_rho)
	, nu(_nu)
	, M(_M)
	, sumToZero(sumToZeroConstraint)
	{
		useShrinking = true;
		examples = kernelMatrix.size();

		classes = gamma.size1();
		cardP = rho.size();
		cardR = 1;
		unsigned int p;
		for (p=0; p<cardP; p++) if (rho[p] >= cardR) cardR = rho[p] +1;
		variables = cardP * examples;

		SHARK_CHECK(
				(gamma.size2() == cardP) &&
				(nu.width() == classes) &&
				(nu.height() == classes * cardP) &&
				(M.width() == cardP) &&
				(M.height() == classes * cardP * classes) &&
				(cardP <= classes),				// makes sense, but is this necessarily so?
				"[QpMcDecomp::QpMcDecomp] dimension conflict"
		);

		if (cardR != 1 && cardR != cardP)
		{
			// constraint of this specific implementation; for efficiency
			throw SHARKEXCEPTION("[QpMcDecomp::QpMcDecomp] currently this solver supports only bijective or trivial rho");
		}
	}


	//!
	//! \brief solve the quadratic program
	//!
	//! \param  target         class labels in {0, ..., classes-1}
	//! \param  C              regularization constant, upper bound on all variables
	//! \param  solutionAlpha  input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param  stop           stopping condition(s)
	//! \param  prop           solution properties (may be NULL)
	//! \param  solutionBias   input: initial bias parameters \f$ b \f$; output: solution \f$ b^* \f$. If this parameter is NULL, then the corresponding problem without bias is solved.
	//!
	void solve(Data<unsigned int> const& target,
					double C,
					RealVector& solutionAlpha,
					QpStoppingCondition& stop,
					QpSolutionProperties* prop = NULL,
					RealVector* solutionBias = NULL)
	{
		SIZE_CHECK(target.numberOfElements() == examples);
		SIZE_CHECK(solutionAlpha.size() == variables);

		std::size_t v, w, i, e;
		unsigned int p;

		double start_time = Timer::now();

		this->C = C;
		alpha = solutionAlpha;
		bias = solutionBias;

		double dualAccuracy = (bias == NULL) ? stop.minAccuracy : 0.5 * stop.minAccuracy;

		// prepare lists
		linear.resize(variables);
		gradient.resize(variables);
		variable.resize(variables);
		storage1.resize(variables);
		storage2.resize(variables);
		example.resize(examples);

		// prepare solver internal variables
		activeEx = examples;
		activeVar = variables;
		for (v=0, i=0; i<examples; i++)
		{
			unsigned int y = target.element(i);
			example[i].index = i;
			example[i].y = y;
			example[i].active = cardP;
			example[i].var = &storage1[cardP * i];
			example[i].avar = &storage2[cardP * i];
			example[i].varsum = 0.0;
			double k = kernelMatrix.entry(i, i);
			for (p=0; p<cardP; p++, v++)
			{
				variable[v].i = i;
				variable[v].p = p;
				variable[v].index = p;
				double Q = M(classes * (y * cardP + p) + y, p) * k;
				variable[v].diagonal = Q;
				storage1[v] = v;
				storage2[v] = v;
				example[i].varsum += solutionAlpha(v);
				double lin = gamma(y, p);
				if (bias != NULL)
				{
					typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP + p);
					for (e=0; e<row.size; e++) lin -= row.entry[e].value * (*bias)(row.entry[e].index);
				}
				linear(v) = gradient(v) = lin;
			}
		}
		SHARK_ASSERT(v == variables);

		// gradient initialization
		e = (std::size_t)(-1);   // invalid value
		QpFloatType* q = NULL;
		for (v=0, i=0; i<examples; i++)
		{
			unsigned int y = example[i].y;
			for (p=0; p<cardP; p++, v++)
			{
				double av = alpha(v);
				if (av != 0.0)
				{
					std::size_t iv = variable[v].i;
					if (iv != e)
					{
						q = kernelMatrix.row(iv, 0, activeEx);
						e = iv;
					}
					unsigned int r = y*cardP+p;
					GRADIENT_UPDATE(r, 0, activeEx, av, q);
				}
			}
		}

		if (bias != NULL) initializeLP();

		bUnshrinked = false;
		std::size_t checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;

		// initial shrinking (useful for dummy variables and warm starts)
		if (useShrinking) shrink(stop.minAccuracy);

		// decomposition loop
		unsigned long long iter = 0;
		while (iter != stop.maxIterations)
		{
			// select a working set and check for optimality
			double acc = selectWorkingSet(v, w);
			if (acc < dualAccuracy)
			{
				// seems to be optimal

				if (useShrinking)
				{
					// do costly unshrinking
					unshrink(dualAccuracy, true);

					// check again on the whole problem
					if (checkKKT() < dualAccuracy)
					{
						if (bias != NULL)
						{
							solveForBias(dualAccuracy);
							if (checkKKT() < stop.minAccuracy)
							{
								if (prop != NULL) prop->type = QpAccuracyReached;
								break;
							}
						}
						else
						{
							if (prop != NULL) prop->type = QpAccuracyReached;
							break;
						}
					}

					shrink(stop.minAccuracy);
					checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;
				}
				else
				{
					if (bias != NULL)
					{
						solveForBias(dualAccuracy);
						if (checkKKT() < stop.minAccuracy)
						{
							if (prop != NULL) prop->type = QpAccuracyReached;
							break;
						}
					}
					else
					{
						if (prop != NULL) prop->type = QpAccuracyReached;
						break;
					}
				}

				selectWorkingSet(v, w);
			}

			// update
			if (v == w)
			{
				// Limit case of a single variable;
				// this means that there is only one
				// non-optimal variable left.
				std::size_t i = variable[v].i;
				unsigned int p = variable[v].p;
				unsigned int y = example[i].y;
				unsigned int r = cardP * y + p;
				QpFloatType* q = kernelMatrix.row(i, 0, activeEx);
				double Qvv = variable[v].diagonal;
				double mu = gradient(v) / Qvv;
				if (mu < 0.0)
				{
					if (mu <= -alpha(v))
					{
						mu = -alpha(v);
						alpha(v) = 0.0;
					}
					else alpha(v) += mu;
					if (cardR < cardP) example[i].varsum += mu;
				}
				else
				{
					if (cardR < cardP)
					{
						double& varsum = example[i].varsum;
						double max_mu = C - varsum;
						double max_alpha = max_mu + alpha(v);
						if (mu >= max_mu)
						{
							mu = max_mu;
							alpha(v) = max_alpha;
							varsum = C;
						}
						else
						{
							alpha(v) += mu;
							varsum += mu;
						}
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
				}
				GRADIENT_UPDATE(r, 0, activeEx, mu, q);
			}
			else
			{
				// S2DO
				std::size_t iv = variable[v].i;
				unsigned int pv = variable[v].p;
				unsigned int yv = example[iv].y;

				std::size_t iw = variable[w].i;
				unsigned int pw = variable[w].p;
				unsigned int yw = example[iw].y;

				// get the matrix rows corresponding to the working set
				QpFloatType* qv = kernelMatrix.row(iv, 0, activeEx);
				QpFloatType* qw = kernelMatrix.row(iw, 0, activeEx);
				unsigned int rv = cardP*yv+pv;
				unsigned int rw = cardP*yw+pw;

				// get the Q-matrix restricted to the working set
				double Qvv = variable[v].diagonal;
				double Qww = variable[w].diagonal;
				double Qvw = M(classes * rv + yw, pw) * qv[iw];

				// solve the sub-problem
				double mu_v = 0.0;
				double mu_w = 0.0;

				if (cardR < cardP)
				{
					if (iv == iw)
					{
						solve2D_triangle(alpha(v), alpha(w),
								example[iv].varsum,
								gradient(v), gradient(w),
								Qvv, Qvw, Qww,
								mu_v, mu_w);
					}
					else
					{
						double& varsum1 = example[iv].varsum;
						double& varsum2 = example[iw].varsum;
						double U1 = C - (varsum1 - alpha(v));
						double U2 = C - (varsum2 - alpha(w));
						solve2D_box(alpha(v), alpha(w),
								gradient(v), gradient(w),
								Qvv, Qvw, Qww,
								U1, U2,
								mu_v, mu_w);

						// improve numerical stability:
						if (alpha(v) == U1) varsum1 = C;
						else varsum1 += mu_v;
						if (alpha(w) == U2) varsum2 = C;
						else varsum2 += mu_w;
					}
				}
				else
				{
					solve2D_box(alpha(v), alpha(w),
							gradient(v), gradient(w),
							Qvv, Qvw, Qww,
							C, C,
							mu_v, mu_w);
				}

				// update the gradient
				GRADIENT_UPDATE(rv, 0, activeEx, mu_v, qv);
				GRADIENT_UPDATE(rw, 0, activeEx, mu_w, qw);
			}

			checkCounter--;
			if (checkCounter == 0)
			{
				// shrink the problem
				if (useShrinking) shrink(stop.minAccuracy);

				checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;

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

			iter++;
		}

		if (iter == stop.maxIterations)
		{
			if (prop != NULL) prop->type = QpMaxIterationsReached;
		}

		// fill in the solution and compute the objective value
RealVector solutionGradient(variables);
		double objective = 0.0;
		for (v=0; v<variables; v++)
		{
			unsigned int w = cardP * example[variable[v].i].index + variable[v].p;
			solutionAlpha(w) = alpha(v);
solutionGradient(w) = gradient(v);
			objective += (gradient(v) + linear(v)) * alpha(v);
		}
		objective *= 0.5;

		double finish_time = Timer::now();

		if (prop != NULL)
		{
			prop->accuracy = checkKKT();
			prop->value = objective;
			prop->iterations = iter;
			prop->seconds = finish_time - start_time;
		}
/*{
	// dump alphas
	for (std::size_t i=0, e=0; i<examples; i++)
	{
		printf("target(%lu) = %u\n", i, target.element(i));
		for (std::size_t c=0; c<classes; c++, e++)
		{
			printf("(%lu, %lu)     alpha: %g   \tgradient: %g\n", i, c, solutionAlpha(e), solutionGradient(e));
		}
	}
}*/

	}

	//!
	//! \brief solve the quadratic program with SMO
	//!
	//! \param  target         class labels in {0, ..., classes-1}
	//! \param  C              regularization constant, upper bound on all variables
	//! \param  solutionAlpha  input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param  stop           stopping condition(s)
	//! \param  prop           solution properties (may be NULL)
	//! \param  solutionBias   input: initial bias parameters \f$ b \f$; output: solution \f$ b^* \f$. If this parameter is NULL, then the corresponding problem without bias is solved.
	//!
	void solveSMO(Data<unsigned int> const& target,
					double C,
					RealVector& solutionAlpha,
					QpStoppingCondition& stop,
					QpSolutionProperties* prop = NULL,
					RealVector* solutionBias = NULL)
	{
		SIZE_CHECK(target.numberOfElements() == examples);
		SIZE_CHECK(solutionAlpha.size() == variables);

		std::size_t v, w, i, e;
		unsigned int p;

		double start_time = Timer::now();

		this->C = C;
		alpha = solutionAlpha;
		bias = solutionBias;

		double dualAccuracy = (bias == NULL) ? stop.minAccuracy : 0.5 * stop.minAccuracy;

		// prepare lists
		linear.resize(variables);
		gradient.resize(variables);
		variable.resize(variables);
		storage1.resize(variables);
		storage2.resize(variables);
		example.resize(examples);

		// prepare solver internal variables
		activeEx = examples;
		activeVar = variables;
		for (v=0, i=0; i<examples; i++)
		{
			unsigned int y = target.element(i);
			example[i].index = i;
			example[i].y = y;
			example[i].active = cardP;
			example[i].var = &storage1[cardP * i];
			example[i].avar = &storage2[cardP * i];
			example[i].varsum = 0.0;
			for (p=0; p<cardP; p++, v++)
			{
				variable[v].i = i;
				variable[v].p = p;
				variable[v].index = p;
				double Q = M(classes * (y * cardP + p) + y, p) * kernelMatrix.entry(i, i);
				variable[v].diagonal = Q;
				storage1[v] = v;
				storage2[v] = v;
				example[i].varsum += solutionAlpha(v);
				double lin = gamma(y, p);
				if (bias != NULL)
				{
					typename QpSparseArray<QpFloatType>::Row const& row = nu.row(y * cardP + p);
					for (e=0; e<row.size; e++) lin -= row.entry[e].value * (*bias)(row.entry[e].index);
				}
				linear(v) = gradient(v) = lin;
			}
		}

		// gradient initialization
		e = 0xffffffff;
		QpFloatType* q = NULL;
		for (v=0, i=0; i<examples; i++)
		{
			unsigned int y = example[i].y;
			for (p=0; p<cardP; p++, v++)
			{
				double av = alpha(v);
				if (av != 0.0)
				{
					std::size_t iv = variable[v].i;
					if (iv != e)
					{
						q = kernelMatrix.row(iv, 0, activeEx);
						e = iv;
					}
					unsigned int r = y*cardP+p;
					GRADIENT_UPDATE(r, 0, activeEx, av, q);
				}
			}
		}

		if (bias != NULL) initializeLP();

		bUnshrinked = false;
		std::size_t checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;

		// initial shrinking (useful for dummy variables and warm starts)
		if (useShrinking) shrink(stop.minAccuracy);

		// decomposition loop
		unsigned long long iter = 0;
		while (iter != stop.maxIterations)
		{
			// select a working set and check for optimality
			double acc = selectWorkingSetSMO(v, w);
			if (acc < dualAccuracy)
			{
				// seems to be optimal

				if (useShrinking)
				{
					// do costly unshrinking
					unshrink(dualAccuracy, true);

					// check again on the whole problem
					if (checkKKT() < dualAccuracy)
					{
						if (bias != NULL)
						{
							solveForBias(dualAccuracy);
							if (checkKKT() < stop.minAccuracy)
							{
								if (prop != NULL) prop->type = QpAccuracyReached;
								break;
							}
						}
						else
						{
							if (prop != NULL) prop->type = QpAccuracyReached;
							break;
						}
					}

					shrink(stop.minAccuracy);
					checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;
				}
				else
				{
					if (bias != NULL)
					{
						solveForBias(dualAccuracy);
						if (checkKKT() < stop.minAccuracy)
						{
							if (prop != NULL) prop->type = QpAccuracyReached;
							break;
						}
					}
					else
					{
						if (prop != NULL) prop->type = QpAccuracyReached;
						break;
					}
				}

				selectWorkingSetSMO(v, w);
			}

			// update
			{
				std::size_t iv = variable[v].i;
				unsigned int pv = variable[v].p;
				unsigned int yv = example[iv].y;

// 				std::size_t iw = variable[w].i;
// 				unsigned int pw = variable[w].p;
// 				unsigned int yw = example[iw].y;

				// get the matrix rows corresponding to the working set
				QpFloatType* qv = kernelMatrix.row(iv, 0, activeEx);
// 				QpFloatType* qw = kernelMatrix.row(iw, 0, activeEx);
				unsigned int rv = cardP*yv+pv;
// 				unsigned int rw = cardP*yw+pw;

				// get the Q-matrix restricted to the working set
				double Qvv = variable[v].diagonal;
// 				double Qww = variable[w].diagonal;
// 				double Qvw = M(classes * rv + yw, pw) * qv[iw];

				// solve the sub-problem
				double mu_v = 0.0;
// 				double mu_w = 0.0;

				if (cardR < cardP)
				{
					throw SHARKEXCEPTION("[QpMcDecomp::solveSMO] SMO is implemented only for box constraints");
				}
				else
				{
					if (v != w) throw SHARKEXCEPTION("[QpMcDecomp::solveSMO] internal error");
					double gv = gradient(v);
					if (Qvv == 0.0)
					{
						if (gv > 0.0)
						{
							mu_v = C - alpha(v);
							alpha(v) = C;
						}
						else
						{
							mu_v = -alpha(v);
							alpha(v) = 0.0;
						}
					}
					else
					{
						mu_v = gv / Qvv;
						double a = alpha(v) + mu_v;
						if (a <= 0.0)
						{
							mu_v = -alpha(v);
							alpha(v) = 0.0;
						}
						else if (a >= C)
						{
							mu_v = C - alpha(v);
							alpha(v) = C;
						}
						else
						{
							alpha(v) = a;
						}
					}
				}

				// update the gradient
				GRADIENT_UPDATE(rv, 0, activeEx, mu_v, qv);
// 				GRADIENT_UPDATE(rw, 0, activeEx, mu_w, qw);
			}

			checkCounter--;
			if (checkCounter == 0)
			{
				// shrink the problem
				if (useShrinking) shrink(stop.minAccuracy);

				checkCounter = (activeVar < ITERATIONS_BETWEEN_SHRINKING) ? activeVar : ITERATIONS_BETWEEN_SHRINKING;

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

			iter++;
		}

		if (iter == stop.maxIterations)
		{
			if (prop != NULL) prop->type = QpMaxIterationsReached;
		}

		// fill in the solution and compute the objective value
		double objective = 0.0;
		for (v=0; v<variables; v++)
		{
			unsigned int w = cardP * example[variable[v].i].index + variable[v].p;
			solutionAlpha(w) = alpha(v);
			objective += (gradient(v) + linear(v)) * alpha(v);
		}
		objective *= 0.5;

		double finish_time = Timer::now();

		if (prop != NULL)
		{
			prop->accuracy = checkKKT();
			prop->value = objective;
			prop->iterations = iter;
			prop->seconds = finish_time - start_time;
		}
	}

	//! enable/disable shrinking
	void setShrinking(bool shrinking = true)
	{ useShrinking = shrinking; }

protected:
	/// Initialize the linear project member LP
	/// for solving the problem with bias parameters.
	void initializeLP()
	{
/*
		std::size_t rows = variables + 1;
		std::size_t cols = examples*cardR + classes;
		std::size_t v, xi, i, r, p, b, c, y;

		// prepare the linear program description
		lp.setMinimize();
		lp.addRows(rows);
		lp.addColumns(cols);

		// slack variables and box and margin constraints
		std::vector<unsigned int> row(1);		// row indices
		std::vector<unsigned int> col(1);		// column indices
		std::vector<double> value(1);			// coefficients
		unsigned int PperR = cardP / cardR;
		for (v=0, xi=0, i=0; i<examples; i++)
		{
			y = example[i].y;
			for (r=0; r<cardR; r++)
			{
				lp.setObjectiveCoefficient(1+xi+r, 1.0);
				lp.setColumnLowerBounded(1+xi+r, 0.0);
			}
			for (p=0; p<cardP; p++)
			{
				r = p / PperR;
				lp.setRowLowerBounded(1+v+p, gradient(v+p));

				// \xi_{i,p}
				row.push_back(1+v+p);
				col.push_back(1+xi+r);
				value.push_back(1.0);
				typename QpSparseArray<QpFloatType>::Row const& nu_row = nu.row(y * cardP + p);
				for (b=0; b<nu_row.size; b++)
				{
					// \nu_{c,p,y_i} \Delta b_c
					row.push_back(1+v+p);
					col.push_back(1+examples*cardR + nu_row.entry[b].index);
					value.push_back(nu_row.entry[b].value);
				}
			}
			v += cardP;
			xi += cardR;
		}

		// bias parameters and equality constraint
		if (sumToZero) lp.setRowFixed(1+v, 0.0);	// sum-to-zero constraint
		for (c=0; c<classes; c++)
		{
			lp.setColumnFree(1+xi+c);
			lp.setObjectiveCoefficient(1+xi+c, 0.0);

			if (sumToZero)
			{
				// b_c
				row.push_back(1+v);
				col.push_back(1+xi+c);
				value.push_back(1.0);
			}
		}

		// set the matrix connecting variables and constraints
		lp.setConstraintMatrix(row, col, value);
*/
	}

	//! Solve the primal problem with fixed weight vectors
	//! for the bias variables (and the slack variables,
	//! but these are ignored).
	void solveForBias(double epsilon)
	{
		SHARK_CHECK(bias != NULL, "[QpMcDecomp::solveForBias] internal error");

		std::size_t i, v, b;
		unsigned int p, c;
		RealVector stepsize(classes,epsilon);
		RealVector prev(classes,0.0);
		RealVector step(classes);

		// Rprop loop
		while (true)
		{
			// compute the primal gradient w.r.t. bias
			RealVector grad(classes,0.0);
			if (cardR < cardP)
			{
				// simplex case
				for (i=0; i<examples; i++)
				{
					tExample& ex = example[i];
					unsigned int largest_p = cardP;
					double largest_value = 0.0;
					for (p=0; p<cardP; p++)
					{
						std::size_t v = ex.var[p];
						SHARK_ASSERT(v < activeVar);
						double g = gradient(v);
						if (g > largest_value)
						{
							largest_value = g;
							largest_p = p;
						}
					}
					if (largest_p < cardP)
					{
						typename QpSparseArray<QpFloatType>::Row const& row = nu.row(ex.y * cardP + largest_p);
						for (b=0; b<row.size; b++) grad(row.entry[b].index) -= row.entry[b].value;
					}
				}
			}
			else
			{
				// box case
				for (v=0; v<variables; v++)
				{
					double g = gradient(v);
					if (g > 0.0)
					{
						tVariable& var = variable[v];
						tExample& ex = example[var.i];
						typename QpSparseArray<QpFloatType>::Row const& row = nu.row(ex.y * cardP + var.p);
						for (b=0; b<row.size; b++) grad(row.entry[b].index) -= row.entry[b].value;
					}
				}
			}

			if (sumToZero)
			{
				// project the gradient
				double mean = sum(grad) / (double)classes;
				grad -= blas::repeat(mean,classes);
			}

			// Rprop
			for (c=0; c<classes; c++)
			{
				double g = grad(c);
				if (g > 0.0) step(c) = -stepsize(c);
				else if (g < 0.0) step(c) = stepsize(c);

				double gg = prev(c) * grad(c);
				if (gg > 0.0) stepsize(c) *= 1.2;
				else stepsize(c) *= 0.5;
			}
			prev = grad;

			if (sumToZero)
			{
				// project the step
				double mean = sum(step) / (double)classes;
				step -= blas::repeat(mean,classes);
			}

			// update the solution and the dual gradient
			(*bias) += step;
			for (v=0; v<variables; v++)
			{
				tVariable& var = variable[v];
				tExample& ex = example[var.i];

				// delta = \sum_m \nu_{m,p,y_i} \Delta b(m)
				typename QpSparseArray<QpFloatType>::Row const& row = nu.row(ex.y * cardP + var.p);
				double delta = 0.0;
				for (b=0; b<row.size; b++)
				{
					delta += row.entry[b].value * step(row.entry[b].index);
				}
				gradient(v) -= delta;
				linear(v) -= delta;
			}

			// stopping criterion
			double maxstep = 0.0;
			for (c=0; c<classes; c++) if (stepsize(c) > maxstep) maxstep = stepsize(c);
			if (maxstep < 0.01 * epsilon) break;
		}

/*
		std::size_t v, xi, i, b;
// 		unsigned int c, p, r, PperR = cardP / cardR;

		// modify lower bounds
		for (v=0, i=0; i<examples; i++)
		{
			tExample& ex = example[i];
			for (p=0; p<cardP; p++, v++)
			{
				lp.setRowLowerBounded(1+v, gradient(ex.var[p]));
			}
		}

		// define initial vertex
		if (cardR == cardP)
		{
			// box case
			for (v=0, xi=0, i=0; i<examples; i++)
			{
				for (p=0; p<cardP; p++, v++)
				{
					if (alpha(v) == C)
					{
						lp.setRowStatus(1+v, false);
						lp.setColumnStatus(1+v, true);
					}
					else
					{
						lp.setRowStatus(1+v, true);
						lp.setColumnStatus(1+v, false);
					}
				}
			}

			if (sumToZero) lp.setRowStatus(1+variables, true);
			for (c=0; c<classes; c++) lp.setColumnStatus(1+examples*cardR+c, false);
		}
		else
		{
			// simplex case
// throw SHARKEXCEPTION("[solveForBias]  simplex case not implemented yet");
			// TODO!!!

// 			if (sumToZero) lp.setRowStatus(1+variables, true);
// 			for (c=0; c<classes; c++) lp.setColumnStatus(1+examples*cardR+c, false);
		}

		// solve the LP
		lp.solve();

		// read out the solution
		RealVector diff(classes);
		for (c=0; c<classes; c++) diff(c) = lp.solution(1+examples*cardR+c);

		// update the solution and the dual gradient
		(*bias) += diff;
		for (v=0; v<variables; v++)
		{
			tVariable& var = variable[v];
			tExample& ex = example[var.i];

			// delta = \sum_m \nu_{m,p,y_i} \Delta b(m)
			typename QpSparseArray<QpFloatType>::Row const& row = nu.row(ex.y * cardP + var.p);
			double delta = 0.0;
			for (b=0; b<row.size; b++)
			{
				delta += row.entry[b].value * diff(row.entry[b].index);
			}
			gradient(v) -= delta;
			linear(v) -= delta;
		}
printf("[solveForBias] diff=(");
for (c=0; c<classes; c++) printf(" %g", diff(c));
printf(")\n");
printf("[solveForBias]    b=(");
for (c=0; c<classes; c++) printf(" %g", (*bias)(c));
printf(")\n");
*/
	}

	//! Exact solver for the one-dimensional sub-problem<br>
	//! maximize \f$ g \alpha - Q/2 \mu^2 \f$<br>
	//! such that \f$ 0 \leq \alpha \leq U \f$<br>
	//! The method returns the optimal alpha as well as
	//! the step mu leading to the update
	//! \f$ \alpha \leftarrow \alpha + \mu \f$.
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

	//! Exact solver for the S2DO problem for the case
	//! that the sub-problem has simplex-constraints.
	//! The method updates alpha and in addition returns
	//! the step mu.
	void solve2D_triangle(double& alphai, double& alphaj,
					double& alphasum,
					double gi, double gj,
					double Qii, double Qij, double Qjj,
					double& mui, double& muj)
	{
		// try the free solution first
		double V = C - alphasum;
		double U = V + alphai + alphaj;
		double detQ = Qii * Qjj - Qij * Qij;
		mui = (Qjj * gi - Qij * gj) / detQ;
		muj = (Qii * gj - Qij * gi) / detQ;
		double opti = alphai + mui;
		double optj = alphaj + muj;
		if (boost::math::isnormal(opti) && boost::math::isnormal(optj) && opti > 0.0 && optj > 0.0 && opti + optj < U)
		{
			alphai = opti;
			alphaj = optj;
			alphasum += mui + muj;
			if (alphasum > C) alphasum = C;		// for numerical stability
			return;
		}

		// otherwise process all edges
		struct tEdgeSolution
		{
			double alphai;
			double alphaj;
			double alphasum;
			double mui;
			double muj;
		};
		tEdgeSolution solution[3];
		tEdgeSolution* sol = solution;
		tEdgeSolution* best = NULL;
		double gain, bestgain = -1e100;
		// edge \alpha_1 = 0
		{
			sol->alphai = 0.0;
			sol->alphaj = alphaj;
			sol->mui = -alphai;
			solveEdge(sol->alphaj, gj + Qij * alphai, Qjj, V + alphaj, sol->muj);
			sol->alphasum = alphasum + sol->mui + sol->muj;
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
			solveEdge(sol->alphai, gi + Qij * alphaj, Qii, V + alphai, sol->mui);
			sol->alphasum = alphasum + sol->mui + sol->muj;
			gain = sol->mui * (gi - 0.5 * (Qii*sol->mui + Qij*sol->muj))
					+ sol->muj * (gj - 0.5 * (Qij*sol->mui + Qjj*sol->muj));
			if (gain > bestgain) { bestgain = gain; best = sol; }
			sol++;
		}
		// edge \alpha_1 + \alpha_2 = U
		{
			double a = 0.0, mu = 0.0;
			double ggi = gi - (U - alphai) * Qii + alphaj * Qij;
			double ggj = gj - (U - alphai) * Qij + alphaj * Qjj;
			solveEdge(a, ggj - ggi, Qii + Qjj - 2.0 * Qij, U, mu);
			sol->alphai = U - a;
			sol->alphaj = a;
			sol->mui = U - a - alphai;
			sol->muj = a - alphaj;
			sol->alphasum = C;
			gain = sol->mui * (gi - 0.5 * (Qii*sol->mui + Qij*sol->muj))
					+ sol->muj * (gj - 0.5 * (Qij*sol->mui + Qjj*sol->muj));
			if (gain > bestgain) { bestgain = gain; best = sol; }
			sol++;
		}

		alphai = best->alphai;
		alphaj = best->alphaj;
		alphasum = best->alphasum;
		mui = best->mui;
		muj = best->muj;

		// improve numerical stability:
		if (alphai + alphaj < 1e-12 * C) alphai = alphaj = alphasum = 0.0;
		if (alphasum > (1.0 - 1e-12) * C)
		{
			alphasum = C;
			if (alphai > (1.0 - 1e-12) * C) { alphai = C; alphaj = 0.0; }
			else if (alphaj > (1.0 - 1e-12) * C) { alphai = 0.0; alphaj = C; }
		}
	}
/*
	//! return the largest KKT violation
	double kktViolationActive(unsigned int example)
	{
	}

	//! return the largest KKT violation
	double kktViolationAll(unsigned int example)
	{
	}
*/
	//! return the largest KKT violation
	double checkKKT()
	{
		if (cardR == cardP)
		{
			double ret = 0.0;
			std::size_t v;
			for (v=0; v<activeVar; v++)
			{
				double a = alpha(v);
				double g = gradient(v);
				if (a < C)
				{
					if (g > ret) ret = g;
				}
				if (a > 0.0)
				{
					if (-g > ret) ret = -g;
				}
			}
			return ret;
		}
		else
		{
			double ret = 0.0;
			std::size_t i, p, pc, v;
			for (i=0; i<activeEx; i++)
			{
				tExample& ex = example[i];
				pc = ex.active;
				bool cangrow = (ex.varsum < C);
				double up = -1e100;
				double down = 1e100;
				for (p=0; p<pc; p++)
				{
					v = ex.avar[p];
					SHARK_ASSERT(v < activeVar);
					double a = alpha(v);
					double g = gradient(v);
					if (cangrow)
					{
						if (g > up) up = g;
					}
					if (a > 0.0)
					{
						if (g < down) down = g;
					}
				}
				if (up - down > ret) ret = up - down;
				if (up > ret) ret = up;
				if (-down > ret) ret = -down;
			}
			return ret;
		}
	}

	//!
	//! \brief select the working set
	//!
	//! Select one or two variables for the sub-problem
	//! and return the maximal KKT violation. The method
	//! MAY select the same index for i and j. In that
	//! case the working set consists of a single variable.
	//! The working set may be invalid if the method reports
	//! a KKT violation of zero, indicating optimality.
	double selectWorkingSet(std::size_t& i, std::size_t& j)
	{
		if (cardR < cardP)
		{
			// simplex case
			double ret = 0.0;

			// first order selection
			std::size_t e;
			bool two = false;
			for (e=0; e<activeEx; e++)
			{
				tExample& ex = example[e];
				unsigned int b, bc = ex.active;
				if (ex.varsum == C)
				{
					unsigned int b2, a2;
					for (b=0; b<bc; b++)
					{
						std::size_t a = ex.avar[b];
						SHARK_ASSERT(a < activeVar);
						double aa = alpha(a);
						double mga = -gradient(a);
						if (aa > 0.0)
						{
							if (mga > ret)
							{
								ret = mga;
								i = a;
								two = false;
							}
							for (b2=0; b2<bc; b2++)
							{
								a2 = ex.avar[b2];
								SHARK_ASSERT(a2 < activeVar);
								double g2 = gradient(a2) + mga;
								if (g2 > ret)
								{
									ret = g2;
									i = a;
									j = a2;
									two = true;
								}
							}
						}
					}
				}
				else
				{
					double up = -1e100;
					double down = 1e100;
					std::size_t i_up = activeVar, i_down = activeVar;
					for (b=0; b<bc; b++)
					{
						std::size_t v = ex.avar[b];
						SHARK_ASSERT(v < activeVar);
						double a = alpha(v);
						double g = gradient(v);
						if (g > up) { i_up = v; up = g; }
						if (a > 0.0 && g < down) { i_down = v; down = g; }
					}
					if (up - down > ret) { two = true; ret = up - down; i = i_up; j = i_down; }
					if (up > ret) { two = false; ret = up; i = i_up; }
					if (-down > ret) { two = false; ret = -down; i = i_down; }
/*
// old version (wrong, not checking combined working set for KKT)
					for (b=0; b<bc; b++)
					{
						std::size_t a = ex.avar[b];
						SHARK_ASSERT(a < activeVar);
						double aa = alpha(a);
						double ga = gradient(a);
						if (ga > ret)
						{
							ret = ga;
							i = a;
							two = false;
						}
						else if (-ga > ret && aa > 0.0)
						{
							ret = -ga;
							i = a;
							two = false;
						}
					}
*/
				}
			}
			if (two || ret == 0.0) return ret;

			// second order selection
			std::size_t b, f, pf;
			tVariable& vari = variable[i];
			std::size_t ii = vari.i;
			unsigned int pi = vari.p;
			unsigned int yi = example[ii].y;
			double di = vari.diagonal;
			double gi = gradient(i);
			QpFloatType* k = kernelMatrix.row(ii, 0, activeEx);
			j = i;
			double bestgain = 0.0;
			double gain_i = gi * gi / di;
			std::size_t a;
			for (a=0; a<activeEx; a++)
			{
				tExample& exa = example[a];
				double varsum = exa.varsum;
				unsigned int ya = exa.y;
				QpFloatType kiia = k[a];
				typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * (yi * cardP + pi) + ya);
				QpFloatType def = row.defaultvalue;
				if (def == 0.0)
				{
					for (pf=0, b=0; b<row.size; b++)
					{
						for (; pf<row.entry[b].index; pf++)
						{
							f = exa.var[pf];
							if (f >= activeVar) continue;
							double af = alpha(f);
							double gf = gradient(f);
							if ((af > 0.0 && gf < 0.0) || (varsum < C && gf > 0.0))
							{
								double df = variable[f].diagonal;
								double gain = gain_i + gf * gf / df;
								if (gain > bestgain && f != i) { bestgain = gain; j = f; }
							}
						}
						{
							GAIN_SELECTION_TRIANGLE(row.entry[b].value * kiia);
							pf++;
						}
					}
					for (; pf<cardP; pf++)
					{
						f = exa.var[pf];
						if (f >= activeVar) continue;
						double af = alpha(f);
						double gf = gradient(f);
						if ((af > 0.0 && gf < 0.0) || (varsum < C && gf > 0.0))
						{
							double df = variable[f].diagonal;
							double gain = gain_i + gf * gf / df;
							if (gain > bestgain && f != i) { bestgain = gain; j = f; }
						}
					}
				}
				else
				{
					for (pf=0, b=0; b<row.size; b++)
					{
						for (; pf<row.entry[b].index; pf++)
						{
							GAIN_SELECTION_TRIANGLE(def * kiia);
						}
						{
							GAIN_SELECTION_TRIANGLE(row.entry[b].value * kiia);
							pf++;
						}
					}
					for (; pf<cardP; pf++)
					{
						GAIN_SELECTION_TRIANGLE(def * kiia);
					}
				}
			}

			return ret;
		}
		else
		{
			// box case
			double ret = 0.0;

			// first order selection
			std::size_t a;
			for (a=0; a<activeVar; a++)
			{
				double aa = alpha(a);
				double ga = gradient(a);
				if (ga > ret && aa < C)
				{
					ret = ga;
					i = a;
				}
				else if (-ga > ret && aa > 0.0)
				{
					ret = -ga;
					i = a;
				}
			}
			if (ret == 0.0) return ret;

			// second order selection
			std::size_t b, f, pf;
			tVariable& vari = variable[i];
			std::size_t ii = vari.i;
			unsigned int pi = vari.p;
			unsigned int yi = example[ii].y;
			double di = vari.diagonal;
			double gi = gradient(i);
			QpFloatType* k = kernelMatrix.row(ii, 0, activeEx);
			j = i;
			double bestgain = 0.0;
			double gain_i = gi * gi / di;
			for (a=0; a<activeEx; a++)
			{
				tExample& exa = example[a];
				unsigned int ya = exa.y;
				QpFloatType kiia = k[a];
				typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * (yi * cardP + pi) + ya);
				QpFloatType def = row.defaultvalue;
				if (def == 0.0)
				{
					for (pf=0, b=0; b<row.size; b++)
					{
						for (; pf<row.entry[b].index; pf++)
						{
							f = exa.var[pf];
							if (f >= activeVar) continue;
							double af = alpha(f);
							double gf = gradient(f);
							if ((af > 0.0 && gf < 0.0) || (af < C && gf > 0.0))
							{
								double df = variable[f].diagonal;
								double gain = gain_i + gf * gf / df;
								if (gain > bestgain && f != i) { bestgain = gain; j = f; }
							}
						}
						{
							GAIN_SELECTION_BOX(row.entry[b].value * kiia);
							pf++;
						}
					}
					for (; pf<cardP; pf++)
					{
						f = exa.var[pf];
						if (f >= activeVar) continue;
						double af = alpha(f);
						double gf = gradient(f);
						if ((af > 0.0 && gf < 0.0) || (af < C && gf > 0.0))
						{
							double df = variable[f].diagonal;
							double gain = gain_i + gf * gf / df;
							if (gain > bestgain && f != i) { bestgain = gain; j = f; }
						}
					}
				}
				else
				{
					for (pf=0, b=0; b<row.size; b++)
					{
						for (; pf<row.entry[b].index; pf++)
						{
							GAIN_SELECTION_BOX(def * kiia);
						}
						{
							GAIN_SELECTION_BOX(row.entry[b].value * kiia);
							pf++;
						}
					}
					for (; pf<cardP; pf++)
					{
						GAIN_SELECTION_BOX(def * kiia);
					}
				}
			}

			return ret;
		}
	}

	//!
	//! \brief select the working set for SMO
	//!
	//! Select one or two variables for the sub-problem
	//! and return the maximal KKT violation. The method
	//! MAY select the same index for i and j. In that
	//! case the working set consists of a single variable.
	//! The working set may be invalid if the method reports
	//! a KKT violation of zero, indicating optimality.
	double selectWorkingSetSMO(std::size_t& i, std::size_t& j)
	{
		if (cardR < cardP)
		{
			// simplex case
			throw SHARKEXCEPTION("[QpMcDecomp::selectWorkingSetSMO] SMO is implemented only for box constraints");
		}
		else
		{
			// box case
			double ret = 0.0;

			// second order selection
			std::size_t a;
			double bestgain = 0.0;
			for (a=0; a<activeVar; a++)
			{
				double aa = alpha(a);
				double ga = gradient(a);
				if (ga > 0.0 && aa < C)
				{
					double gain = ga * ga / variable[a].diagonal;
					if (gain > bestgain)
					{
						i = a;
						bestgain = gain;
					}
					if (ga > ret) ret = ga;
				}
				else if (ga < 0.0 && aa > 0.0)
				{
					double gain = ga * ga / variable[a].diagonal;
					if (gain > bestgain)
					{
						i = a;
						bestgain = gain;
					}
					if (-ga > ret) ret = -ga;
				}
			}
			j = i;
			return ret;
		}
	}

	//! Shrink the problem
	void shrink(double epsilon)
	{
		int a;
		double v, g;

		if (! bUnshrinked)
		{
			double largest = 0.0;
			for (a = 0; a < (int)activeVar; a++)
			{
				if (alpha(a) < C)
				{
					if (gradient(a) > largest) largest = gradient(a);
				}
				if (alpha(a) > 0.0)
				{
					if (-gradient(a) > largest) largest = -gradient(a);
				}
			}
			if (largest < 10.0 * epsilon)
			{
				// unshrink the problem at this accuracy level
				unshrink(epsilon, false);
				bUnshrinked = true;
				return;
			}
		}

		// shrink variables
		bool se = false;
		for (a = activeVar - 1; a >= 0; a--)
		{
			v = alpha(a);
			g = gradient(a);

			if ((v == 0.0 && g <= 0.0) || (v == C && g >= 0.0))
			{
				// In this moment no feasible step including this variable
				// can improve the objective. Thus deactivate the variable.
				std::size_t e = variable[a].i;
				deactivateVariable(a);
				if (example[e].active == 0)
				{
					se = true;
				}
			}
		}

		if (se)
		{
			// exchange examples such that shrinked examples
			// are moved to the ends of the lists
			for (a = activeEx - 1; a >= 0; a--)
			{
				if (example[a].active == 0) deactivateExample(a);
			}

			// shrink the corresponding cache entries
			//~ for (a = 0; a < (int)activeEx; a++)
			//~ {
				//~ if (kernelMatrix.getCacheRowSize(a) > activeEx) kernelMatrix.cacheRowResize(a, activeEx);
			//~ }
			//todo: mt: new shrinking action -> test & verify, remove above 3 lines
			//kernelMatrix.setTruncationIndex( activeEx );
			kernelMatrix.setMaxCachedIndex(activeEx);
		}
	}

	//! Activate all variables
	void unshrink(double epsilon, bool complete)
	{
		if (activeVar == variables) return;

		std::size_t v, i;
		double mu;

		// compute the inactive gradient components (quadratic time complexity)
		RealVectorRange(gradient, Range(activeVar, variables)) = ConstRealVectorRange(linear, Range(activeVar, variables));
// 		for (v=activeVar; v<variables; v++)
// 		{
// 			gradient(v) = linear(v);
// 		}
		for (v=0; v<variables; v++)
		{
			mu = alpha(v);
			if (mu == 0.0) continue;

			std::size_t iv = variable[v].i;
			unsigned int pv = variable[v].p;
			unsigned int yv = example[iv].y;
			unsigned int r = cardP * yv + pv;
			std::vector<QpFloatType> q(examples);
			kernelMatrix.row(iv, 0, examples, &q[0]);

			std::size_t a, b, f;
			for (a=0; a<examples; a++)
			{
				double k = (q)[a];
				tExample& ex = example[a];
				typename QpSparseArray<QpFloatType>::Row const& row = M.row(classes * r + ex.y);
				QpFloatType def = row.defaultvalue;
				if (def == 0.0)
				{
					for (b=0; b<row.size; b++)
					{
						f = ex.var[row.entry[b].index];
						if (f >= activeVar) gradient(f) -= mu * row.entry[b].value * k;
					}
				}
				else
				{
					for (b=0; b<row.size; b++)
					{
						f = ex.var[row.entry[b].index];
						if (f >= activeVar) gradient(f) -= mu * (row.entry[b].value - def) * k;
					}
					double upd = (mu) * def * (k);
					for (b=ex.active; b<cardP; b++)
					{
						f = ex.avar[b];
						SHARK_ASSERT(f >= activeVar);
						gradient(f) -= upd;
					}
				}
			}
		}

		for (i=0; i<examples; i++) example[i].active = cardP;
		activeEx = examples;
		activeVar = variables;
		//todo: mt: activate line below (new unshrink action) -> verify & test
		//kernelMatrix.setTruncationIndex( activeEx ); //disable cache truncation again


		if (! complete) shrink(epsilon);
	}

	//! true if the problem has already been unshrinked
	bool bUnshrinked;

	//! shrink a variable
	void deactivateVariable(std::size_t v)
	{
		std::size_t ev = variable[v].i;
		unsigned int iv = variable[v].index;
		unsigned int pv = variable[v].p;
		tExample* exv = &example[ev];

		std::size_t ih = exv->active - 1;
		std::size_t h = exv->avar[ih];
		variable[v].index = ih;
		variable[h].index = iv;
		std::swap(exv->avar[iv], exv->avar[ih]);
		iv = ih;
		exv->active--;

		std::size_t j = activeVar - 1;
		std::size_t ej = variable[j].i;
		unsigned int ij = variable[j].index;
		unsigned int pj = variable[j].p;
		tExample* exj = &example[ej];

		// exchange entries in the lists
		std::swap(alpha(v), alpha(j));
		std::swap(gradient(v), gradient(j));
		std::swap(linear(v), linear(j));
		std::swap(variable[v], variable[j]);

		variable[exv->avar[iv]].index = ij;
		variable[exj->avar[ij]].index = iv;
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

		std::swap(example[e], example[j]);

		std::size_t v;
		std::size_t* pe = example[e].var;
		std::size_t* pj = example[j].var;
		for (v = 0; v < cardP; v++)
		{
			SHARK_ASSERT(pj[v] >= activeVar);
			variable[pe[v]].i = e;
			variable[pj[v]].i = j;
		}

		// notify the matrix cache
		//kernelMatrix.cacheRowRelease(e);
		//todo: mt: new shrinking action. test & verify, then delete line above
		//kernelMatrix.cacheRedeclareOldest(e);
		kernelMatrix.flipColumnsAndRows(e, j);

		activeEx--;
	}

	//! data structure describing one variable of the problem
	struct tVariable
	{
		std::size_t i;				// index into the example list
		unsigned int p;				// constraint corresponding to this variable
		unsigned int index;			// index into example->variables
		double diagonal;			// diagonal entry of the big Q-matrix
	};

	//! data structure describing one training example
	struct tExample
	{
		std::size_t index;			// example index in the dataset, not the example vector!
		unsigned int y;				// label of this example
		unsigned int active;		// number of active variables
		std::size_t* var;			// list of all cardP variables, in order of the p-index
		std::size_t* avar;			// list of active variables
		double varsum;				// sum of all variables corresponding to this example
	};

	//! information about each training example
	std::vector<tExample> example;

	//! information about each variable of the problem
	std::vector<tVariable> variable;

	//! space for the example[i].var pointers
	std::vector<std::size_t> storage1;

	//! space for the example[i].avar pointers
	std::vector<std::size_t> storage2;

	//! number of examples in the problem (size of the kernel matrix)
	std::size_t examples;

	//! number of classes in the problem
	unsigned int classes;

	//! number of dual variables per example
	unsigned int cardP;

	//! number of slack variables per example
	unsigned int cardR;

	//! number of variables in the problem = examples times cardP
	std::size_t variables;

	//! kernel matrix (precomputed matrix or matrix cache)
	Matrix& kernelMatrix;

	//! target margins
	RealMatrix const& gamma;			// \gamma(y, c) = target margin for \mu_c(x, y)

	//! mapping connecting constraints (dual variables) to slack variables
	UIntVector const& rho;

	//! margin coefficients
    QpSparseArray<QpFloatType> const& nu;			// \nu(y, c, m)

	//! kernel modifiers
	QpSparseArray<QpFloatType> const& M;			// M(|P|*y_i+p, y_j, q)

	//! indicates whether there a sum-to-zero constraint in the problem
	bool sumToZero;

	//! complexity constant; upper bound on all variables
	double C;

	//! linear part of the objective function
	RealVector linear;

	//! number of currently active examples
	std::size_t activeEx;

	//! number of currently active variables
	std::size_t activeVar;

	//! gradient of the objective function
	//! The gradient array is of fixed size and not subject to shrinking.
	RealVector gradient;

	//! solution candidate
	RealVector alpha;

	//! solution candidate
	RealVector* bias;

	//! should the solver use the shrinking heuristics?
	bool useShrinking;

	//! linear program for the bias solver
//	LP lp;
};


#undef GRADIENT_UPDATE
#undef GAIN_SELECTION_BOX
#undef GAIN_SELECTION_TRIANGLE


}
#endif
