//===========================================================================
/*!
 *  \file QpSvmDecomp.h
 *
 *  \brief Quadratic programming solver for box-constrained probems with equality constraint
 *
 *
 *  \author  T. Glasmachers
 *  \date    2007-2011
 *
 *  \par Copyright (c) 1999-2011:
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_QP_QPSVMDECOMP_H
#define SHARK_ALGORITHMS_QP_QPSVMDECOMP_H


#include <shark/Core/Timer.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>

namespace shark{


#define ITERATIONS_BETWEEN_SHRINKING 1000


///
/// \brief Quadratic program solver for binary SVMs
///
/// \par
/// The QpSvmDecomp class is a decomposition-based solver
/// for the quadratic program occuring when training a
/// standard binary support vector machine (SVM).
/// This problem has the following structure (for
/// \f$ \alpha \in R^{\ell} \f$):
///
/// \par
/// maximize \f$ W(\alpha) = v^T \alpha - \frac{1}{2} \alpha^T M \alpha \f$<br>
/// s.t. \f$ \sum_{i=1}^{\ell} \alpha_i = z \f$ (equality constraint)<br>
/// and \f$ l_i \leq \alpha_i \leq u_i \f$ for all \f$ 1 \leq i \leq \ell \f$ (box constraints).
///
/// \par
/// Here, z is a number, v is any vector and, M is a
/// positive definite symmetric quadratic matrix.
/// \f$ l_i \leq u_i \f$ are lower and upper bounds
/// on the variables.
///
/// \par
/// The quadratic program is special in that it has a
/// special box form of its inequality constraints, and
/// that is has a single specially aligned equality
/// constraint. These properties make the SMO algorithm
/// (Platt, 1999) and variants thereof (Fan et al., 2005;
/// Glasmachers and Igel, 2006) directly applicable.
///
/// \par
/// This solver uses the basic SMO algorithm with
/// caching and shrinking techniques (Joachims, 1998) and
/// a switching between two highly efficient working set
/// selection algorithms based on second order information.
///
template <class Matrix>
class QpSvmDecomp
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

	/// Constructor
	/// \param  quadraticPart  quadratic part of the objective function and matrix cache
	QpSvmDecomp(Matrix& quadraticPart)
	: quadratic(quadraticPart)
	{
		WSS_Strategy = NULL;
		useShrinking = true;

		dimension = quadratic.size();

		// prepare lists
		alpha.resize(dimension);
		diagonal.resize(dimension);
		permutation.resize(dimension);
		gradient.resize(dimension);
		linear.resize(dimension);
		boxMin.resize(dimension);
		boxMax.resize(dimension);

		// prepare the permutation and the diagonal
		for (std::size_t i=0; i<dimension; i++)
		{
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
		}
	}


	///
	/// \brief solve the quadratic program
	///
	/// \par
	/// This is the core method of the QpSvmDecomp class.
	/// It computes the solution of the problem defined by the
	/// parameters. This interface allows for the solution of
	/// multiple problems with the same quadratic part, reusing
	/// the matrix cache, but with arbirtrary linear part, box
	/// constraints and stopping conditions.
	///
	/// \param linearPart       linear part v of the objective function
	/// \param boxLower         vector l of lower bounds
	/// \param boxUpper         vector u of upper bounds
	/// \param solutionVector   input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	/// \param stop             stopping condition(s)
	/// \param prop             solution properties
	///
	void solve(const RealVector& linearPart,
				 const RealVector& boxLower,
				 const RealVector& boxUpper,
				 RealVector& solutionVector,
				 QpStoppingCondition& stop,
				 QpSolutionProperties* prop = NULL)
	{
		SIZE_CHECK(linearPart.size() == dimension);
		SIZE_CHECK(boxLower.size() == dimension);
		SIZE_CHECK(boxUpper.size() == dimension);
		SIZE_CHECK(solutionVector.size() == dimension);

		double start_time = Timer::now();

		// translate problem into internal representation
		for (std::size_t i=0; i<dimension; i++)
		{
			std::size_t j = permutation[i];
			alpha(i) = solutionVector(j);
			linear(i) = linearPart(j);
			boxMin(i) = boxLower(j);
			boxMax(i) = boxUpper(j);
		}

		// compute the gradient
		gradient = linear;
		for (std::size_t i=0; i<dimension; i++)
		{
			double v = alpha(i);
			if (v != 0.0)
			{
				QpFloatType* q = quadratic.row(i, 0, dimension);
				for (std::size_t a=0; a<dimension; a++) gradient(a) -= q[a] * v;
			}
		}

		// solve the problem
		warmStart(solutionVector, stop, prop);

		if (prop != NULL)
		{
			double finish_time = Timer::now();
			prop->seconds = finish_time - start_time;
		}
	}

	///
	/// \brief Solve a quadratic program with "warm-start".
	///
	/// \par
	/// This method assumes a previous, successful run of
	/// the solver. In many circumstances one needs to
	/// solve a series of related problems. The warmStart
	/// method solves the current problem, which can be a
	/// modified version of the original problem (see the
	/// various modify* methds).
	///
	/// \param solutionVector   input: ignored; output: solution \f$ \alpha^* \f$
	/// \param stop             stopping condition(s)
	/// \param prop             solution properties
	///
	void warmStart(
			RealVector& solutionVector,
			QpStoppingCondition& stop,
			QpSolutionProperties* prop = NULL)
	{
		SIZE_CHECK(solutionVector.size() == dimension);

		unsigned long long iter = 0;
		double start_time = Timer::now();

		std::size_t a, i, j = 0;
		QpFloatType* qi;
		QpFloatType* qj;

		for (i=0; i<dimension; i++)
		{
			if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpSvmDecomp::warmStart] The feasible region is empty.");
			if (alpha(i) < boxMin(i) || alpha(i) > boxMax(i)) throw SHARKEXCEPTION("[QpSvmDecomp::warmStart] The initial solution is infeasible.");
		}

		active = dimension;

		bFirst = true;
		bUnshrinked = false;
		std::size_t shrinkCounter = (active < ITERATIONS_BETWEEN_SHRINKING) ? active : ITERATIONS_BETWEEN_SHRINKING;

		selectWSS();

		// decomposition loop
		while (iter != stop.maxIterations)
		{
			// select a working set and check for optimality
			if (selectWorkingSet(i, j) < stop.minAccuracy)
			{
				// seems to be optimal

				if (! useShrinking)
				{
					if (prop != NULL) prop->type = QpAccuracyReached;
					break;
				}

				// do costly unshrinking
				unshrink(stop.minAccuracy, true);

				// check again on the whole problem
				if (selectWorkingSet(i, j) < stop.minAccuracy)
				{
					if (prop != NULL) prop->type = QpAccuracyReached;
					break;
				}
				else
				{
					shrink(stop.minAccuracy);
					shrinkCounter = (active < ITERATIONS_BETWEEN_SHRINKING) ? active : ITERATIONS_BETWEEN_SHRINKING;
					selectWorkingSet(i, j);
				}
			}

			// SMO update
			{
				double ai = alpha(i);
				double aj = alpha(j);
				double Ui = boxMax(i);
				double Lj = boxMin(j);

				// get the matrix rows corresponding to the working set
				qi = quadratic.row(i, 0, active);
				qj = quadratic.row(j, 0, active);

				// update alpha, that is, solve the sub-problem defined by i and j
				double numerator = gradient(i) - gradient(j);
				double denominator = diagonal(i) + diagonal(j) - 2.0 * qi[j];
				double mu = numerator / denominator;

				// do the update carefully - avoid numerical problems
				if (mu >= std::min(Ui - ai, aj - Lj))
				{
					if (Ui - ai > aj - Lj)
					{
						mu = aj - Lj;
						alpha(i) += mu;
						alpha(j) = Lj;
					}
					else if (Ui - ai < aj - Lj)
					{
						mu = Ui - ai;
						alpha(i) = Ui;
						alpha(j) -= mu;
					}
					else
					{
						mu = Ui - ai;
						alpha(i) = Ui;
						alpha(j) = Lj;
					}
				}
				else
				{
					alpha(i) += mu;
					alpha(j) -= mu;
				}

				// update the gradient
				for (a = 0; a < active; a++) gradient(a) -= mu * (qi[a] - qj[a]);
			}

			shrinkCounter--;
			if (shrinkCounter == 0)
			{
				// shrink the problem
				if (useShrinking) shrink(stop.minAccuracy);

				shrinkCounter = (active < ITERATIONS_BETWEEN_SHRINKING) ? active : ITERATIONS_BETWEEN_SHRINKING;

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
		for (i = 0; i < dimension; i++)
		{
			solutionVector(permutation[i]) = alpha(i);
			objective += (gradient(i) + linear(i)) * alpha(i);
		}
		objective *= 0.5;

		double finish_time = Timer::now();

		if (prop != NULL)
		{
			prop->accuracy = selectWorkingSet(i, j);
			prop->value = objective;
			prop->iterations = iter;
			prop->seconds = finish_time - start_time;
		}
	}

	/// \brief Modify the problem for a warm-start.
	///
	/// \par
	/// This method replaces the box constraints with new
	/// bounds.
	void modifyBoxConstraints(RealVector const& lower, RealVector const& upper)
	{
		for (std::size_t i=0; i<dimension; i++)
		{
			std::size_t j = permutation[i];
			boxMin(i) = lower(j);
			boxMax(i) = upper(j);
		}
	}

	/// \brief Modify the problem for a warm-start.
	///
	/// \par
	/// This method adds a vector to the linear part of the
	/// objective function. In effect, the same vector is
	/// added to the current gradient.
	void modifyLinearPart(RealVector const& addToLinear)
	{
		for (std::size_t i=0; i<dimension; i++)
		{
			std::size_t j = permutation[i];
			linear(i) += addToLinear(j);
			gradient(i) += addToLinear(j);
		}
	}

	/// \brief Modify the problem for a warm-start.
	///
	/// \par
	/// This method moves the current solution by increasing one
	/// and decreasing a second component.
	///
	/// \param  index1        component to increase (step size is added)
	/// \param  index2        component to decrease (step size is subtracted)
	/// \param  stepsize      size of the modification step
	/// \param  snapToBounds  Neighborhood size. The solution "snaps" to the bound in this neighborhood. This reduces numerical problems, increases sparseness, and thus fosters solver speed.
	void modifyStep(std::size_t index1, std::size_t index2, double stepsize, double snapToBounds = 1e-10)
	{
		unsigned int n = 2;
		std::size_t i1 = 0, i2 = 0;
		for (std::size_t i=0; i<dimension; i++)
		{
			if (permutation[i] == index1)
			{
				i1 = i;
				n--;
				if (n == 0) break;
			}
			if (permutation[i] == index2)
			{
				i2 = i;
				n--;
				if (n == 0) break;
			}
		}
		alpha(i1) += stepsize;
		alpha(i2) -= stepsize;
		if (fabs(alpha(i1) - boxMin(i1)) <= snapToBounds) alpha(i1) = boxMin(i1);
		else if (fabs(alpha(i1) - boxMax(i1)) <= snapToBounds) alpha(i1) = boxMax(i1);
		if (fabs(alpha(i2) - boxMin(i2)) <= snapToBounds) alpha(i2) = boxMin(i2);
		else if (fabs(alpha(i2) - boxMax(i2)) <= snapToBounds) alpha(i2) = boxMax(i2);
		QpFloatType* q1 = quadratic.row(i1, 0, dimension);
		QpFloatType* q2 = quadratic.row(i2, 0, dimension);
		for (std::size_t j=0; j<dimension; j++) gradient(j) -= stepsize * (q1[j] - q2[j]);
	}

	///
	/// \brief compute the inner product of a training example with a linear combination of the training examples
	///
	/// \par
	/// This method computes the inner product of
	/// a training example with a linear combination
	/// of all training examples.
	/// The returned value is \f$ \sum_j coeff_j k(x_{index}, x_j) \f$,
	/// where x indicated training examples and k is
	/// the kernel function. The computation makes
	/// use of the kernel cache if possible, but it
	/// does not modify the cache in any way.
	///
	/// \param  index  index of the training example
	/// \param  coeff  list of coefficients of the training examples
	/// \return        result of the inner product
	///
	double computeInnerProduct(std::size_t index, RealVector const& coeff)
	{
		std::size_t e;
		for (e=0; e<dimension; e++)
		{
			if (permutation[e] == index) break;
		}
		SHARK_CHECK(e < dimension, "[QpSvmDecomp::computeInnerProduct] index not found in permutation?!?");
		std::size_t crs_e = quadratic.getCacheRowSize(e);
		QpFloatType* qe = quadratic.row(e, 0, crs_e);

		double ret = 0.0;
		std::size_t i = 0;
		for (; i<crs_e; i++) ret += coeff(permutation[i]) * qe[i];
		for (; i<dimension; i++)
		{
			double c = coeff(permutation[i]);
			if (c != 0.0)
			{
				std::size_t crs_i = quadratic.getCacheRowSize(i);
				if (e < crs_i) ret += c * quadratic.row(i, 0, crs_i)[e];
				else ret += c * quadratic.entry(e, i);
			}
		}

		return ret;
	}

	///
	/// \brief return the gradient of the objective function in the optimum
	///
	RealVector getGradient()
	{
		RealVector grad(dimension);
		for (std::size_t i=0; i<dimension; i++) grad(permutation[i]) = gradient(i);
		return grad;
	}

	/// set the working set strategy parameter
	inline void setStrategy(const char* strategy = NULL)
	{ WSS_Strategy = strategy; }

	/// enable/disable shrinking
	inline void setShrinking(bool shrinking = true)
	{ useShrinking = shrinking; }

protected:
	/// problem dimension
	std::size_t dimension;

	/// representation of the quadratic part of the objective function
	Matrix& quadratic;

	/// linear part of the objective function
	RealVector linear;

	/// box constraint lower bound, that is, minimal variable value
	RealVector boxMin;

	/// box constraint upper bound, that is, maximal variable value
	RealVector boxMax;

	/// solution statistics: number of iterations
// 	unsigned long long iter;

	/// working set selection strategy to follow
	const char* WSS_Strategy;

	/// should the solver use the shrinking heuristics?
	bool useShrinking;

	/// pointer to the currently used working set selection algorithm
	double(QpSvmDecomp::*currentWSS)(std::size_t&, std::size_t&);

	/// number of currently active variables
	std::size_t active;

	/// permutation of the variables alpha, gradient, etc.
	std::vector<std::size_t> permutation;

	/// diagonal matrix entries
	/// The diagonal array is of fixed size and not subject to shrinking.
	RealVector diagonal;

	/// gradient of the objective function
	/// The gradient array is of fixed size and not subject to shrinking.
	RealVector gradient;

	/// Solution candidate
	RealVector alpha;

	/// indicator of the first decomposition iteration
	bool bFirst;

	/// first component of the previous working set
	std::size_t old_i;

	/// second component of the previous working set
	std::size_t old_j;

	/// \brief Select the most violatig pair (MVP)
	///
	/// \return maximal KKT vioation
	///  \param i  first working set component
	///  \param j  second working set component
	double MVP(std::size_t& i, std::size_t& j)
	{
		double largestUp = -1e100;
		double smallestDown = 1e100;

		for (std::size_t a=0; a < active; a++)
		{
			if (alpha(a) < boxMax(a))
			{
				if (gradient(a) > largestUp)
				{
					largestUp = gradient(a);
					i = a;
				}
			}
			if (alpha(a) > boxMin(a))
			{
				if (gradient(a) < smallestDown)
				{
					smallestDown = gradient(a);
					j = a;
				}
			}
		}

		// MVP stopping condition
		return largestUp - smallestDown;
	}

	/// \brief Select a working set according to the hybrid maximum gain (HMG) algorithm
	///
	/// \return maximal KKT vioation
	///  \param i  first working set component
	///  \param j  second working set component
	double HMG(std::size_t& i, std::size_t& j)
	{
		if (bFirst)
		{
			// the cache is empty - use MVP
			bFirst = false;
			return Libsvm28(i, j);		// better: use second order algorithm
		}

		// check the corner condition
		{
			double Li = boxMin(old_i);
			double Ui = boxMax(old_i);
			double Lj = boxMin(old_j);
			double Uj = boxMax(old_j);
			double eps_i = 1e-8 * (Ui - Li);
			double eps_j = 1e-8 * (Uj - Lj);
			if ((alpha(old_i) <= Li + eps_i || alpha(old_i) >= Ui - eps_i)
					&& ((alpha(old_j) <= Lj + eps_j || alpha(old_j) >= Uj - eps_j)))
			{
				return Libsvm28(i, j);		// better: use second order algorithm
			}
		}

		// generic situation: use the MG selection
		std::size_t a;
		double aa, ab;					// alpha values
		double da, db;					// diagonal entries of Q
		double ga, gb;					// gradient in coordinates a and b
		double gain;
		double La, Ua, Lb, Ub;
		double denominator;
		QpFloatType* q;
		double mu_max, mu_star;

		double best = 0.0;
		double mu_best = 0.0;

		// try combinations with b = old_i
		q = quadratic.row(old_i, 0, active);
		ab = alpha(old_i);
		db = diagonal(old_i);
		Lb = boxMin(old_i);
		Ub = boxMax(old_i);
		gb = gradient(old_i);
		for (a = 0; a < active; a++)
		{
			if (a == old_i || a == old_j) continue;

			aa = alpha(a);
			da = diagonal(a);
			La = boxMin(a);
			Ua = boxMax(a);
			ga = gradient(a);

			denominator = (da + db - 2.0 * q[a]);
			mu_max = (ga - gb) / denominator;
			mu_star = mu_max;

			if (aa + mu_star < La) mu_star = La - aa;
			else if (mu_star + aa > Ua) mu_star = Ua - aa;
			if (ab - mu_star < Lb) mu_star = ab - Lb;
			else if (ab - mu_star > Ub) mu_star = ab - Ub;

			gain = mu_star * (2.0 * mu_max - mu_star) * denominator;

			// select the largest gain
			if (gain > best)
			{
				best = gain;
				mu_best = mu_star;
				i = a;
				j = old_i;
			}
		}

		// try combinations with old_j
		q = quadratic.row(old_j, 0, active);
		ab = alpha(old_j);
		db = diagonal(old_j);
		Lb = boxMin(old_j);
		Ub = boxMax(old_j);
		gb = gradient(old_j);
		for (a = 0; a < active; a++)
		{
			if (a == old_i || a == old_j) continue;

			aa = alpha(a);
			da = diagonal(a);
			La = boxMin(a);
			Ua = boxMax(a);
			ga = gradient(a);

			denominator = (da + db - 2.0 * q[a]);
			mu_max = (ga - gb) / denominator;
			mu_star = mu_max;

			if (aa + mu_star < La) mu_star = La - aa;
			else if (mu_star + aa > Ua) mu_star = Ua - aa;
			if (ab - mu_star < Lb) mu_star = ab - Lb;
			else if (ab - mu_star > Ub) mu_star = ab - Ub;

			gain = mu_star * (2.0 * mu_max - mu_star) * denominator;

			// select the largest gain
			if (gain > best)
			{
				best = gain;
				mu_best = mu_star;
				i = a;
				j = old_j;
			}
		}

		// stopping condition
		return fabs(mu_best);
	}

	/// \brief Select a working set according to the second order algorithm of LIBSVM 2.8
	///
	/// \return maximal KKT vioation
	///  \param i  first working set component
	///  \param j  second working set component
	double Libsvm28(std::size_t& i, std::size_t& j)
	{
		i = 0;
		j = 1;

		double largestUp = -1e100;
		double smallestDown = 1e100;
		std::size_t a;

		// find the first index of the MVP
		for (a = 0; a < active; a++)
		{
			if (alpha(a) < boxMax(a))
			{
				if (gradient(a) > largestUp)
				{
					largestUp = gradient(a);
					i = a;
				}
			}
		}
		if (largestUp == -1e100) return 0.0;

		// find the second index using second order information
		QpFloatType* q = quadratic.row(i, 0, active);
		double best = 0.0;
		for (a = 0; a < active; a++)
		{
			if (alpha(a) > boxMin(a))
			{
				if (gradient(a) < smallestDown) smallestDown = gradient(a);

				double grad_diff = largestUp - gradient(a);
				if (grad_diff > 0.0)
				{
					double quad_coef = diagonal(i) + diagonal(a) - 2.0 * q[a];
					if (quad_coef == 0.0) continue;
					double obj_diff = (grad_diff * grad_diff) / quad_coef;

					if (obj_diff > best)
					{
						best = obj_diff;
						j = a;
					}
				}
			}
		}

		if (best == 0.0) return 0.0;		// numerical accuracy reached :(

		// MVP stopping condition
		return largestUp - smallestDown;
	}

	/// \brief Select a working set
	///
	/// \par
	/// This member is implemented as a wrapper to HMG.
	/// \return maximal KKT vioation
	/// \param  i  first working set component
	/// \param  j  second working set component
	double selectWorkingSet(std::size_t& i, std::size_t& j)
	{
		// dynamic working set selection call
		double ret = (this->*(this->currentWSS))(i, j);
		if (gradient(i) < gradient(j)) std::swap(i, j);

		old_i = i;
		old_j = j;
		return ret;
	}

	/// Choose a suitable working set algorithm
	void selectWSS()
	{
		if (WSS_Strategy != NULL && strcmp(WSS_Strategy, "MVP") == 0)
		{
			// most violating pair, used e.g. in LIBSVM 2.71
			currentWSS = &QpSvmDecomp::MVP;
		}
		else if (WSS_Strategy != NULL && strcmp(WSS_Strategy, "HMG") == 0)
		{
			// hybrid maximum gain, suitable for large problems
			currentWSS = &QpSvmDecomp::HMG;
		}
		else if (WSS_Strategy != NULL && strcmp(WSS_Strategy, "LIBSVM28") == 0)
		{
			// LIBSVM 2.8 second order algorithm
			currentWSS = &QpSvmDecomp::Libsvm28;
		}
		else
		{
			// default strategy:
			// use HMG as long as the problem does not fit into the cache,
			// use the LIBSVM 2.8 algorithm afterwards
			if (active * active > quadratic.getMaxCacheSize())
				currentWSS = &QpSvmDecomp::HMG;
			else
				currentWSS = &QpSvmDecomp::Libsvm28;
		}
	}

	/// Shrink the problem
	void shrink(double epsilon)
	{
		double largestUp = -1e100;
		double smallestDown = 1e100;
		std::vector<std::size_t> shrinked;
		std::size_t a;
		double v, g;

		for (a = 0; a < active; a++)
		{
			v = alpha(a);
			g = gradient(a);
			if (v > boxMin(a))
			{
				if (g < smallestDown) smallestDown = g;
			}
			if (v < boxMax(a))
			{
				if (g > largestUp) largestUp = g;
			}
		}

		if (! bUnshrinked && (largestUp - smallestDown < 10.0 * epsilon))
		{
			// unshrink the problem at this accuracy level
			unshrink(epsilon, false);
			bUnshrinked = true;
			selectWSS();
			return;
		}

		// identify the variables to shrink
		for (a = 0; a < active; a++)
		{
			if (a == old_i) continue;
			if (a == old_j) continue;
			v = alpha(a);
			g = gradient(a);

			if (v == boxMin(a))
			{
				if (g > smallestDown) continue;
			}
			else if (v == boxMax(a))
			{
				if (g < largestUp) continue;
			}
			else continue;

			// In this moment no feasible step including this variable
			// can improve the objective. Thus deactivate the variable.
			shrinked.push_back(a);
			if (quadratic.getCacheRowSize(a) > 0) quadratic.cacheRowRelease(a);
		}

		int s, sc = shrinked.size();
		if (sc == 0) return;
		std::size_t new_active = active - sc;

		// exchange variables such that shrinked variables
		// are moved to the ends of the lists.
		std::size_t k, high = active;
		for (s = sc - 1; s >= 0; s--)
		{
			k = shrinked[s];
			high--;

			// exchange the variables "k" and "high"
			flipCoordinates(k, high);
		}

		// shrink the cache entries
		for (a = 0; a < active; a++)
		{
			if (quadratic.getCacheRowSize(a) > new_active) quadratic.cacheRowResize(a, new_active);
		}

		active = new_active;

		selectWSS();
	}

	/// Activate all variables
	void unshrink(double epsilon, bool complete)
	{
		if (active == dimension) return;

		std::size_t i, a;
		QpFloatType* q;
		double v, g;
		double largestUp = -1e100;
		double smallestDown = 1e100;

		// compute the inactive gradient components (quadratic time complexity)
		for (a = active; a < dimension; a++) gradient(a) = linear(a);
		for (i = 0; i < dimension; i++)
		{
			v = alpha(i);
			if (v == 0.0) continue;

			q = quadratic.row(i, active, dimension, true);
			for (a = active; a < dimension; a++) gradient(a) -= q[a] * v;
		}

		if (complete)
		{
			active = dimension;
			return;
		}

		// find largest KKT violations
		for (a = 0; a < dimension; a++)
		{
			g = gradient(a);
			v = alpha(a);

			if (v > boxMin(a) && g < smallestDown) smallestDown = g;
			if (v < boxMax(a) && g > largestUp) largestUp = g;
		}

		// identify the variables to activate
		for (a = active; a < dimension; a++)
		{
			if (a == old_i) continue;
			if (a == old_j) continue;
			g = gradient(a);
			v = alpha(a);

			if (v == boxMin(a))
			{
				if (g <= smallestDown) continue;
			}
			else if (v == boxMax(a))
			{
				if (g >= largestUp) continue;
			}

			flipCoordinates(active, a);
			active++;
		}
	}

	/// true if the problem has already been unshrinked
	bool bUnshrinked;

	/// exchange two variables via the permutation
	void flipCoordinates(std::size_t i, std::size_t j)
	{
		if (i == j) return;

		// check the previous working set
		if (old_i == i) old_i = j;
		else if (old_i == j) old_i = i;

		if (old_j == i) old_j = j;
		else if (old_j == j) old_j = i;

		// exchange entries in the simple lists
		XCHG_A(double, boxMin, i, j);
		XCHG_A(double, boxMax, i, j);
		XCHG_A(double, linear, i, j);
		XCHG_A(double, alpha, i, j);
		XCHG_V(std::size_t, permutation, i, j);
		XCHG_A(double, diagonal, i, j);
		XCHG_A(double, gradient, i, j);

		// notify the matrix cache
		quadratic.flipColumnsAndRows(i, j);
	}
};


}
#endif
