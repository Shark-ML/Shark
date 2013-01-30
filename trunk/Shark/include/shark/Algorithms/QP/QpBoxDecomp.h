//===========================================================================
/*!
 *  \file QpBoxDecomp.h
 *
 *  \brief Quadratic programming solver for box-constrained probems
 *
 *
 *  \author  T. Glasmachers
 *  \date    2007-2012
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_QP_QPBOXDECOMP_H
#define SHARK_ALGORITHMS_QP_QPBOXDECOMP_H

#include <shark/Core/Timer.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>


namespace shark{


#define ITERATIONS_BETWEEN_SHRINKING 1000

///
/// \brief Quadratic program solver for box-constrained problems
///
/// \par
/// The QpBoxDecomp class is a decomposition-based solver
/// for the quadratic program occuring when training many
/// types of SVMs without bias term.
/// This problem has the following structure (for
/// \f$ \alpha \in R^{\ell} \f$):
///
/// \par
/// maximize \f$ W(\alpha) = v^T \alpha - \frac{1}{2} \alpha^T M \alpha \f$<br>
/// s.t. \f$ l_i \leq \alpha_i \leq u_i \f$ for all \f$ 1 \leq i \leq \ell \f$.
///
/// \par
/// Here, v is any vector and M is a
/// positive definite symmetric quadratic matrix.
/// \f$ l_i \leq u_i \f$ are lower and upper bounds
/// on the variables.
///
template <class Matrix>
class QpBoxDecomp
{
public:

	//////////////////////////////////////////////////////////////////
	// The statements below define the type used for caching kernel values. The default is float,
	// since this type offers sufficient accuracy in the vast majority of cases, at a memory
	// cost of only four bytes. However, the type definition makes it easy to use double instead
	// (e.g., in case high accuracy training is needed).
	typedef typename Matrix::QpFloatType QpFloatType;
	typedef blas::matrix<QpFloatType> QpMatrixType;
	typedef blas::matrix_row<QpMatrixType> QpMatrixRowType;
	typedef blas::matrix_column<QpMatrixType> QpMatrixColumnType;

	/// \brief Constructor
	///
	/// \param  quadraticPart  quadratic part of the objective function and matrix cache
	QpBoxDecomp(Matrix& quadraticPart)
	: quadratic(quadraticPart)
	{
		useShrinking = true;

		wss = 2;

		dimension = quadratic.size();

		// prepare lists
		alpha.resize(dimension, false);
		diagonal.resize(dimension, false);
		permutation.resize(dimension);
		gradient.resize(dimension, false);
		linear.resize(dimension, false);
		boxMin.resize(dimension, false);
		boxMax.resize(dimension, false);

		// prepare the permutation and the diagonal
		for (std::size_t i=0; i<dimension; i++)
		{
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
		}
	}


	///
	/// \brief Solve a quadratic program.
	///
	/// \par
	/// This is the main solver interface method.
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
	void solve(
			const RealVector& linearPart,
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
		QpFloatType* q;

		for (i=0; i<dimension; i++)
		{
			if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpBoxDecomp::warmStart] The feasible region is empty.");
			if (alpha(i) < boxMin(i) || alpha(i) > boxMax(i)) throw SHARKEXCEPTION("[QpBoxDecomp::warmStart] The initial solution is infeasible.");
		}

		active = dimension;

		bUnshrinked = false;
		std::size_t shrinkCounter = (active < ITERATIONS_BETWEEN_SHRINKING) ? active : ITERATIONS_BETWEEN_SHRINKING;

		// decomposition loop
		if (wss == 1)
		{
			while (iter != stop.maxIterations)
			{
				// select a working set and check for optimality
				if (selectWorkingSet(i) < stop.minAccuracy)
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
					if (selectWorkingSet(i) < stop.minAccuracy)
					{
						if (prop != NULL) prop->type = QpAccuracyReached;
						break;
					}
					else
					{
						shrink(stop.minAccuracy);
						shrinkCounter = (active < ITERATIONS_BETWEEN_SHRINKING) ? active : ITERATIONS_BETWEEN_SHRINKING;
						selectWorkingSet(i);
					}
				}

				// SMO update
				{
					double ai = alpha(i);
					double Li = boxMin(i);
					double Ui = boxMax(i);

					// get the matrix row corresponding to the working set
					q = quadratic.row(i, 0, active);

					// update alpha, that is, solve the sub-problem defined by i
					double numerator = gradient(i);
					double denominator = diagonal(i);
					double mu = numerator / denominator;

					// do the update carefully - avoid numerical problems
					if (ai + mu < Li)
					{
						mu = Li - ai;
						alpha(i) = Li;
					}
					else if (ai + mu > Ui)
					{
						mu = Ui - ai;
						alpha(i) = Ui;
					}
					else alpha(i) += mu;

					// update the gradient
					for (a = 0; a < active; a++) gradient(a) -= mu * q[a];
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
		}
		else if (wss == 2)
		{
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
						double a = selectWorkingSet(i, j);
						SHARK_ASSERT(a >= stop.minAccuracy);
					}
				}

				// SMO update
				if (i == j)
				{
					double ai = alpha(i);
					double Li = boxMin(i);
					double Ui = boxMax(i);

					// get the matrix row corresponding to the working set
					q = quadratic.row(i, 0, active);

					// update alpha, that is, solve the sub-problem defined by i
					double numerator = gradient(i);
					double denominator = diagonal(i);
					double mu = numerator / denominator;

					// do the update carefully - avoid numerical problems
					if (ai + mu < Li)
					{
						mu = Li - ai;
						alpha(i) = Li;
					}
					else if (ai + mu > Ui)
					{
						mu = Ui - ai;
						alpha(i) = Ui;
					}
					else alpha(i) += mu;

					// update the gradient
					for (a = 0; a < active; a++) gradient(a) -= mu * q[a];
				}
				else
				{
					double ai = alpha(i);
					double Li = boxMin(i);
					double Ui = boxMax(i);

					double aj = alpha(j);
					double Lj = boxMin(j);
					double Uj = boxMax(j);

					// get the matrix rows corresponding to the working set
					QpFloatType* q_i = quadratic.row(i, 0, active);
					QpFloatType* q_j = quadratic.row(j, 0, active);

					// get the Q-matrix restricted to the working set
					double Qii = diagonal(i);
					double Qjj = diagonal(j);
					double Qij = q_i[j];

					// solve the sub-problem
					double mu_i = 0.0;
					double mu_j = 0.0;
					solve2D(ai, aj,
							gradient(i), gradient(j),
							Qii, Qij, Qjj,
							Li, Ui, Lj, Uj,
							mu_i, mu_j);

					// do the update carefully - avoid numerical problems
					if (ai + mu_i >= Ui)
					{
						mu_i = Ui - ai;
						alpha(i) = Ui;
					}
					else if (ai + mu_i <= Li)
					{
						mu_i = Li - ai;
						alpha(i) = Li;
					}
					else alpha(i) += mu_i;
					if (aj + mu_j >= Uj)
					{
						mu_j = Uj - aj;
						alpha(j) = Uj;
					}
					else if (aj + mu_j <= Lj)
					{
						mu_j = Lj - aj;
						alpha(j) = Lj;
					}
					else alpha(j) += mu_j;

					// update the gradient
					for (a = 0; a < active; a++) gradient(a) -= (mu_i * q_i[a] + mu_j * q_j[a]);
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
		}
		else throw SHARKEXCEPTION("[QpBoxDecomp::Solve] invalid working set size");

		if (iter == stop.maxIterations)
		{
			if (prop != NULL) prop->type = QpMaxIterationsReached;
		}

		// fill in the solution and compute the objective value
		double objective = 0.0;
		solutionVector.resize(dimension);
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
	/// This method moves the current solution by adding a
	/// value to one component.
	void modifyStepBy(std::size_t index, double addToAlpha)
	{
		std::size_t i = 0;
		for (; i<dimension; i++) if (permutation[i] == index) break;
		alpha(i) += addToAlpha;
		QpFloatType* q = quadratic.row(i, 0, dimension);
		for (std::size_t j=0; j<dimension; j++) gradient(j) -= addToAlpha * q[j];
	}

	/// \brief Modify the problem for a warm-start.
	///
	/// \par
	/// This method moves the current solution by changing
	/// a component to a given value.
	void modifyStepTo(std::size_t index, double newAlpha)
	{
		std::size_t i = 0;
		for (; i<dimension; i++) if (permutation[i] == index) break;
		double mu = newAlpha - alpha(i);
		alpha(i) = newAlpha;
		QpFloatType* q = quadratic.row(i, 0, dimension);
		for (std::size_t j=0; j<dimension; j++) gradient(j) -= mu * q[j];
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

	/// enable/disable shrinking
	inline void setShrinking(bool shrinking = true)
	{ useShrinking = shrinking; }

	/// set the number of variables in the working set
	inline void SetWssStrategy(std::size_t num = 1)
	{
		RANGE_CHECK(num == 1 || num == 2);
		wss = num;
	}

protected:
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
	void solve2D(double alphai, double alphaj,
					double gi, double gj,
					double Qii, double Qij, double Qjj,
					double Li, double Ui, double Lj, double Uj,
					double& mui, double& muj)
	{
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

	/// should the solver use the shrinking heuristics?
	bool useShrinking;

	/// number of currently active variables
	std::size_t active;

	/// working set size
	std::size_t wss;

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

	/// \brief Select a working set
	///
	/// \par
	/// \return maximal KKT vioation
	/// \param  i  index of the active variable
	double selectWorkingSet(std::size_t& i)
	{
		double largest = 0.0;
		std::size_t a;

		for (a = 0; a < active; a++)
		{
			double v = alpha(a);
			double g = gradient(a);
			if (v < boxMax(a))
			{
				if (g > largest)
				{
					largest = g;
					i = a;
				}
			}
			if (v > boxMin(a))
			{
				if (-g > largest)
				{
					largest = -g;
					i = a;
				}
			}
		}

		return largest;
	}

	/// \brief Select a working set of size two
	///
	/// \par
	/// \return maximal KKT vioation
	/// \param  i  index of the first active variable
	/// \param  j  index of the second active variable
	double selectWorkingSet(std::size_t& i, std::size_t& j)
	{
		std::size_t a;

		// select first variable i
		// with first order method
		double maxGrad = 0.0;
		for (a = 0; a < active; a++)
		{
			double v = alpha(a);
			double g = gradient(a);
			if (v < boxMax(a))
			{
				if (g > maxGrad)
				{
					maxGrad = g;
					i = a;
				}
			}
			if (v > boxMin(a))
			{
				if (-g > maxGrad)
				{
					maxGrad = -g;
					i = a;
				}
			}
		}
		if (maxGrad == 0.0) return maxGrad;

		double gi = gradient(i);
		QpFloatType* q = quadratic.row(i, 0, active);
		double Qii = diagonal(i);

		// select second variable j
		// with second order method
		double maxGain = 0.0;
		j = i;
		for (a=0; a<active; a++)
		{
			if (a == i) continue;

			double ga = gradient(a);
			if ((alpha(a) > boxMin(a) && ga < 0.0)
					|| (alpha(a) < boxMax(a) && ga > 0.0))
			{
				double Qia = q[a];
				double Qaa = diagonal(a);

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
			return;
		}

		// identify the variables to shrink
		for (a = 0; a < active; a++)
		{
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
