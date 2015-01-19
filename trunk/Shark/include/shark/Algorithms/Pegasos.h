//===========================================================================
/*!
 * 
 *
 * \brief       Pegasos solvers for linear SVMs
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2012
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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


#ifndef SHARK_ALGORITHMS_PEGASOS_H
#define SHARK_ALGORITHMS_PEGASOS_H


#include <shark/LinAlg/Base.h>
#include <shark/Data/Dataset.h>
#include <shark/Rng/GlobalRng.h>
#include <cmath>
#include <iostream>


namespace shark {


///
/// \brief Pegasos solver for linear (binary) support vector machines.
///
template <class VectorType>
class Pegasos
{
public:
	/// \brief Solve the primal SVM problem.
	///
	/// In addition to "standard" Pegasos this solver checks a
	/// meaningful stopping criterion.
	///
	/// The function returns the number of model predictions
	/// during training (this is comparable to SMO iterations).
	template <class WeightType>
	static std::size_t solve(
			LabeledData<VectorType, unsigned int> const& data,  ///< training data
			double C,                                           ///< SVM regularization parameter
			WeightType& w,                                      ///< weight vector
			std::size_t batchsize = 1,                          ///< number of samples in each gradient estimate
			double varepsilon = 0.001)                          ///< solution accuracy (factor by which the primal gradient should be reduced)
	{
		std::size_t ell = data.numberOfElements();
		double lambda = 1.0 / (ell * C);
		SHARK_ASSERT(batchsize > 0);

		double initialPrimal = 1.0;
		double normbound2 = initialPrimal / lambda;     // upper bound for |sigma * w|^2
		double norm_w2 = 0.0;                           // squared norm of w
		double sigma = 1.0;                             // scaling factor for w
		VectorType gradient(w.size());                  // gradient (to be computed in each iteration)
		w = RealZeroVector(w.size());                   // clear does not work on matrix rows (ublas sucks!)

		// pegasos main loop
		std::size_t start = 10;
		std::size_t checkinterval = (2 * ell) / batchsize;
		std::size_t nextcheck = start + ell / batchsize;
		std::size_t predictions = 0;
		for (std::size_t t=start; ; t++)
		{
			// check the stopping criterion: \|gradient\| < epsilon ?
			if (t >= nextcheck)
			{
				// compute the gradient
				gradient = (lambda * sigma * (double)ell) * w;
				for (std::size_t i=0; i<ell; i++)
				{
					VectorType const& x = data(i).input;
					unsigned int y = data(i).label;
					double f = sigma * inner_prod(w, x);
					lg(x, y, f, gradient);
				}
				predictions += ell;

				// compute the norm of the gradient
				double n2 = inner_prod(gradient, gradient);
				double n = std::sqrt(n2) / (double)ell;

				// check the stopping criterion
				if (n < varepsilon)
				{
//					std::cout << "target accuracy reached." << std::endl;
//					std::cout << "accuracy: " << n << std::endl;
//					std::cout << "predictions: " << predictions << std::endl;
					break;
				}

				nextcheck = t + checkinterval;
			}

			// compute the gradient
			gradient.clear();
			bool nonzero = true;
			for (unsigned int i=0; i<batchsize; i++)
			{
				// select the active variable (sample with replacement)
				std::size_t active = Rng::discrete(0, ell-1);
				VectorType const& x = data(active).input;
				unsigned int y = data(active).label;
				SHARK_ASSERT(y < 2);

				// compute the prediction
				double f = sigma * inner_prod(w, x);
				predictions++;

				// compute the loss gradient
				lg(x, y, f, gradient);
			}

			// update
			sigma *= (1.0 - 1.0 / (double)t);
			if (nonzero)
			{
				double eta = 1.0 / (sigma * lambda * t * batchsize);
				gradient *= eta;
				norm_w2 += inner_prod(gradient, gradient) - 2.0 * inner_prod(w, gradient);
				noalias(w) -= gradient;

				// project to the ball
				double n2 = sigma * sigma * norm_w2;
				if (n2 > normbound2) sigma *= std::sqrt(normbound2 / n2);
			}
		}

		// rescale the solution
		w *= sigma;
		return predictions;
	}

protected:
	// gradient of the loss
	static bool lg(
			VectorType const& x,
			unsigned int y,
			double f,
			VectorType& gradient)
	{
		if (y == 0)
		{
			if (f > -1.0)
			{
				gradient += x;
				return true;
			}
		}
		else if (y == 1)
		{
			if (f < 1.0)
			{
				gradient -= x;
				return true;
			}
		}
		return false;
	}
};


///
/// \brief Pegasos solver for linear multi-class support vector machines.
///
template <class VectorType>
class McPegasos
{
public:
	/// \brief Multi-class margin type.
	enum eMarginType
	{
		emRelative,
		emAbsolute,
	};

	/// \brief Multi-class loss function type.
	enum eLossType
	{
		elNaiveHinge,
		elDiscriminativeMax,
		elDiscriminativeSum,
		elTotalMax,
		elTotalSum,
	};

	/// \brief Solve the primal multi-class SVM problem.
	///
	/// In addition to "standard" Pegasos this solver checks a
	/// meaningful stopping criterion.
	///
	/// The function returns the number of model predictions
	/// during training (this is comparable to SMO iterations).
	template <class WeightType>
	static std::size_t solve(
			LabeledData<VectorType, unsigned int> const& data,  ///< training data
			eMarginType margintype,                             ///< margin function type
			eLossType losstype,                                 ///< loss function type
			bool sumToZero,                                     ///< enforce the sum-to-zero constraint?
			double C,                                           ///< SVM regularization parameter
			std::vector<WeightType>& w,                         ///< class-wise weight vectors
			std::size_t batchsize = 1,                          ///< number of samples in each gradient estimate
			double varepsilon = 0.001)                          ///< solution accuracy (factor by which the primal gradient should be reduced)
	{
		SHARK_ASSERT(batchsize > 0);
		std::size_t ell = data.numberOfElements();
		unsigned int classes = w.size();
		SHARK_ASSERT(classes >= 2);
		double lambda = 1.0 / (ell * C);

		double initialPrimal = -1.0;
		LossGradientFunction lg = NULL;
		if (margintype == emRelative)
		{
			if (losstype == elDiscriminativeMax || losstype == elTotalMax)
			{
				// CS case
				initialPrimal = 1.0;
				lg = lossGradientRDM;
			}
			else if (losstype == elDiscriminativeSum || losstype == elTotalSum)
			{
				// WW case
				initialPrimal = classes - 1.0;
				lg = lossGradientRDS;
			}
		}
		else if (margintype == emAbsolute)
		{
			if (losstype == elNaiveHinge)
			{
				// MMR case
				initialPrimal = 1.0;
				lg = lossGradientANH;
			}
			else if (losstype == elDiscriminativeMax)
			{
				// ADM case
				initialPrimal = 1.0;
				lg = lossGradientADM;
			}
			else if (losstype == elDiscriminativeSum)
			{
				// LLW case
				initialPrimal = classes - 1.0;
				lg = lossGradientADS;
			}
			else if (losstype == elTotalMax)
			{
				// ATM case
				initialPrimal = 1.0;
				lg = lossGradientATM;
			}
			else if (losstype == elTotalSum)
			{
				// ATS/OVA case
				initialPrimal = classes;
				lg = lossGradientATS;
			}
		}
		if (initialPrimal <= 0.0 || lg == NULL) throw SHARKEXCEPTION("[McPegasos::solve] the combination of margin and loss is not implemented");

		double normbound2 = initialPrimal / lambda;     // upper bound for |sigma * w|^2
		double norm_w2 = 0.0;                           // squared norm of w
		double sigma = 1.0;                             // scaling factor for w
		double target = initialPrimal * varepsilon;     // target gradient norm
		std::vector<VectorType> gradient(classes);      // gradient (to be computed in each iteration)
		RealVector f(classes);                          // machine prediction (computed for each example)
		for (unsigned int c=0; c<classes; c++)
		{
			gradient[c].resize(w[c].size());
			w[c] = RealZeroVector(w[c].size());
		}

		// pegasos main loop
		std::size_t start = 10;
		std::size_t checkinterval = (2 * ell) / batchsize;
		std::size_t nextcheck = start + ell / batchsize;
		std::size_t predictions = 0;
		for (std::size_t t=start; ; t++)
		{
			// check the stopping criterion: \|gradient\| < epsilon ?
			if (t >= nextcheck)
			{
				// compute the gradient
				for (unsigned int c=0; c<classes; c++) gradient[c] = (lambda * sigma * (double)ell) * w[c];
				for (std::size_t i=0; i<ell; i++)
				{
					VectorType const& x = data(i).input;
					unsigned int y = data(i).label;
					for (unsigned int c=0; c<classes; c++) f(c) = sigma * inner_prod(w[c], x);
					lg(x, y, f, gradient, sumToZero);
				}
				predictions += ell;

				// compute the norm of the gradient
				double n2 = 0.0;
				for (unsigned int c=0; c<classes; c++) n2 += inner_prod(gradient[c], gradient[c]);
				double n = std::sqrt(n2) / (double)ell;

				// check the stopping criterion
				if (n < target)
				{
//					std::cout << "target accuracy reached." << std::endl;
//					std::cout << "accuracy: " << n << std::endl;
//					std::cout << "predictions: " << predictions << std::endl;
					break;
				}

				nextcheck = t + checkinterval;
			}

			// compute the gradient
			for (unsigned int c=0; c<classes; c++) gradient[c].clear();
			bool nonzero = true;
			for (unsigned int i=0; i<batchsize; i++)
			{
				// select the active variable (sample with replacement)
				std::size_t active = Rng::discrete(0, ell-1);
				VectorType const& x = data(active).input;
				unsigned int y = data(active).label;
				SHARK_ASSERT(y < classes);

				// compute the prediction
				for (unsigned int c=0; c<classes; c++) f(c) = sigma * inner_prod(w[c], x);
				predictions++;

				// compute the loss gradient
				lg(x, y, f, gradient, sumToZero);
			}

			// update
			sigma *= (1.0 - 1.0 / (double)t);
			if (nonzero)
			{
				double eta = 1.0 / (sigma * lambda * t * batchsize);
				for (unsigned int c=0; c<classes; c++)
				{
					gradient[c] *= eta;
					norm_w2 += inner_prod(gradient[c], gradient[c]) - 2.0 * inner_prod(w[c], gradient[c]);
					noalias(w[c]) -= gradient[c];
				}

				// project to the ball
				double n2 = sigma * sigma * norm_w2;
				if (n2 > normbound2) sigma *= std::sqrt(normbound2 / n2);
			}
		}

		// rescale the solution
		for (unsigned int c=0; c<classes; c++) w[c] *= sigma;
		return predictions;
	}

protected:
	// Function type for the computation of the gradient
	// of the loss. A return value of true indicates that
	// the gradient is non-zero.
	typedef bool(*LossGradientFunction)(VectorType const&, unsigned int, RealVector const&, std::vector<VectorType>&, bool);

	// absolute margin, naive hinge loss
	static bool lossGradientANH(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		if (f(y) < 1.0)
		{
			gradient[y] -= x;
			if (sumToZero)
			{
				VectorType xx = (1.0 / (f.size() - 1.0)) * x;
				for (std::size_t c=0; c<f.size(); c++) if (c != y) gradient[c] += xx;
			}
			return true;
		}
		else return false;
	}

	// relative margin, max loss
	static bool lossGradientRDM(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		unsigned int argmax = 0;
		double max = -1e100;
		for (std::size_t c=0; c<f.size(); c++)
		{
			if (c != y && f(c) > max)
			{
				max = f(c);
				argmax = c;
			}
		}
		if (f(y) < 1.0 + max)
		{
			gradient[y]      -= x;
			gradient[argmax] += x;
			return true;
		}
		else return false;
	}

	// relative margin, sum loss
	static bool lossGradientRDS(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		bool nonzero = false;
		for (std::size_t c=0; c<f.size(); c++)
		{
			if (c != y && f(y) < 1.0 + f(c))
			{
				gradient[y] -= x;
				gradient[c] += x;
				nonzero = true;
			}
		}
		return nonzero;
	}

	// absolute margin, discriminative sum loss
	static bool lossGradientADS(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		bool nonzero = false;
		for (std::size_t c=0; c<f.size(); c++)
		{
			if (c != y && f(c) > -1.0)
			{
				gradient[c] += x;
				nonzero = true;
			}
		}
		if (sumToZero && nonzero)
		{
			VectorType mean = gradient[0];
			for (std::size_t c=1; c<f.size(); c++) mean += gradient[c];
			mean /= f.size();
			for (std::size_t c=0; c<f.size(); c++) gradient[c] -= mean;
		}
		return nonzero;
	}

	// absolute margin, discriminative max loss
	static bool lossGradientADM(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		double max = -1e100;
		std::size_t argmax = 0;
		for (std::size_t c=0; c<f.size(); c++)
		{
			if (c == y) continue;
			if (f(c) > max)
			{
				max = f(c);
				argmax = c;
			}
		}
		if (max > -1.0)
		{
			gradient[argmax] += x;
			if (sumToZero)
			{
				VectorType xx = (1.0 / (f.size() - 1.0)) * x;
				for (std::size_t c=0; c<f.size(); c++) if (c != argmax) gradient[c] -= xx;
			}
			return true;
		}
		else return false;
	}

	// absolute margin, total sum loss
	static bool lossGradientATS(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		bool nonzero = false;
		for (std::size_t c=0; c<f.size(); c++)
		{
			if (c == y)
			{
				if (f(c) < 1.0)
				{
					gradient[c] -= x;
					nonzero = true;
				}
			}
			else
			{
				if (f(c) > -1.0)
				{
					gradient[c] += x;
					nonzero = true;
				}
			}
		}
		if (sumToZero && nonzero)
		{
			VectorType mean = gradient[0];
			for (std::size_t c=1; c<f.size(); c++) mean += gradient[c];
			mean /= f.size();
			for (std::size_t c=0; c<f.size(); c++) gradient[c] -= mean;
		}
		return nonzero;
	}

	// absolute margin, total max loss
	static bool lossGradientATM(
			VectorType const& x,
			unsigned int y,
			RealVector const& f,
			std::vector<VectorType>& gradient,
			bool sumToZero)
	{
		double max = -1e100;
		std::size_t argmax = 0;
		for (std::size_t c=0; c<f.size(); c++)
		{
			if (c == y)
			{
				if (-f(c) > max)
				{
					max = -f(c);
					argmax = c;
				}
			}
			else
			{
				if (f(c) > max)
				{
					max = f(c);
					argmax = c;
				}
			}
		}
		if (max > -1.0)
		{
			if (argmax == y)
			{
				gradient[argmax] -= x;
				if (sumToZero)
				{
					VectorType xx = (1.0 / (f.size() - 1.0)) * x;
					for (std::size_t c=0; c<f.size(); c++) if (c != argmax) gradient[c] += xx;
				}
			}
			else
			{
				gradient[argmax] += x;
				if (sumToZero)
				{
					VectorType xx = (1.0 / (f.size() - 1.0)) * x;
					for (std::size_t c=0; c<f.size(); c++) if (c != argmax) gradient[c] -= xx;
				}
			}
			return true;
		}
		else return false;
	}
};


}
#endif
