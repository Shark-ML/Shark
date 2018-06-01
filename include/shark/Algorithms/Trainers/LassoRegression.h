//===========================================================================
/*!
 *
 *
 * \brief       LASSO Regression
 *
 *
 *
 * \author      T. Glasmachers
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_TRAINERS_LASSOREGRESSION_H
#define SHARK_ALGORITHMS_TRAINERS_LASSOREGRESSION_H

#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <cmath>


namespace shark {



///  \brief LASSO Regression
///
///  LASSO Regression extracts a sparse vector of regression
///  coefficients. The original method amounts to L1-constrained
///  least squares regression, while this implementation uses an
///  L1 penalty instead of a constraint (which is equivalent).
///
///  For data vectors \f$ x_i \f$ with real-valued labels \f$ y_i \f$
///  the trainer solves the problem
///  \f$ \min_w \quad \frac{1}{2} \sum_i (w^T x_i - y_i)^2 + \lambda \|w\|_1 \f$.
///  The target accuracy of the solution is measured in terms of the
///  smallest component of the gradient of the objective function.
///
///  The trainer has one template parameter, namely the type of
///  the input vectors \f$ x_i \f$. These need to be vector valued,
///  typically either RealVector of CompressedRealVector. The
///  resulting weight vector w is represented by a LinearModel
///  object. Currently model outputs and labels are restricted to a
///  single dimension.
/// \ingroup supervised_trainer
template <class InputVectorType = RealVector>
class LassoRegression : public AbstractTrainer<LinearModel<InputVectorType> >, public IParameterizable<>
{
public:
	typedef LinearModel<InputVectorType> ModelType;
	typedef LabeledData<InputVectorType, RealVector> DataType;

	/// \brief Constructor.
	///
	/// \param  lambda    value of the regularization parameter (see class description)
	/// \param  accuracy  stopping criterion for the iterative solver, maximal gradient component of the objective function (see class description)
	LassoRegression(double lambda, double accuracy = 0.01)
	: m_lambda(lambda)
	, m_accuracy(accuracy)
	{
		RANGE_CHECK(m_lambda >= 0.0);
		RANGE_CHECK(m_accuracy > 0.0);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LASSO regression"; }


	/// \brief Return the current setting of the regularization parameter.
	double lambda() const
	{
		return m_lambda;
	}

	/// \brief Set the regularization parameter.
	void setLambda(double lambda)
	{
		RANGE_CHECK(lambda >= 0.0);
		m_lambda = lambda;
	}

	/// \brief Return the current setting of the accuracy (maximal gradient component of the optimization problem).
	double accuracy() const
	{
		return m_accuracy;
	}

	/// \brief Set the accuracy (maximal gradient component of the optimization problem).
	void setAccuracy(double accuracy)
	{
		RANGE_CHECK(accuracy > 0.0);
		m_accuracy = accuracy;
	}

	/// \brief Get the regularization parameter lambda through the IParameterizable interface.
	RealVector parameterVector() const
	{
		return RealVector(1, m_lambda);
	}

	/// \brief Set the regularization parameter lambda through the IParameterizable interface.
	void setParameterVector(const RealVector& param)
	{
		SIZE_CHECK(param.size() == 1);
		RANGE_CHECK(param(0) >= 0.0);
		m_lambda = param(0);
	}

	/// \brief Return the number of parameters (one in this case).
	size_t numberOfParameters() const
	{
		return 1;
	}

	/// \brief Train a linear model with LASSO regression.
	void train(ModelType& model, DataType const& dataset){

		// strategy constants
		const double CHANGE_RATE = 0.2;
		const double PREF_MIN = 0.05;
		const double PREF_MAX = 20.0;

		// console output
		const bool verbose = false;

		std::size_t dim = inputDimension(dataset);
		RealVector w(dim, 0.0);

		// transpose the dataset and push it inside a single matrix
		auto data = createBatch(dataset.inputs().elements());
		data = trans(data);
		auto label_mat = createBatch(dataset.labels().elements());
		RealVector label = column(label_mat,0);

		RealVector diag(dim);
		RealVector difference = -label;
		UIntVector index(dim);

		// pre-calculate diagonal matrix entries (feature-wise squared norms)
		for (size_t i=0; i<dim; i++){
			diag[i] = norm_sqr(row(data,i));
		}

		// prepare preferences for scheduling
		RealVector pref(dim, 1.0);
		double prefsum = (double)dim;

		// prepare performance monitoring for self-adaptation
		const double gain_learning_rate = 1.0 / dim;
		double average_gain = 0.0;
		bool canstop = true;
		const double lambda = m_lambda;

		// main optimization loop
		std::size_t iter = 0;
		std::size_t steps = 0;
		while (true)
		{
			double maxvio = 0.0;

			// define schedule
			double psum = prefsum;
			prefsum = 0.0;
			int pos = 0;

			for (std::size_t i=0; i<dim; i++)
			{
				double p = pref[i];
				double n;
				if (psum >= 1e-6 && p < psum)
					n = (dim - pos) * p / psum;
				else
					n = (double)(dim - pos);          // for numerical stability

				unsigned int m = (unsigned int)floor(n);
				double prob = n - m;
				if ((double)rand() / (double)RAND_MAX < prob) m++;
				for (std::size_t  j=0; j<m; j++)
				{
					index[pos] = (unsigned int)i;
					pos++;
				}
				psum -= p;
				prefsum += p;
			}
			for (std::size_t i=0; i<dim; i++)
			{
				std::size_t r = rand() % dim;
				std::swap(index[r], index[i]);
			}

			steps += dim;
			for (size_t s=0; s<dim; s++)
			{
				std::size_t i = index[s];
				double a = w[i];
				double d = diag[i];

				// compute gradient
				double grad = inner_prod(difference, row(data,i));

				// compute optimal coordinate descent step and corresponding gain
				double vio = 0.0;
				double gain = 0.0;
				double delta = 0.0;
				if (a == 0.0)
				{
					if (grad > lambda)
					{
						vio = grad - lambda;
						delta = -vio / d;
						gain = 0.5 * d * delta * delta;
					}
					else if (grad < -lambda)
					{
						vio = -grad - lambda;
						delta = vio / d;
						gain = 0.5 * d * delta * delta;
					}
				}
				else if (a > 0.0)
				{
					grad += lambda;
					vio = std::fabs(grad);
					delta = -grad / d;
					if (delta < -a)
					{
						delta = -a;
						gain = delta * (grad - 0.5 * d * delta);
						double g0 = grad - a * d - 2.0 * lambda;
						if (g0 > 0.0)
						{
							double dd = -g0 / d;
							gain = dd * (grad - 0.5 * d * dd);
							delta += dd;
						}
					}
					else gain = 0.5 * d * delta * delta;
				}
				else
				{
					grad -= lambda;
					vio = std::fabs(grad);
					delta = -grad / d;
					if (delta > -a)
					{
						delta = -a;
						gain = delta * (grad - 0.5 * d * delta);
						double g0 = grad - a * d + 2.0 * lambda;
						if (g0 < 0.0)
						{
							double dd = -g0 / d;
							gain = dd * (grad - 0.5 * d * dd);
							delta += dd;
						}
					}
					else gain = 0.5 * d * delta * delta;
				}

				// update state
				if (vio > maxvio) maxvio = vio;
				if (delta != 0.0)
				{
					w[i] += delta;
					noalias(difference) += delta*row(data,i);
				}

				// update gain-based preferences
				{
					if (iter == 0)
						average_gain += gain / (double)dim;
					else
					{
						double change = CHANGE_RATE * (gain / average_gain - 1.0);
						double newpref = pref[i] * std::exp(change);
						newpref = std::min(std::max(newpref, PREF_MIN), PREF_MAX);
						prefsum += newpref - pref[i];
						pref[i] = newpref;
						average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
					}
				}
			}
			iter++;

			if (maxvio <= m_accuracy)
			{
				if (canstop)
					break;
				else
				{
					// prepare full sweep for a reliable check of the stopping criterion
					canstop = true;
					noalias(pref) = blas::repeat(1.0, dim);
					prefsum = (double)dim;
					if (verbose) std::cout << "*" << std::flush;
				}
			}
			else
			{
				canstop = false;
				if (verbose) std::cout << "." << std::flush;
			}
		}

		// write the weight vector into the model
		RealMatrix mat(1, w.size());
		row(mat, 0) = w;
		model.setStructure(mat);
	}

protected:
	double m_lambda;             ///< regularization parameter
	double m_accuracy;           ///< gradient accuracy
};


}
#endif
