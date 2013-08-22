//===========================================================================
/*!
 *  \brief LASSO Regression
 *
 *  \author T. Glasmachers
 *  \date 2013
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


#ifndef SHARK_ALGORITHMS_TRAINERS_LASSOREGRESSION_H
#define SHARK_ALGORITHMS_TRAINERS_LASSOREGRESSION_H

#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <cmath>


namespace shark {


/*!
 *  \brief LASSO Regression
 *
 *  LASSO Regression extracts a sparse vector of regression
 *  coefficients. The original method amounts to L1-constrained
 *  least squares regression, while this implementation uses an
 *  L1 penalty instead of a constraint (which is equivalent).
 *
 *  For data vectors \f$ x_i \f$ with real-valued labels \f$ y_i \f$
 *  the trainer solves the problem
 *  \f$ \min_w \quad \frac{1}{2} \sum_i (w^T x_i - y_i)^2 + \lambda \|w\|_1 \f$.
 *  The target accuracy of the solution is measured in terms of the
 *  smallest component (L1 norm) of the gradient of the objective
 *  function.
 *
 *  The trainer has one template parameter, namely the type of
 *  the input vectors \f$ x_i \f$. These need to be vector valued,
 *  typically either RealVector of CompressedRealVector. The
 *  resulting weight vector w is represented by a LinearModel
 *  object. Currently model outputs and labels are restricted to a
 *  single dimension.
 */
template <class InputVectorType = RealVector>
class LassoRegression : public AbstractTrainer<LinearModel<InputVectorType, RealVector> >, public IParameterizable
{
public:
	typedef LinearModel<InputVectorType, RealVector> ModelType;
	typedef LabeledData<InputVectorType, RealVector> DataType;

	/// \brief Constructor.
	///
	/// \param  _lambda    value of the regularization parameter (see class description)
	/// \param  _accuracy  stopping criterion for the iterative solver, maximal gradient component of the objective function (see class description)
	LassoRegression(double _lambda, double _accuracy = 0.01)
	: m_lambda(_lambda)
	, m_accuracy(_accuracy)
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
	void setAccuracy(double _accuracy)
	{
		RANGE_CHECK(_accuracy > 0.0);
		m_accuracy = _accuracy;
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
	void train(ModelType& model, DataType const& dataset)
	{
		SIZE_CHECK(model.outputSize() == 1);

		dim = inputDimension(dataset);
		RealVector alpha(dim, 0.0);
		trainInternal(alpha, dataset);

		std::size_t nnz = 0;
		for (std::size_t i=0; i<alpha.size(); i++) if (alpha(i) != 0.0) nnz++;

		if (2 * nnz < alpha.size())
		{
			// use sparse model
			CompressedRealMatrix mat(1, dim);
			row(mat, 0) = alpha;
			model.setStructure(mat);
		}
		else
		{
			// use dense model
			RealMatrix mat(1, dim);
			row(mat, 0) = alpha;
			model.setStructure(mat);
		}
	}

protected:
	/// \brief Create internal data representation for fast processing.
	void fillData(DataType const& dataset)
	{
		// pass 1: find number of points and number of entries
		ell = 0;
		size_t num_entries = 0;
		for (std::size_t i=0; i<dataset.numberOfBatches(); i++)
		{
			typename Batch<InputVectorType>::type const& b = dataset.inputs().batch(i);
			std::size_t bsz = boost::size(b);
			ell += bsz;
			for (std::size_t j=0; j<bsz; j++)
			{
				typename Batch<InputVectorType>::const_reference e = get(b, j);
				if (traits::IsSparse<typename Batch<InputVectorType>::const_reference>::value)
				{
					// num_entries += e.nnz() + 1;
					typename Batch<InputVectorType>::const_reference::const_iterator begin = e.begin();
					typename Batch<InputVectorType>::const_reference::const_iterator end = e.end();
					num_entries += std::distance(begin, end);
				}
				else
				{
					for (std::size_t q=0; q<dim; q++) if (e(q) != 0.0) num_entries++;
				}
				num_entries++;
			}
		}

		// prepare memory
		label.resize(ell);
		data.resize(dim);
		storage.resize(num_entries);
		UIntVector feature(dim, 0u);

		// pass 2: count entries per feature
		for (std::size_t i=0; i<dataset.numberOfBatches(); i++)
		{
			typename Batch<InputVectorType>::type const& b = dataset.inputs().batch(i);
			std::size_t bsz = boost::size(b);
			for (std::size_t j=0; j<bsz; j++)
			{
				typename Batch<InputVectorType>::const_reference e = get(b, j);
				if (traits::IsSparse<typename Batch<InputVectorType>::const_reference>::value)
				{
					for (typename Batch<InputVectorType>::const_reference::const_iterator it=e.begin(); it != e.end(); ++it) feature(it.index())++;
				}
				else
				{
					for (std::size_t q=0; q<dim; q++) if (e(q) != 0.0) feature(q)++;
				}
			}
		}

		// prepare storage and start indices
		size_t spos = 0;
		for (size_t i=0; i<dim; i++)
		{
			data[i] = &storage[spos];
			spos += feature[i];
			storage[spos].index = ((std::size_t)-1);
			spos++;
			feature[i] = 0;
		}

		// pass 3: copy entries and labels
		for (std::size_t i=0, m=0; i<dataset.numberOfBatches(); i++)
		{
			typename Batch<InputVectorType>::type const& b = dataset.inputs().batch(i);
			std::size_t bsz = boost::size(b);
			for (std::size_t j=0; j<bsz; j++, m++)
			{
				typename Batch<InputVectorType>::const_reference e = get(b, j);
				if (traits::IsSparse<typename Batch<InputVectorType>::const_reference>::value)
				{
					for (typename Batch<InputVectorType>::const_reference::const_iterator it=e.begin(); it != e.end(); ++it)
					{
						std::size_t index = it.index();
						double value = *it;

						data[index][feature[index]].index = m;
						data[index][feature[index]].value = value;
						feature[index]++;
					}
				}
				else
				{
					for (std::size_t index=0; index<dim; index++)
					{
						double value = e(index);
						if (value == 0.0) continue;

						data[index][feature[index]].index = m;
						data[index][feature[index]].value = value;
						feature[index]++;
					}
				}
			}
		}
		for (std::size_t i=0, m=0; i<dataset.numberOfBatches(); i++)
		{
			Batch<RealVector>::type const& b = dataset.labels().batch(i);
			std::size_t bsz = boost::size(b);
			for (std::size_t j=0; j<bsz; j++, m++)
			{
				label(m) = get(b, j)(0);
			}
		}
	}

	/// \brief Actual training procedure.
	void trainInternal(RealVector& alpha, DataType const& dataset)
	{
		// strategy constants
		const double CHANGE_RATE = 0.2;
		const double PREF_MIN = 0.05;
		const double PREF_MAX = 20.0;

		// console output
		const bool verbose = false;

		fillData(dataset);

		RealVector diag(dim);
		RealVector w = label;
		UIntVector index(dim);
		RealVector pref(dim);

		// pre-calculate diagonal matrix entries (feature-wise squared norms)
		for (size_t i=0; i<dim; i++)
		{
			double sum = 0.0;
			for (Entry* e = data[i]; e->index != ((std::size_t)-1); e++) sum += e->value * e->value;
			diag[i] = sum;
		}

		// prepare preferences for scheduling
		for (size_t i=0; i<dim; i++) pref[i] = 1.0;
		double prefsum = (double)dim;

		// prepare performance monitoring for self-adaptation
		const double gain_learning_rate = 1.0 / dim;
		double average_gain = 0.0;
		int canstop = 1;
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
				if (psum < 1e-6) n = dim - pos;      // for numerical stability
				else if (p < psum) n = (dim - pos) * p / psum;
				else n = (dim - pos);                // for numerical stability
				int m = (int)floor(n);
				double prob = n - m;
				if ((double)rand() / (double)RAND_MAX < prob) m++;
				for (std::size_t  j=0; j<m; j++)
				{
					index[pos] = i;
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
				double a = alpha[i];
				double d = diag[i];

				// compute "gradient component" <w, X_i>
				double grad = 0.0;
				for (Entry* e = data[i]; e->index != ((std::size_t)-1); e++) grad += w[e->index] * e->value;

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
					alpha[i] += delta;
					for (Entry* e = data[i]; e->index != ((std::size_t)-1); e++) w[e->index] += delta * e->value;
				}

				// update gain-based preferences
				{
					if (iter == 0) average_gain += gain / (double)dim;
					else
					{
						double change = CHANGE_RATE * (gain / average_gain - 1.0);
						double newpref = pref[i] * std::exp(change);
						if (newpref < PREF_MIN) newpref = PREF_MIN;
						else if (newpref > PREF_MAX) newpref = PREF_MAX;
						prefsum += newpref - pref[i];
						pref[i] = newpref;
						average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
					}
				}
			}
			iter++;

			if (maxvio <= m_accuracy)
			{
				if (canstop) break;
				else
				{
					// prepare full sweep for a reliable check of the stopping criterion
					canstop = 1;
					for (size_t i=0; i<dim; i++) pref[i] = 1.0;
					prefsum = (double)dim;
					if (verbose) std::cout << "*" << std::flush;
				}
			}
			else
			{
				canstop = 0;
				if (verbose) std::cout << "." << std::flush;
			}
		}
/*
		// compute objective value and count non-zero features
		size_t nnz = 0;
		double obj = 0.0;
		for (size_t i=0; i<dim; i++)
		{
			if (alpha[i] != 0.0)
			{
				nnz++;
				obj += fabs(alpha[i]);
			}
		}
		obj *= lambda;
		for (size_t i=0; i<ell; i++) obj += w[i] * w[i];

		// output statistics
		printf("  optimization time:   %.3g seconds\n", seconds);
		printf("  update steps:        %lu\n", steps);
		printf("  objective value:     %f\n", obj);
		printf("  non-zero features:   %lu\n",nnz);
*/
	}

	/// \brief Sparse vector entry.
	struct Entry
	{
		std::size_t index;
		double value;
	};

	double m_lambda;             ///< regularization parameter
	double m_accuracy;           ///< gradient accuracy
	std::size_t dim;             ///< dimension; number of features
	std::size_t ell;             ///< number of points
	RealVector label;            ///< dense label vector, one entry per point
	std::vector<Entry*> data;    ///< array of sparse vectors, one per feature
	std::vector<Entry> storage;  ///< linear memory
};


}
#endif
