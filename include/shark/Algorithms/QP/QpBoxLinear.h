//===========================================================================
/*!
 * 
 *
 * \brief       Quadratic programming solver linear SVM training without bias
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        -
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


#ifndef SHARK_ALGORITHMS_QP_QPBOXLINEAR_H
#define SHARK_ALGORITHMS_QP_QPBOXLINEAR_H

#include <shark/Core/Timer.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>
#include <cmath>
#include <iostream>


namespace shark {


///
/// \brief Quadratic program solver for box-constrained problems with linear kernel
///
/// \par
/// The QpBoxLinear class is a decomposition-based solver for linear
/// support vector machine classifiers trained with the hinge loss.
/// Its optimization is largely based on the paper<br>
///   "A Dual Coordinate Descent Method for Large-scale Linear SVM"
///   by Hsieh, Chang, and Lin, ICML, 2007.
/// However, the present algorithm differs quite a bit, since it
/// learns variable preferences as a replacement for the missing
/// working set selection. At the same time, this method replaces
/// the shrinking heuristic.
///
template <class InputT>
class QpBoxLinear
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	///
	/// \brief Constructor
	///
	/// \param  dataset  training data
	/// \param  dim      problem dimension
	///
	QpBoxLinear(const DatasetType& dataset, std::size_t dim)
	: m_data(dataset)
	, m_dim(dim)
	, m_xSquared(m_data.size())
	, m_alpha(m_data.size(),0.0)
	, m_weights(m_dim,0.0)
	, m_pref(m_data.size(),1.0)
	, m_offset(0)
	
	{
		SHARK_ASSERT(dim > 0);

		// pre-compute squared norms
		for (std::size_t i=0; i<m_data.size(); i++)
		{
			auto const& x_i = m_data[i];
			m_xSquared(i) = norm_sqr(x_i.input);
		}
	}
	
	void setOffset(double newOffset){
		m_offset = newOffset;
	}
	
	double offsetGradient()const{
		double result = 0;
		for(std::size_t i = 0; i != m_data.size(); ++i){
			double y_i = (m_data[i].label > 0) ? +1.0 : -1.0;
			result += m_alpha(i) * y_i;
		}
		return result;
	}
	
	RealVector const& solutionWeightVector()const{
		return m_weights;
	}

	///
	/// \brief Solve the SVM training problem.
	///
	/// \param  bound    upper bound for m_alpha-components, complexity parameter of the hinge loss SVM
	/// \param  reg      coefficient of the penalty term \f$-\frac{reg}{2} \cdot \|\m_alpha\|^2\f$, reg=1/C where C is the complexity parameter of the squared hinge loss SVM
	/// \param  stop     stopping condition(s)
	/// \param  prop     solution properties
	/// \param  verbose  if true, the solver prints status information and solution statistics
	///
	void solve(
		double bound,
		double reg,
		QpStoppingCondition& stop,
		QpSolutionProperties* prop = NULL,
		bool verbose = false
	){
		// sanity checks
		SHARK_ASSERT(bound > 0.0);
		SHARK_ASSERT(reg >= 0.0);

		// measure training time
		Timer timer;

		// prepare dimensions and vectors
		std::size_t ell = m_data.size();
		double prefsum = sum(m_pref);               // normalization constant for m_pref
		std::vector<std::size_t> schedule(ell);

		// prepare counters
		std::size_t epoch = 0;
		std::size_t steps = 0;

		// prepare performance monitoring for self-adaptation
		double max_violation = 0.0;
		const double gain_learning_rate = 1.0 / ell;
		double average_gain = 0.0;
		bool canstop = true;

		// outer optimization loop
		while (true)
		{
			// define schedule
			double psum = prefsum;
			prefsum = 0.0;
			std::size_t pos = 0;
			for (std::size_t i=0; i<ell; i++)
			{
				double p = m_pref[i];
				double num = (psum < 1e-6) ? ell - pos : std::min((double)(ell - pos), (ell - pos) * p / psum);
				std::size_t n = (std::size_t)std::floor(num);
				double prob = num - n;
				if (random::coinToss(random::globalRng,prob)) n++;
				for (std::size_t j=0; j<n; j++)
				{
					schedule[pos] = i;
					pos++;
				}
				psum -= p;
				prefsum += p;
			}
			SHARK_ASSERT(pos == ell);
			std::shuffle(schedule.begin(),schedule.end(),random::globalRng);

			// inner loop
			max_violation = 0.0;
			for (std::size_t j=0; j<ell; j++)
			{
				// active variable
				std::size_t i = schedule[j];
				auto const& e_i = m_data[i];
				double y_i = (e_i.label > 0) ? +1.0 : -1.0;

				// compute gradient and projected gradient
				double a = m_alpha(i);
				double wyx = y_i * inner_prod(m_weights, e_i.input);
				double g = 1.0 - m_offset * y_i - wyx - reg * a;
				double pg = (a == 0.0 && g < 0.0) ? 0.0 : (a == bound && g > 0.0 ? 0.0 : g);

				// update maximal KKT violation over the epoch
				max_violation = std::max(max_violation, std::abs(pg));
				double gain = 0.0;

				// perform the step
				if (pg != 0.0)
				{
					// SMO-style coordinate descent step
					double q = m_xSquared(i) + reg;
					double mu = g / q;
					double new_a = a + mu;

					// numerically stable update
					if (new_a <= 0.0)
					{
						mu = -a;
						new_a = 0.0;
					}
					else if (new_a >= bound)
					{
						mu = bound - a;
						new_a = bound;
					}

					// update both representations of the weight vector: m_alpha and m_weights
					m_alpha(i) = new_a;
					noalias(m_weights) += (mu * y_i) * e_i.input;
					gain = mu * (g - 0.5 * q * mu);

					steps++;
				}

				// update gain-based preferences
				{
					if (epoch == 0) average_gain += gain / (double)ell;
					else
					{
						// strategy constants
						constexpr double CHANGE_RATE = 0.2;
						constexpr double PREF_MIN = 0.05;
						constexpr double PREF_MAX = 20.0;

						double change = CHANGE_RATE * (gain / average_gain - 1.0);
						double newpref = std::min(PREF_MAX, std::max(PREF_MIN, m_pref(i) * std::exp(change)));
						prefsum += newpref - m_pref(i);
						m_pref[i] = newpref;
						average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
					}
				}
			}

			epoch++;

			// stopping criteria
			if (stop.maxIterations > 0 && ell * epoch >= stop.maxIterations)
			{
				if (prop != NULL) prop->type = QpMaxIterationsReached;
				break;
			}

			if (timer.stop() >= stop.maxSeconds)
			{
				if (prop != NULL) prop->type = QpTimeout;
				break;
			}

			if (max_violation < stop.minAccuracy)
			{
				if (verbose) std::cout << "#" << std::flush;
				if (canstop)
				{
					if (prop != NULL) prop->type = QpAccuracyReached;
					break;
				}
				else
				{
					// prepare full sweep for a reliable checking of the stopping criterion
					canstop = true;
					for (std::size_t i=0; i<ell; i++) m_pref[i] = 1.0;
					prefsum = (double)ell;
				}
			}
			else
			{
				if (verbose) std::cout << "." << std::flush;
				canstop = false;
			}
		}

		timer.stop();

		// compute solution statistics
		std::size_t free_SV = 0;
		std::size_t bounded_SV = 0;
		double objective = -0.5 * norm_sqr(m_weights);
		for (std::size_t i=0; i<ell; i++)
		{
			double a = m_alpha(i);
			if (a > 0.0)
			{
				objective += a;
				objective -= reg/2.0 * a * a;
				if (a == bound) bounded_SV++;
				else free_SV++;
			}
		}

		// return solution statistics
		if (prop != NULL)
		{
			prop->accuracy = max_violation;       // this is approximate, but a good guess
			prop->iterations = ell * epoch;
			prop->value = objective;
			prop->seconds = timer.lastLap();
		}

		// output solution statistics
		if (verbose)
		{
			std::cout << std::endl;
			std::cout << "training time (seconds): " << timer.lastLap() << std::endl;
			std::cout << "number of epochs: " << epoch << std::endl;
			std::cout << "number of iterations: " << (ell * epoch) << std::endl;
			std::cout << "number of non-zero steps: " << steps << std::endl;
			std::cout << "dual accuracy: " << max_violation << std::endl;
			std::cout << "dual objective value: " << objective << std::endl;
			std::cout << "number of free support vectors: " << free_SV << std::endl;
			std::cout << "number of bounded support vectors: " << bounded_SV << std::endl;
		}
	}

protected:
	DataView<const DatasetType> m_data;               ///< view on training data
	std::size_t m_dim;                                ///< input space dimension
	RealVector m_xSquared;                            ///< diagonal entries of the quadratic matrix
	RealVector m_alpha;                               ///< storage of the m_alpha values for warm start
	RealVector m_weights;                                   ///< storage of weight vector for warm start
	RealVector m_pref;				  ///< measure of success of individual steps
	double m_offset;
};


//~ template < >
//~ class QpBoxLinear<CompressedRealVector>
//~ {
//~ public:
	//~ typedef LabeledData<CompressedRealVector, unsigned int> DatasetType;

	//~ ///
	//~ /// \brief Constructor
	//~ ///
	//~ /// \param  dataset  training data
	//~ /// \param  dim      problem dimension
	//~ ///
	//~ QpBoxLinear(const DatasetType& dataset, std::size_t dim)
	//~ : x(dataset.numberOfElements())
	//~ , y(dataset.numberOfElements())
	//~ , diagonal(dataset.numberOfElements())
	//~ , m_dim(dim)
	//~ {
		//~ SHARK_ASSERT(dim > 0);

		//~ // transform ublas sparse vectors into a fast format
		//~ // (yes, ublas is slow...), and compute the diagonal
		//~ // elements of the quadratic matrix
		//~ SparseVector sparse;
		//~ for (std::size_t b=0, j=0; b<dataset.numberOfBatches(); b++)
		//~ {
			//~ DatasetType::const_batch_reference batch = dataset.batch(b);
			//~ for (std::size_t i=0; i<batch.size(); i++)
			//~ {
				//~ auto const& x_i = shark::get(batch, i).input;
				//~ // if (x_i.nnz() == 0) continue;

				//~ unsigned int y_i = shark::get(batch, i).label;
				//~ y[j] = 2.0 * y_i - 1.0;
				//~ double d = 0.0;
				//~ for (auto it=x_i.begin(); it != x_i.end(); ++it)
				//~ {
					//~ double v = *it;
					//~ sparse.index = it.index();
					//~ sparse.value = v;
					//~ storage.push_back(sparse);
					//~ d += v * v;
				//~ }
				//~ sparse.index = (std::size_t)-1;
				//~ storage.push_back(sparse);
				//~ diagonal(j) = d;
				//~ j++;
			//~ }
		//~ }
		//~ for (std::size_t b=0, j=0, k=0; b<dataset.numberOfBatches(); b++)
		//~ {
			//~ DatasetType::const_batch_reference batch = dataset.batch(b);
			//~ for (std::size_t i=0; i<batch.size(); i++)
			//~ {
				//~ auto const& x_i = shark::get(batch, i).input;
				//~ // if (x_i.nnz() == 0) continue;

				//~ x[j] = &storage[k];   // cannot be done in the first loop because of vector reallocation
				//~ for (; storage[k].index != (std::size_t)-1; k++);
				//~ k++;
				//~ j++;
			//~ }
		//~ }
	//~ }

	//~ ///
	//~ /// \brief Solve the SVM training problem.
	//~ ///
	//~ /// \param  bound    upper bound for m_alpha-components, complexity parameter of the hinge loss SVM
	//~ /// \param  reg      coefficient of the penalty term \f$-\frac{reg}{2} \cdot \|\m_alpha\|^2\f$, reg=1/C where C is the complexity parameter of the squared hinge loss SVM
	//~ /// \param  stop     stopping condition(s)
	//~ /// \param  prop     solution properties
	//~ /// \param  verbose  if true, the solver prints status information and solution statistics
	//~ ///
	//~ RealVector solve(
			//~ double bound,
			//~ double reg,
			//~ QpStoppingCondition& stop,
			//~ QpSolutionProperties* prop = NULL,
			//~ bool verbose = false)
	//~ {
		//~ // sanity checks
		//~ SHARK_ASSERT(bound > 0.0);
		//~ SHARK_ASSERT(reg >= 0.0);

		//~ // measure training time
		//~ Timer timer;
		//~ timer.start();

		//~ // prepare dimensions and vectors
		//~ std::size_t ell = x.size();
		//~ RealVector m_alpha(ell, 0.0);
		//~ RealVector m_weights(m_dim, 0.0);
		//~ RealVector m_pref(ell, 1.0);          // measure of success of individual steps
		//~ double prefsum = ell;               // normalization constant
		//~ std::vector<std::size_t> schedule(ell);

		//~ // prepare counters
		//~ std::size_t epoch = 0;
		//~ std::size_t steps = 0;

		//~ // prepare performance monitoring for self-adaptation
		//~ double max_violation = 0.0;
		//~ const double gain_learning_rate = 1.0 / ell;
		//~ double average_gain = 0.0;
		//~ bool canstop = true;

		//~ // outer optimization loop
		//~ while (true)
		//~ {
			//~ // define schedule
			//~ double psum = prefsum;
			//~ prefsum = 0.0;
			//~ std::size_t pos = 0;
			//~ for (std::size_t i=0; i<ell; i++)
			//~ {
				//~ double p = m_pref[i];
				//~ double num = (psum < 1e-6) ? ell - pos : std::min((double)(ell - pos), (ell - pos) * p / psum);
				//~ std::size_t n = (std::size_t)std::floor(num);
				//~ double prob = num - n;
				//~ if (random::uni() < prob) n++;
				//~ for (std::size_t j=0; j<n; j++)
				//~ {
					//~ schedule[pos] = i;
					//~ pos++;
				//~ }
				//~ psum -= p;
				//~ prefsum += p;
			//~ }
			//~ SHARK_ASSERT(pos == ell);
			//~ for (std::size_t i=0; i<ell; i++) std::swap(schedule[i], schedule[random::discrete(0, ell - 1)]);

			//~ // inner loop
			//~ max_violation = 0.0;
			//~ for (std::size_t j=0; j<ell; j++)
			//~ {
				//~ // active variable
				//~ std::size_t i = schedule[j];
				//~ const SparseVector* x_i = x[i];

				//~ // compute gradient and projected gradient
				//~ double a = m_alpha(i);
				//~ double wyx = y(i) * inner_prod(m_weights, x_i);
				//~ double g = 1.0 - wyx - reg * a;
				//~ double pg = (a == 0.0 && g < 0.0) ? 0.0 : (a == bound && g > 0.0 ? 0.0 : g);

				//~ // update maximal KKT violation over the epoch
				//~ max_violation = std::max(max_violation, std::abs(pg));
				//~ double gain = 0.0;

				//~ // perform the step
				//~ if (pg != 0.0)
				//~ {
					//~ // SMO-style coordinate descent step
					//~ double q = diagonal(i) + reg;
					//~ double mu = g / q;
					//~ double new_a = a + mu;

					//~ // numerically stable update
					//~ if (new_a <= 0.0)
					//~ {
						//~ mu = -a;
						//~ new_a = 0.0;
					//~ }
					//~ else if (new_a >= bound)
					//~ {
						//~ mu = bound - a;
						//~ new_a = bound;
					//~ }

					//~ // update both representations of the weight vector: m_alpha and m_weights
					//~ m_alpha(i) = new_a;
					//~ // m_weights += (mu * y(i)) * x_i;
					//~ axpy(m_weights, mu * y(i), x_i);
					//~ gain = mu * (g - 0.5 * q * mu);

					//~ steps++;
				//~ }

				//~ // update gain-based preferences
				//~ {
					//~ if (epoch == 0) average_gain += gain / (double)ell;
					//~ else
					//~ {
						//~ // strategy constants
						//~ constexpr double CHANGE_RATE = 0.2;
						//~ constexpr double PREF_MIN = 0.05;
						//~ constexpr double PREF_MAX = 20.0;

						//~ double change = CHANGE_RATE * (gain / average_gain - 1.0);
						//~ double newpref = std::min(PREF_MAX, std::max(PREF_MIN, m_pref(i) * std::exp(change)));
						//~ prefsum += newpref - m_pref(i);
						//~ m_pref[i] = newpref;
						//~ average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
					//~ }
				//~ }
			//~ }

			//~ epoch++;

			//~ // stopping criteria
			//~ if (stop.maxIterations > 0 && ell * epoch >= stop.maxIterations)
			//~ {
				//~ if (prop != NULL) prop->type = QpMaxIterationsReached;
				//~ break;
			//~ }

			//~ if (timer.stop() >= stop.maxSeconds)
			//~ {
				//~ if (prop != NULL) prop->type = QpTimeout;
				//~ break;
			//~ }

			//~ if (max_violation < stop.minAccuracy)
			//~ {
				//~ if (verbose) std::cout << "#" << std::flush;
				//~ if (canstop)
				//~ {
					//~ if (prop != NULL) prop->type = QpAccuracyReached;
					//~ break;
				//~ }
				//~ else
				//~ {
					//~ // prepare full sweep for a reliable checking of the stopping criterion
					//~ canstop = true;
					//~ for (std::size_t i=0; i<ell; i++) m_pref[i] = 1.0;
					//~ prefsum = ell;
				//~ }
			//~ }
			//~ else
			//~ {
				//~ if (verbose) std::cout << "." << std::flush;
				//~ canstop = false;
			//~ }
		//~ }

		//~ timer.stop();

		//~ // compute solution statistics
		//~ std::size_t free_SV = 0;
		//~ std::size_t bounded_SV = 0;
		//~ double objective = -0.5 * shark::blas::inner_prod(m_weights, m_weights);
		//~ for (std::size_t i=0; i<ell; i++)
		//~ {
			//~ double a = m_alpha(i);
			//~ if (a > 0.0)
			//~ {
				//~ objective += a;
				//~ objective -= reg/2.0 * a * a;
				//~ if (a == bound) bounded_SV++;
				//~ else free_SV++;
			//~ }
		//~ }

		//~ // return solution statistics
		//~ if (prop != NULL)
		//~ {
			//~ prop->accuracy = max_violation;       // this is approximate, but a good guess
			//~ prop->iterations = ell * epoch;
			//~ prop->value = objective;
			//~ prop->seconds = timer.lastLap();
		//~ }

		//~ // output solution statistics
		//~ if (verbose)
		//~ {
			//~ std::cout << std::endl;
			//~ std::cout << "training time (seconds): " << timer.lastLap() << std::endl;
			//~ std::cout << "number of epochs: " << epoch << std::endl;
			//~ std::cout << "number of iterations: " << (ell * epoch) << std::endl;
			//~ std::cout << "number of non-zero steps: " << steps << std::endl;
			//~ std::cout << "dual accuracy: " << max_violation << std::endl;
			//~ std::cout << "dual objective value: " << objective << std::endl;
			//~ std::cout << "number of free support vectors: " << free_SV << std::endl;
			//~ std::cout << "number of bounded support vectors: " << bounded_SV << std::endl;
		//~ }

		//~ // return the solution
		//~ return m_weights;
	//~ }

//~ protected:
	//~ /// \brief Data structure for sparse vectors.
	//~ struct SparseVector
	//~ {
		//~ std::size_t index;
		//~ double value;
	//~ };

	//~ /// \brief Famous "axpy" product, here adding a multiple of a sparse vector to a dense one.
	//~ static inline void axpy(RealVector& m_weights, double m_alpha, const SparseVector* xi)
	//~ {
		//~ while (true)
		//~ {
			//~ if (xi->index == (std::size_t)-1) return;
			//~ m_weights[xi->index] += m_alpha * xi->value;
			//~ xi++;
		//~ }
	//~ }

	//~ /// \brief Inner product between a dense and a sparse vector.
	//~ static inline double inner_prod(RealVector const& m_weights, const SparseVector* xi)
	//~ {
		//~ double ret = 0.0;
		//~ while (true)
		//~ {
			//~ if (xi->index == (std::size_t)-1) return ret;
			//~ ret += m_weights[xi->index] * xi->value;
			//~ xi++;
		//~ }
	//~ }

	//~ std::vector<SparseVector> storage;                ///< storage for sparse vectors
	//~ std::vector<SparseVector*> x;                     ///< sparse vectors
	//~ RealVector y;                                     ///< +1/-1 labels
	//~ RealVector diagonal;                              ///< diagonal entries of the quadratic matrix
	//~ std::size_t m_dim;                                ///< input space dimension
//~ };


}
#endif
