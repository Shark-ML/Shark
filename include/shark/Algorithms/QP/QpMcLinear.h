//===========================================================================
/*!
 *  \brief Quadratic programming solvers for linear multi-class SVM training without bias.
 *
 *  \author  T. Glasmachers
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


#ifndef SHARK_ALGORITHMS_QP_QPMCLINEAR_H
#define SHARK_ALGORITHMS_QP_QPMCLINEAR_H

#include <shark/Core/Timer.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>
#include <cmath>
#include <iostream>
#include <vector>


namespace shark {


#define ACF
//#define SHRINKING

// strategy constants
#define CHANGE_RATE 0.2
#define PREF_MIN 0.05
#define PREF_MAX 20.0

// inner iteration limit
#define MAXITER_MULTIPLIER 10


/// \brief Generic solver skeleton for linear multi-class SVM problems.
template <class InputT>
class QpMcLinear
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;
	typedef typename LabeledData<InputT, unsigned int>::const_element_reference ElementType;
	typedef typename Batch<InputT>::const_reference InputReferenceType;

	///
	/// \brief Constructor
	///
	/// \param  dataset  training data
	/// \param  dim      problem dimension
	/// \param  classes  number of classes in the problem
	///
	QpMcLinear(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: m_data(dataset)
	, m_xSquared(dataset.numberOfElements())
	, m_dim(dim)
	, m_classes(classes)
	{
		SHARK_ASSERT(m_dim > 0);

		for (std::size_t i=0; i<m_data.size(); i++)
		{
			m_xSquared(i) = inner_prod(m_data[i].input, m_data[i].input);
		}
	}

	///
	/// \brief Solve the SVM training problem.
	///
	/// \param  C        regularization constant of the SVM
	/// \param  stop     stopping condition(s)
	/// \param  prop     solution properties
	/// \param  verbose  if true, the solver prints status information and solution statistics
	///
	RealMatrix solve(
			double C,
			QpStoppingCondition& stop,
			QpSolutionProperties* prop = NULL,
			bool verbose = false)
	{
		// sanity checks
		SHARK_ASSERT(C > 0.0);

		// measure training time
		Timer timer;
		timer.start();

		// prepare dimensions and vectors
		std::size_t ell = m_data.size();             // number of training examples
		RealMatrix alpha(ell, m_classes + 1, 0.0);   // Lagrange multipliers; dual variables. Reserve one extra column.
		RealMatrix w(m_classes, m_dim, 0.0);         // weight vectors; primal variables

#ifdef ACF
		// scheduling of steps
		RealVector pref(ell, 1.0);                   // example-wise measure of success
		double prefsum = ell;                        // normalization constant
#endif
		std::vector<std::size_t> schedule(ell);
#ifndef ACF
		for (std::size_t i=0; i<ell; i++) schedule[i] = i;
#endif
#ifdef SHRINKING
		std::size_t active = ell;
#endif

		// prepare counters
		std::size_t epoch = 0;
		std::size_t steps = 0;

		// prepare performance monitoring
		double objective = 0.0;
		double max_violation = 0.0;
#ifdef ACF
		const double gain_learning_rate = 1.0 / ell;
		double average_gain = 0.0;
#endif

		// outer optimization loop (epochs)
		bool canstop = true;
		while (true)
		{
#ifdef ACF
			// define schedule
			double psum = prefsum;
			prefsum = 0.0;
			std::size_t pos = 0;
			for (std::size_t i=0; i<ell; i++)
			{
				double p = pref(i);
				double num = (psum < 1e-6) ? ell - pos : std::min((double)(ell - pos), (ell - pos) * p / psum);
				std::size_t n = (std::size_t)std::floor(num);
				double prob = num - n;
				if (Rng::uni() < prob) n++;
				for (std::size_t j=0; j<n; j++)
				{
					schedule[pos] = i;
					pos++;
				}
				psum -= p;
				prefsum += p;
			}
			SHARK_ASSERT(pos == ell);
#endif
#ifdef SHRINKING
			for (std::size_t i=0; i<active; i++) std::swap(schedule[i], schedule[Rng::discrete(0, active - 1)]);
#else
			for (std::size_t i=0; i<ell; i++) std::swap(schedule[i], schedule[Rng::discrete(0, ell - 1)]);
#endif

			// inner loop (one epoch)
			max_violation = 0.0;
#ifdef SHRINKING
			for (std::size_t j=0; j<active; j++)
#else
			for (std::size_t j=0; j<ell; j++)
#endif
			{
				// active example
				double gain = 0.0;
				const std::size_t i = schedule[j];
				InputReferenceType x_i = m_data[i].input;
				const unsigned int y_i = m_data[i].label;
				const double q = m_xSquared(i);
				RealMatrixRow a = row(alpha, i);

				// compute gradient and KKT violation
				RealVector wx(m_classes,0.0);
				axpy_prod(w,x_i,wx,false);
				RealVector g(m_classes);
				double kkt = calcGradient(g, wx, a, C, y_i);

				if (kkt > 0.0)
				{
					max_violation = std::max(max_violation, kkt);

					// perform the step on alpha
					RealVector mu(m_classes, 0.0);
					gain = solveSub(0.1 * stop.minAccuracy, g, q, C, y_i, a, mu);
					objective += gain;
					steps++;

					// update weight vectors
					updateWeightVectors(w, mu, i);
				}
#ifdef SHRINKING
				else
				{
					active--;
					std::swap(schedule[j], schedule[active]);
					j--;
				}
#endif

#ifdef ACF
				// update gain-based preferences
				{
					if (epoch == 0) average_gain += gain / (double)ell;
					else
					{
						double change = CHANGE_RATE * (gain / average_gain - 1.0);
						double newpref = std::min(PREF_MAX, std::max(PREF_MIN, pref(i) * std::exp(change)));
						prefsum += newpref - pref(i);
						pref(i) = newpref;
						average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
					}
				}
#endif
			}

			epoch++;

			// stopping criteria
			if (stop.maxIterations > 0 && epoch * ell >= stop.maxIterations)
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
#ifdef ACF
					// prepare full sweep for a reliable checking of the stopping criterion
					canstop = true;
					for (std::size_t i=0; i<ell; i++) pref(i) = 1.0;
					prefsum = ell;
#endif
#ifdef SHRINKING
					// prepare full sweep for a reliable checking of the stopping criterion
					active = ell;
					canstop = true;
#endif
				}
			}
			else
			{
				if (verbose) std::cout << "." << std::flush;
#ifdef ACF
				canstop = false;
#endif
#ifdef SHRINKING
				canstop = (active == ell);
#endif
			}
		}
		timer.stop();

		// calculate dual objective value
		objective = 0.0;
		for (std::size_t j=0; j<m_classes; j++)
		{
			for (std::size_t d=0; d<m_dim; d++) objective -= w(j, d) * w(j, d);
		}
		objective *= 0.5;
		for (std::size_t i=0; i<ell; i++)
		{
			for (std::size_t j=0; j<m_classes; j++) objective += alpha(i, j);
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
		}

		// return the solution
		return w;
	}

protected:
	// for all c: row(w, c) += mu(c) * x
	void add_scaled(RealMatrix& w, RealVector const& mu, InputReferenceType x)
	{
		for (std::size_t c=0; c<m_classes; c++) noalias(row(w, c)) += mu(c) * x;
	}

	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	///
	/// \param  gradient  gradient vector to be filled in. The vector is correctly sized.
	/// \param  wx        inner products of weight vectors with the current sample; wx(c) = <w_c, x>
	/// \param  alpha     variables corresponding to the current sample
	/// \param  C         upper bound on the variables
	/// \param  y         label of the current sample
	///
	/// \return  The function must return the violation of the KKT conditions.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y) = 0;

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	///
	/// \param  w   matrix of (dense) weight vectors (as rows)
	/// \param  mu  dual step on the variables corresponding to the current sample
	/// \param  index   current sample
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index) = 0;

	/// \brief Solve the sub-problem posed by a single training example.
	///
	/// \param  epsilon   accuracy (dual gradient) up to which the sub-problem should be solved
	/// \param  gradient  gradient of the objective function w.r.t. alpha
	/// \param  q         squared norm of the current sample
	/// \param  C         upper bound on the variables
	/// \param  y         label of the current sample
	/// \param  alpha     input: initial point; output: (near) optimal point
	/// \param  mu        step from initial point to final point
	///
	/// \return  The function must return the gain of the step, i.e., the improvement of the objective function.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu) = 0;

	DataView<const DatasetType> m_data;               ///< view on training data
	RealVector m_xSquared;                            ///< diagonal entries of the quadratic matrix
	std::size_t m_dim;                                ///< input space dimension
	std::size_t m_classes;                            ///< number of classes
};


//~ /// \brief Generic solver skeleton for linear multi-class SVM problems.
//~ template < >
//~ class QpMcLinear<CompressedRealVector>
//~ {
//~ public:
	//~ typedef LabeledData<CompressedRealVector, unsigned int> DatasetType;

	//~ ///
	//~ /// \brief Constructor
	//~ ///
	//~ /// \param  dataset  training data
	//~ /// \param  dim      problem dimension
	//~ /// \param  classes  number of classes in the problem
	//~ ///
	//~ QpMcLinear(
			//~ const DatasetType& dataset,
			//~ std::size_t dim,
			//~ std::size_t classes)
	//~ : m_data(dataset.numberOfElements())
	//~ , m_xSquared(dataset.numberOfElements())
	//~ , m_dim(dim)
	//~ , m_classes(classes)
	//~ {
		//~ SHARK_ASSERT(m_dim > 0);

		//~ // transform ublas sparse vectors into a fast format
		//~ // (yes, ublas is slow...), and compute the squared
		//~ // norms of the training examples
		//~ SparseVector sparse;
		//~ for (std::size_t b=0, j=0; b<dataset.numberOfBatches(); b++)
		//~ {
			//~ DatasetType::const_batch_reference batch = dataset.batch(b);
			//~ for (std::size_t i=0; i<batch.size(); i++)
			//~ {
				//~ CompressedRealVector x_i = shark::get(batch, i).input;
				//~ unsigned int y_i = shark::get(batch, i).label;
				//~ m_data[j].label = y_i;
				//~ double d = 0.0;
				//~ for (CompressedRealVector::const_iterator it=x_i.begin(); it != x_i.end(); ++it)
				//~ {
					//~ double v = *it;
					//~ sparse.index = it.index();
					//~ sparse.value = v;
					//~ storage.push_back(sparse);
					//~ d += v * v;
				//~ }
				//~ sparse.index = (std::size_t)-1;
				//~ storage.push_back(sparse);
				//~ m_xSquared(j) = d;
				//~ j++;
			//~ }
		//~ }
		//~ for (std::size_t i=0, k=0; i<m_data.size(); i++)
		//~ {
			//~ CompressedRealVector x_i = dataset.element(i).input;
			//~ m_data[i].input = &storage[k];
			//~ for (; storage[k].index != (std::size_t)-1; k++);
			//~ k++;
		//~ }
	//~ }

	//~ ///
	//~ /// \brief Solve the SVM training problem.
	//~ ///
	//~ /// \param  C        regularization constant of the SVM
	//~ /// \param  stop     stopping condition(s)
	//~ /// \param  prop     solution properties
	//~ /// \param  verbose  if true, the solver prints status information and solution statistics
	//~ ///
	//~ RealMatrix solve(
			//~ double C,
			//~ QpStoppingCondition& stop,
			//~ QpSolutionProperties* prop = NULL,
			//~ bool verbose = false)
	//~ {
		//~ // sanity checks
		//~ SHARK_ASSERT(C > 0.0);

		//~ // measure training time
		//~ Timer timer;

		//~ // prepare dimensions and vectors
		//~ std::size_t ell = m_data.size();             // number of training examples
		//~ RealMatrix alpha(ell, m_classes + 1, 0.0);   // Lagrange multipliers; dual variables. Reserve one extra column.
		//~ RealMatrix w(m_classes, m_dim, 0.0);         // weight vectors; primal variables

		//~ // scheduling of steps
		//~ RealVector pref(ell, 1.0);                   // example-wise measure of success
		//~ double prefsum = ell;                        // normalization constant
		//~ std::vector<std::size_t> schedule(ell);

		//~ // prepare counters
		//~ std::size_t epoch = 0;
		//~ std::size_t steps = 0;

		//~ // prepare performance monitoring
		//~ double objective = 0.0;
		//~ double max_violation = 0.0;
		//~ const double gain_learning_rate = 1.0 / ell;
		//~ double average_gain = 0.0;

		//~ // outer optimization loop (epochs)
		//~ bool canstop = true;
		//~ while (true)
		//~ {
			//~ // define schedule
			//~ double psum = prefsum;
			//~ prefsum = 0.0;
			//~ std::size_t pos = 0;
			//~ for (std::size_t i=0; i<ell; i++)
			//~ {
				//~ double p = pref(i);
				//~ double num = (psum < 1e-6) ? ell - pos : std::min((double)(ell - pos), (ell - pos) * p / psum);
				//~ std::size_t n = (std::size_t)std::floor(num);
				//~ double prob = num - n;
				//~ if (Rng::uni() < prob) n++;
				//~ for (std::size_t j=0; j<n; j++)
				//~ {
					//~ schedule[pos] = i;
					//~ pos++;
				//~ }
				//~ psum -= p;
				//~ prefsum += p;
			//~ }
			//~ SHARK_ASSERT(pos == ell);
			//~ for (std::size_t i=0; i<ell; i++) std::swap(schedule[i], schedule[Rng::discrete(0, ell - 1)]);

			//~ // inner loop (one epoch)
			//~ max_violation = 0.0;
			//~ for (std::size_t j=0; j<ell; j++)
			//~ {
				//~ // active example
				//~ double gain = 0.0;
				//~ const std::size_t i = schedule[j];
				//~ const SparseVector* x_i = m_data[i].input;
				//~ const unsigned int y_i = m_data[i].label;
				//~ const double q = m_xSquared(i);
				//~ RealMatrixRow a = row(alpha, i);

				//~ // compute gradient and KKT violation
				//~ RealVector wx(m_classes, 0.0);
				//~ for (const SparseVector* p=x_i; p->index != (std::size_t)-1; p++)
				//~ {
					//~ const std::size_t idx = p->index;
					//~ const double v = p->value;
					//~ for (size_t c=0; c<m_classes; c++) wx(c) += w(c, idx) * v;
				//~ }
				//~ RealVector g(m_classes);
				//~ double kkt = calcGradient(g, wx, a, C, y_i);

				//~ if (kkt > 0.0)
				//~ {
					//~ max_violation = std::max(max_violation, kkt);

					//~ // perform the step on alpha
					//~ RealVector mu(m_classes, 0.0);
					//~ gain = solveSub(0.1 * stop.minAccuracy, g, q, C, y_i, a, mu);
					//~ objective += gain;
					//~ steps++;

					//~ // update weight vectors
					//~ updateWeightVectors(w, mu, i);
				//~ }

				//~ // update gain-based preferences
				//~ {
					//~ if (epoch == 0) average_gain += gain / (double)ell;
					//~ else
					//~ {
						//~ double change = CHANGE_RATE * (gain / average_gain - 1.0);
						//~ double newpref = std::min(PREF_MAX, std::max(PREF_MIN, pref(i) * std::exp(change)));
						//~ prefsum += newpref - pref(i);
						//~ pref(i) = newpref;
						//~ average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
					//~ }
				//~ }
			//~ }

			//~ epoch++;

			//~ // stopping criteria
			//~ if (stop.maxIterations > 0 && epoch * ell >= stop.maxIterations)
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
					//~ for (std::size_t i=0; i<ell; i++) pref(i) = 1.0;
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

		//~ // calculate dual objective value
		//~ objective = 0.0;
		//~ for (std::size_t j=0; j<m_classes; j++)
		//~ {
			//~ for (std::size_t d=0; d<m_dim; d++) objective -= w(j, d) * w(j, d);
		//~ }
		//~ objective *= 0.5;
		//~ for (std::size_t i=0; i<ell; i++)
		//~ {
			//~ for (std::size_t j=0; j<m_classes; j++) objective += alpha(i, j);
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
		//~ }

		//~ // return the solution
		//~ return w;
	//~ }

//~ protected:
	//~ /// \brief Data structure for sparse vectors.
	//~ struct SparseVector
	//~ {
		//~ std::size_t index;
		//~ double value;
	//~ };

	//~ struct ElementType
	//~ {
		//~ const SparseVector* input;
		//~ unsigned int label;
	//~ };

	//~ // for all c: row(w, c) += mu(c) * x
	//~ void add_scaled(RealMatrix& w, RealVector const& mu, const SparseVector* x)
	//~ {
		//~ for (; x->index != (std::size_t)-1; x++)
		//~ {
			//~ const std::size_t index = x->index;
			//~ const double value = x->value;
			//~ for (std::size_t c=0; c<m_classes; c++) w(c, index) += mu(c) * value;
		//~ }
	//~ }

	//~ /// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	//~ ///
	//~ /// \param  gradient  gradient vector to be filled in. The vector is correctly sized.
	//~ /// \param  wx        inner products of weight vectors with the current sample; wx(c) = <w_c, x>
	//~ /// \param  alpha     variables corresponding to the current sample
	//~ /// \param  C         upper bound on the variables
	//~ /// \param  y         label of the current sample
	//~ ///
	//~ /// \return  The function must return the violation of the KKT conditions.
	//~ virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y) = 0;

	//~ /// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	//~ ///
	//~ /// \param  w      matrix of (dense) weight vectors (as rows)
	//~ /// \param  mu     dual step on the variables corresponding to the current sample
	//~ /// \param  index  example index
	//~ virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index) = 0;

	//~ /// \brief Solve the sub-problem posed by a single training example.
	//~ ///
	//~ /// \param  epsilon   accuracy (dual gradient) up to which the sub-problem should be solved
	//~ /// \param  gradient  gradient of the objective function w.r.t. alpha
	//~ /// \param  q         squared norm of the current sample
	//~ /// \param  C         upper bound on the variables
	//~ /// \param  y         label of the current sample
	//~ /// \param  alpha     input: initial point; output: (near) optimal point
	//~ /// \param  mu        step from initial point to final point
	//~ ///
	//~ /// \return  The function must return the gain of the step, i.e., the improvement of the objective function.
	//~ virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu) = 0;

	//~ std::vector<SparseVector> storage;                ///< storage for sparse vectors
	//~ std::vector<ElementType> m_data;                  ///< resembles data view interface
	//~ RealVector m_xSquared;                            ///< squared norms of the training data
	//~ std::size_t m_dim;                                ///< input space dimension
	//~ std::size_t m_classes;                            ///< number of classes
//~ };


/// \brief Solver for the multi-class SVM by Weston & Watkins.
template <class InputT>
class QpMcLinearWW : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearWW(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		double violation = 0.0;
		for (std::size_t c=0; c<wx.size(); c++)
		{
			if (c == y)
			{
				gradient(c) = 0.0;
			}
			else
			{
				const double g = 1.0 - 0.5 * (wx(y) - wx(c));
				gradient(c) = g;
				if (g > violation && alpha(c) < C) violation = g;
				else if (-g > violation && alpha(c) > 0.0) violation = -g;
			}
		}
		return violation;
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		double sum_mu = 0.0;
		for (std::size_t c=0; c<m_classes; c++) sum_mu += mu(c);
		unsigned int y = m_data[index].label;
		RealVector step(-0.5 * mu);
		step(y) = 0.5 * sum_mu;
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double qq = 0.5 * q;
		double gain = 0.0;

		// SMO loop
		size_t iter, maxiter = MAXITER_MULTIPLIER * m_classes;
		for (iter=0; iter<maxiter; iter++)
		{
			// select working set
			std::size_t idx = 0;
			double kkt = 0.0;
			for (std::size_t c=0; c<m_classes; c++)
			{
				if (c == y) continue;

				const double g = gradient(c);
				const double a = alpha(c);
				if (g > kkt && a < C) { kkt = g; idx = c; }
				else if (-g > kkt && a > 0.0) { kkt = -g; idx = c; }
			}

			// check stopping criterion
			if (kkt < epsilon) break;

			// perform step
			const double a = alpha(idx);
			const double g = gradient(idx);
			double m = g / qq;
			double a_new = a + m;
			if (a_new <= 0.0)
			{
				m = -a;
				a_new = 0.0;
			}
			else if (a_new >= C)
			{
				m = C - a;
				a_new = C;
			}
			alpha(idx) = a_new;
			mu(idx) += m;

			// update gradient and total gain
			const double dg = 0.5 * m * qq;
			for (std::size_t c=0; c<m_classes; c++) gradient(c) -= dg;
			gradient(idx) -= dg;

			gain += m * (g - dg);
		}

		return gain;
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


/// \brief Solver for the multi-class SVM by Lee, Lin & Wahba.
template <class InputT>
class QpMcLinearLLW : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearLLW(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		double violation = 0.0;
		for (std::size_t c=0; c<m_classes; c++)
		{
			if (c == y)
			{
				gradient(c) = 0.0;
			}
			else
			{
				const double g = 1.0 + wx(c);
				gradient(c) = g;
				if (g > violation && alpha(c) < C) violation = g;
				else if (-g > violation && alpha(c) > 0.0) violation = -g;
			}
		}
		return violation;
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		double mean_mu = 0.0;
		for (std::size_t c=0; c<m_classes; c++) mean_mu += mu(c);
		mean_mu /= (double)m_classes;
		RealVector step(m_classes);
		for (std::size_t c=0; c<m_classes; c++) step(c) = mean_mu - mu(c);
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double ood = 1.0 / m_classes;
		const double qq = (1.0 - ood) * q;
		double gain = 0.0;

		// SMO loop
		size_t iter, maxiter = MAXITER_MULTIPLIER * m_classes;
		for (iter=0; iter<maxiter; iter++)
		{
			// select working set
			std::size_t idx = 0;
			double kkt = 0.0;
			for (std::size_t c=0; c<m_classes; c++)
			{
				if (c == y) continue;

				const double g = gradient(c);
				const double a = alpha(c);
				if (g > kkt && a < C) { kkt = g; idx = c; }
				else if (-g > kkt && a > 0.0) { kkt = -g; idx = c; }
			}

			// check stopping criterion
			if (kkt < epsilon) break;

			// perform step
			const double a = alpha(idx);
			const double g = gradient(idx);
			double m = g / qq;
			double a_new = a + m;
			if (a_new <= 0.0)
			{
				m = -a;
				a_new = 0.0;
			}
			else if (a_new >= C)
			{
				m = C - a;
				a_new = C;
			}
			alpha(idx) = a_new;
			mu(idx) += m;

			// update gradient and total gain
			const double dg = m * q;
			const double dgc = dg / m_classes;
			for (std::size_t c=0; c<m_classes; c++) gradient(c) += dgc;
			gradient(idx) -= dg;

			gain += m * (g - 0.5 * (dg - dgc));
		}

		return gain;
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


/// \brief Solver for the multi-class SVM with absolute margin and total sum loss.
template <class InputT>
class QpMcLinearATS : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearATS(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		double violation = 0.0;
		for (std::size_t c=0; c<m_classes; c++)
		{
			const double g = (c == y) ? 1.0 - wx(y) : 1.0 + wx(c);
			gradient(c) = g;
			if (g > violation && alpha(c) < C) violation = g;
			else if (-g > violation && alpha(c) > 0.0) violation = -g;
		}
		return violation;
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		unsigned int y = m_data[index].label;
		double mean = -2.0 * mu(y);
		for (std::size_t c=0; c<m_classes; c++) mean += mu(c);
		mean /= (double)m_classes;
		RealVector step(m_classes);
		for (std::size_t c=0; c<m_classes; c++) step(c) = ((c == y) ? (mu(c) + mean) : (mean - mu(c)));
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double ood = 1.0 / m_classes;
		const double qq = (1.0 - ood) * q;
		double gain = 0.0;

		// SMO loop
		size_t iter, maxiter = MAXITER_MULTIPLIER * m_classes;
		for (iter=0; iter<maxiter; iter++)
		{
			// select working set
			std::size_t idx = 0;
			double kkt = 0.0;
			for (std::size_t c=0; c<m_classes; c++)
			{
				const double g = gradient(c);
				const double a = alpha(c);
				if (g > kkt && a < C) { kkt = g; idx = c; }
				else if (-g > kkt && a > 0.0) { kkt = -g; idx = c; }
			}

			// check stopping criterion
			if (kkt < epsilon) break;

			// perform step
			const double a = alpha(idx);
			const double g = gradient(idx);
			double m = g / qq;
			double a_new = a + m;
			if (a_new <= 0.0)
			{
				m = -a;
				a_new = 0.0;
			}
			else if (a_new >= C)
			{
				m = C - a;
				a_new = C;
			}
			alpha(idx) = a_new;
			mu(idx) += m;

			// update gradient and total gain
			const double dg = m * q;
			const double dgc = dg / m_classes;
			if (idx == y)
			{
				for (std::size_t c=0; c<m_classes; c++) gradient(c) -= dgc;
				gradient(idx) -= dg - 2.0 * dgc;
			}
			else
			{
				for (std::size_t c=0; c<m_classes; c++) gradient(c) += (c == y) ? -dgc : dgc;
				gradient(idx) -= dg;
			}

			gain += m * (g - 0.5 * (dg - dgc));
		}

		return gain;
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


/// \brief Solver for the multi-class maximum margin regression SVM
template <class InputT>
class QpMcLinearMMR : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearMMR(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		for (std::size_t c=0; c<m_classes; c++) gradient(c) = 0.0;
		const double g = 1.0 - wx(y);
		gradient(y) = g;
		const double a = alpha(0);
		if (g > 0.0)
		{
			if (a == C) return 0.0;
			else return g;
		}
		else
		{
			if (a == 0.0) return 0.0;
			else return -g;
		}
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		unsigned int y = m_data[index].label;
		double s = mu(0);
		double sc = -s / m_classes;
		double sy = s + sc;
		RealVector step(m_classes);
		for (size_t c=0; c<m_classes; c++) step(c) = (c == y) ? sy : sc;
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double ood = 1.0 / m_classes;
		const double qq = (1.0 - ood) * q;

		double kkt = 0.0;
		const double g = gradient(y);
		const double a = alpha(0);
		if (g > kkt && a < C) kkt = g;
		else if (-g > kkt && a > 0.0) kkt = -g;

		// check stopping criterion
		if (kkt < epsilon) return 0.0;

		// perform step
		double m = g / qq;
		double a_new = a + m;
		if (a_new <= 0.0)
		{
			m = -a;
			a_new = 0.0;
		}
		else if (a_new >= C)
		{
			m = C - a;
			a_new = C;
		}
		alpha(0) = a_new;
		mu(0) = m;

		// return the gain
		return m * (g - 0.5 * m * qq);
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


/// \brief Solver for the multi-class SVM by Crammer & Singer.
template <class InputT>
class QpMcLinearCS : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearCS(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		if (alpha(m_classes) < C)
		{
			double violation = 0.0;
			for (std::size_t c=0; c<wx.size(); c++)
			{
				if (c == y)
				{
					gradient(c) = 0.0;
				}
				else
				{
					const double g = 1.0 - 0.5 * (wx(y) - wx(c));
					gradient(c) = g;
					if (g > violation) violation = g;
					else if (-g > violation && alpha(c) > 0.0) violation = -g;
				}
			}
			return violation;
		}
		else
		{
			// double kkt_up = -1e100, kkt_down = 1e100;
			double kkt_up = 0.0, kkt_down = 1e100;
			for (std::size_t c=0; c<m_classes; c++)
			{
				if (c == y)
				{
					gradient(c) = 0.0;
				}
				else
				{
					const double g = 1.0 - 0.5 * (wx(y) - wx(c));
					gradient(c) = g;
					if (g > kkt_up && alpha(c) < C) kkt_up = g;
					if (g < kkt_down && alpha(c) > 0.0) kkt_down = g;
				}
			}
			return std::max(0.0, kkt_up - kkt_down);
		}
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		unsigned int y = m_data[index].label;
		double sum_mu = 0.0;
		for (std::size_t c=0; c<m_classes; c++) if (c != y) sum_mu += mu(c);
		RealVector step(-0.5 * mu);
		step(y) = 0.5 * sum_mu;
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double qq = 0.5 * q;
		double gain = 0.0;

		// SMO loop
		size_t iter, maxiter = MAXITER_MULTIPLIER * m_classes;
		for (iter=0; iter<maxiter; iter++)
		{
			// select working set
			std::size_t idx = 0;
			std::size_t idx_up = 0, idx_down = 0;
			bool size2 = false;
			double kkt = 0.0;
			double grad = 0.0;
			if (alpha(m_classes) == C)
			{
				double kkt_up = -1e100, kkt_down = 1e100;
				for (std::size_t c=0; c<m_classes; c++)
				{
					if (c == y) continue;

					const double g = gradient(c);
					const double a = alpha(c);
					if (g > kkt_up && a < C) { kkt_up = g; idx_up = c; }
					if (g < kkt_down && a > 0.0) { kkt_down = g; idx_down = c; }
				}

				if (kkt_up <= 0.0)
				{
					idx = idx_down;
					grad = kkt_down;
					kkt = -kkt_down;
				}
				else
				{
					grad = kkt_up - kkt_down;
					kkt = grad;
					size2 = true;
				}
			}
			else
			{
				for (std::size_t c=0; c<m_classes; c++)
				{
					if (c == y) continue;

					const double g = gradient(c);
					const double a = alpha(c);
					if (g > kkt) { kkt = g; idx = c; }
					else if (-g > kkt && a > 0.0) { kkt = -g; idx = c; }
				}
				grad = gradient(idx);
			}

			// check stopping criterion
			if (kkt < epsilon) return gain;

			if (size2)
			{
				// perform step
				const double a_up = alpha(idx_up);
				const double a_down = alpha(idx_down);
				double m = grad / qq;
				double a_up_new = a_up + m;
				double a_down_new = a_down - m;
				if (a_down_new <= 0.0)
				{
					m = a_down;
					a_up_new = a_up + m;
					a_down_new = 0.0;
				}
				alpha(idx_up) = a_up_new;
				alpha(idx_down) = a_down_new;
				mu(idx_up) += m;
				mu(idx_down) -= m;

				// update gradient and total gain
				const double dg = 0.5 * m * qq;
				gradient(idx_up) -= dg;
				gradient(idx_down) += dg;
				gain += m * (grad - 2.0 * dg);
			}
			else
			{
				// perform step
				const double a = alpha(idx);
				const double a_sum = alpha(m_classes);
				double m = grad / qq;
				double a_new = a + m;
				double a_sum_new = a_sum + m;
				if (a_new <= 0.0)
				{
					m = -a;
					a_new = 0.0;
					a_sum_new = a_sum + m;
				}
				else if (a_sum_new >= C)
				{
					m = C - a_sum;
					a_sum_new = C;
					a_new = a + m;
				}
				alpha(idx) = a_new;
				alpha(m_classes) = a_sum_new;
				mu(idx) += m;

				// update gradient and total gain
				const double dg = 0.5 * m * qq;
				for (std::size_t c=0; c<m_classes; c++) gradient(c) -= dg;
				gradient(idx) -= dg;
				gain += m * (grad - dg);
			}
		}

		return gain;
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


/// \brief Solver for the multi-class SVM with absolute margin and discriminative maximum loss.
template <class InputT>
class QpMcLinearADM : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearADM(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		if (alpha(m_classes) < C)
		{
			double violation = 0.0;
			for (std::size_t c=0; c<m_classes; c++)
			{
				if (c == y)
				{
					gradient(c) = 0.0;
				}
				else
				{
					const double g = 1.0 + wx(c);
					gradient(c) = g;
					if (g > violation) violation = g;
					else if (-g > violation && alpha(c) > 0.0) violation = -g;
				}
			}
			return violation;
		}
		else
		{
			double kkt_up = 0.0, kkt_down = 1e100;
			for (std::size_t c=0; c<m_classes; c++)
			{
				if (c == y)
				{
					gradient(c) = 0.0;
				}
				else
				{
					const double g = 1.0 + wx(c);
					gradient(c) = g;
					if (g > kkt_up && alpha(c) < C) kkt_up = g;
					if (g < kkt_down && alpha(c) > 0.0) kkt_down = g;
				}
			}
			return std::max(0.0, kkt_up - kkt_down);
		}
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		double mean_mu = 0.0;
		for (std::size_t c=0; c<m_classes; c++) mean_mu += mu(c);
		mean_mu /= (double)m_classes;
		RealVector step(m_classes);
		for (size_t c=0; c<m_classes; c++) step(c) = mean_mu - mu(c);
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double ood = 1.0 / m_classes;
		const double qq = (1.0 - ood) * q;
		double gain = 0.0;

		// SMO loop
		size_t iter, maxiter = MAXITER_MULTIPLIER * m_classes;
		for (iter=0; iter<maxiter; iter++)
		{
			// select working set
			std::size_t idx = 0;
			std::size_t idx_up = 0, idx_down = 0;
			bool size2 = false;
			double kkt = 0.0;
			double grad = 0.0;
			if (alpha(m_classes) == C)
			{
				double kkt_up = -1e100, kkt_down = 1e100;
				for (std::size_t c=0; c<m_classes; c++)
				{
					if (c == y) continue;

					const double g = gradient(c);
					const double a = alpha(c);
					if (g > kkt_up && a < C) { kkt_up = g; idx_up = c; }
					if (g < kkt_down && a > 0.0) { kkt_down = g; idx_down = c; }
				}

				if (kkt_up <= 0.0)
				{
					idx = idx_down;
					grad = kkt_down;
					kkt = -kkt_down;
				}
				else
				{
					grad = kkt_up - kkt_down;
					kkt = grad;
					size2 = true;
				}
			}
			else
			{
				for (std::size_t c=0; c<m_classes; c++)
				{
					if (c == y) continue;

					const double g = gradient(c);
					const double a = alpha(c);
					if (g > kkt) { kkt = g; idx = c; }
					else if (-g > kkt && a > 0.0) { kkt = -g; idx = c; }
				}
				grad = gradient(idx);
			}

			// check stopping criterion
			if (kkt < epsilon) return gain;

			if (size2)
			{
				// perform step
				const double a_up = alpha(idx_up);
				const double a_down = alpha(idx_down);
				double m = grad / (2.0 * q);
				double a_up_new = a_up + m;
				double a_down_new = a_down - m;
				if (a_down_new <= 0.0)
				{
					m = a_down;
					a_up_new = a_up + m;
					a_down_new = 0.0;
				}
				alpha(idx_up) = a_up_new;
				alpha(idx_down) = a_down_new;
				mu(idx_up) += m;
				mu(idx_down) -= m;

				// update gradient and total gain
				const double dg = m * q;
				const double dgc = dg / m_classes;
				gradient(idx_up) -= dg;
				gradient(idx_down) += dg;
				gain += m * (grad - (dg - dgc));
			}
			else
			{
				// perform step
				const double a = alpha(idx);
				const double a_sum = alpha(m_classes);
				double m = grad / qq;
				double a_new = a + m;
				double a_sum_new = a_sum + m;
				if (a_new <= 0.0)
				{
					m = -a;
					a_new = 0.0;
					a_sum_new = a_sum + m;
				}
				else if (a_sum_new >= C)
				{
					m = C - a_sum;
					a_sum_new = C;
					a_new = a + m;
				}
				alpha(idx) = a_new;
				alpha(m_classes) = a_sum_new;
				mu(idx) += m;

				// update gradient and total gain
				const double dg = m * q;
				const double dgc = dg / m_classes;
				for (std::size_t c=0; c<m_classes; c++) gradient(c) += dgc;
				gradient(idx) -= dg;
				gain += m * (grad - 0.5 * (dg - dgc));
			}
		}

		return gain;
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


/// \brief Solver for the multi-class SVM with absolute margin and total maximum loss.
template <class InputT>
class QpMcLinearATM : public QpMcLinear<InputT>
{
public:
	typedef LabeledData<InputT, unsigned int> DatasetType;

	/// \brief Constructor
	QpMcLinearATM(
			const DatasetType& dataset,
			std::size_t dim,
			std::size_t classes)
	: QpMcLinear<InputT>(dataset, dim, classes)
	{ }

protected:
	/// \brief Compute the gradient from the inner products of the weight vectors with the current sample.
	virtual double calcGradient(RealVector& gradient, RealVector wx, RealMatrixRow const& alpha, double C, unsigned int y)
	{
		if (alpha(m_classes) < C)
		{
			double violation = 0.0;
			for (std::size_t c=0; c<m_classes; c++)
			{
				const double g = (c == y) ? 1.0 - wx(y) : 1.0 + wx(c);
				gradient(c) = g;
				if (g > violation) violation = g;
				else if (-g > violation && alpha(c) > 0.0) violation = -g;
			}
			return violation;
		}
		else
		{
			double kkt_up = 0.0, kkt_down = 1e100;
			for (std::size_t c=0; c<m_classes; c++)
			{
				const double g = (c == y) ? 1.0 - wx(y) : 1.0 + wx(c);
				gradient(c) = g;
				if (g > kkt_up && alpha(c) < C) kkt_up = g;
				if (g < kkt_down && alpha(c) > 0.0) kkt_down = g;
			}
			return std::max(0.0, kkt_up - kkt_down);
		}
	}

	/// \brief Update the weight vectors (primal variables) after a step on the dual variables.
	virtual void updateWeightVectors(RealMatrix& w, RealVector const& mu, std::size_t index)
	{
		unsigned int y = m_data[index].label;
		double mean = -2.0 * mu(y);
		for (std::size_t c=0; c<m_classes; c++) mean += mu(c);
		mean /= (double)m_classes;
		RealVector step(m_classes);
		for (size_t c=0; c<m_classes; c++) step(c) = (c == y) ? (mu(c) + mean) : (mean - mu(c));
		add_scaled(w, step, m_data[index].input);
	}

	/// \brief Solve the sub-problem posed by a single training example.
	virtual double solveSub(double epsilon, RealVector gradient, double q, double C, unsigned int y, RealMatrixRow& alpha, RealVector& mu)
	{
		const double ood = 1.0 / m_classes;
		const double qq = (1.0 - ood) * q;
		double gain = 0.0;

		// SMO loop
		size_t iter, maxiter = MAXITER_MULTIPLIER * m_classes;
		for (iter=0; iter<maxiter; iter++)
		{
			// select working set
			std::size_t idx = 0;
			std::size_t idx_up = 0, idx_down = 0;
			bool size2 = false;
			double kkt = 0.0;
			double grad = 0.0;
			if (alpha(m_classes) == C)
			{
				double kkt_up = -1e100, kkt_down = 1e100;
				for (std::size_t c=0; c<m_classes; c++)
				{
					const double g = gradient(c);
					const double a = alpha(c);
					if (g > kkt_up && a < C) { kkt_up = g; idx_up = c; }
					if (g < kkt_down && a > 0.0) { kkt_down = g; idx_down = c; }
				}

				if (kkt_up <= 0.0)
				{
					idx = idx_down;
					grad = kkt_down;
					kkt = -kkt_down;
				}
				else
				{
					grad = kkt_up - kkt_down;
					kkt = grad;
					size2 = true;
				}
			}
			else
			{
				for (std::size_t c=0; c<m_classes; c++)
				{
					const double g = gradient(c);
					const double a = alpha(c);
					if (g > kkt) { kkt = g; idx = c; }
					else if (-g > kkt && a > 0.0) { kkt = -g; idx = c; }
				}
				grad = gradient(idx);
			}

			// check stopping criterion
			if (kkt < epsilon) return gain;

			if (size2)
			{
				// perform step
				const double a_up = alpha(idx_up);
				const double a_down = alpha(idx_down);
				double m = grad / (2.0 * q);
				double a_up_new = a_up + m;
				double a_down_new = a_down - m;
				if (a_down_new <= 0.0)
				{
					m = a_down;
					a_up_new = a_up + m;
					a_down_new = 0.0;
				}
				alpha(idx_up) = a_up_new;
				alpha(idx_down) = a_down_new;
				mu(idx_up) += m;
				mu(idx_down) -= m;

				// update gradient and total gain
				const double dg = m * q;
				const double dgc = dg / m_classes;
				if (idx_up == y)
				{
					for (std::size_t c=0; c<m_classes; c++) gradient(c) -= dgc;
					gradient(idx_up) -= dg - 2.0 * dgc;
					gradient(idx_down) += dg;
				}
				else if (idx_down == y)
				{
					gradient(idx_up) -= dg;
					gradient(idx_down) += dg - 2.0 * dgc;
				}
				else
				{
					gradient(idx_up) -= dg;
					gradient(idx_down) += dg;
				}
				gain += m * (grad - (dg - dgc));
			}
			else
			{
				// perform step
				const double a = alpha(idx);
				const double a_sum = alpha(m_classes);
				double m = grad / qq;
				double a_new = a + m;
				double a_sum_new = a_sum + m;
				if (a_new <= 0.0)
				{
					m = -a;
					a_new = 0.0;
					a_sum_new = a_sum + m;
				}
				else if (a_sum_new >= C)
				{
					m = C - a_sum;
					a_sum_new = C;
					a_new = a + m;
				}
				alpha(idx) = a_new;
				alpha(m_classes) = a_sum_new;
				mu(idx) += m;

				// update gradient and total gain
				const double dg = m * q;
				const double dgc = dg / m_classes;
				if (idx == y)
				{
					for (std::size_t c=0; c<m_classes; c++) gradient(c) -= dgc;
					gradient(idx) -= dg - 2.0 * dgc;
				}
				else
				{
					for (std::size_t c=0; c<m_classes; c++) gradient(c) += (c == y) ? -dgc : dgc;
					gradient(idx) -= dg;
				}
				gain += m * (grad - 0.5 * (dg - dgc));
			}
		}

		return gain;
	}

protected:
	using QpMcLinear<InputT>::add_scaled;
	using QpMcLinear<InputT>::m_data;
	using QpMcLinear<InputT>::m_classes;
};


}
#endif
