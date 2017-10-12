//===========================================================================
/*!
 * 
 *
 * \brief       Definition of an OpenML Run.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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

#ifndef SHARK_OPENML_RUN_H
#define SHARK_OPENML_RUN_H


#include "Entity.h"
#include "CachedFile.h"
#include "Task.h"
#include "Flow.h"
#include <shark/Data/Dataset.h>
#include <shark/Data/Arff.h>
#include <vector>


namespace shark {
namespace openML {


/// \brief Representation of an OpenML run.
SHARK_EXPORT_SYMBOL class Run : public Entity
{
public:
	/// \brief Construct an existing OpenML run from its ID.
	Run(IDType id, bool downloadPredictions = false);

	/// \brief Construct a new OpenML run from task and flow.
	///
	/// The run is created in memory, but not yet in the OpenML service.
	/// Call Run::commit to complete the process.
	Run(std::shared_ptr<Task> task, std::shared_ptr<Flow> flow);

	/// \brief Add a tag to the entity.
	void tag(std::string const& tagname) override;

	/// \brief Remove a tag from the entity.
	void untag(std::string const& tagname) override;

	/// \brief Print a human readable summary of the entity.
	void print(std::ostream& os = std::cout) const override;

	/// \brief Obtain the underlying task.
	std::shared_ptr<Task> task()
	{ return m_task; }

	/// \brief Obtain the underlying task.
	std::shared_ptr<const Task> task() const
	{ return m_task; }

	/// \brief Obtain the underlying flow.
	std::shared_ptr<Flow> flow()
	{ return m_flow; }

	/// \brief Obtain the underlying flow.
	std::shared_ptr<const Flow> flow() const
	{ return m_flow; }

	/// \brief Load the predictions into memory.
	void load() const;

	/// \brief Obtain the number of hyperparameters of the flow.
	std::size_t numberOfHyperparameters() const
	{ return m_flow->numberOfHyperparameters(); }

	/// \brief Obtain the value of a hyperparameter by name.
	///
	/// Each value can be obtained as a string. This function casts the
	/// value to the requested return type with boost::lexical_cast.
	template <typename ValueType = std::string>
	ValueType hyperparameterValue(std::string const& name)
	{
		return boost::lexical_cast<ValueType>(m_hyperparameterValue[m_flow->hyperparameterIndex(name)]);
	}

	/// \brief Define the value of a hyperparameter used for this run.
	///
	/// Call this function only on new runs. It fails for runs that were
	/// already committed to the server, and also for runs that were
	/// obtained from the server.
	template <typename ValueType>
	void setHyperparameterValue(std::string const& name, ValueType const& value)
	{
		if (id() != invalidID) throw SHARKEXCEPTION("[Rub::setHyperparameterValue] Cannot set hyperparameter value for a committed run.");
		m_hyperparameterValue[m_flow->hyperparameterIndex(name)] = boost::lexical_cast<std::string>(value);
	}

	/// \brief Obtain predictions for a given repetition and fold.
	template <typename ContainerType>
	void predictions(std::size_t repetition, std::size_t fold, ContainerType& predictions) const
	{
		load();
		RANGE_CHECK(repetition < m_predictions.size());
		SHARK_ASSERT(m_predictions.size() == m_task->repetitions());
		std::vector<double> const& p = m_predictions[repetition];
		std::vector<std::size_t> const& split = m_task->splitIndices(repetition);
		SHARK_ASSERT(p.size() == split.size());
		predictions.clear();
		for (std::size_t i=0; i<p.size(); i++)
		{
			if (split[i] == fold) predictions.push_back(p[i]);
		}
	}

	/// \brief Obtain predictions for a given repetition and fold.
	template <typename VectorType>
	void predictions(std::size_t repetition, std::size_t fold, Data<VectorType>& predictions) const
	{
		load();
		RANGE_CHECK(repetition < m_predictions.size());
		SHARK_ASSERT(m_predictions.size() == m_task->repetitions());
		std::vector<double> const& p = m_predictions[repetition];
		std::vector<std::size_t> const& split = m_task->splitIndices(repetition);
		SHARK_ASSERT(p.size() == split.size());
		std::vector<VectorType> tmp;
		VectorType v(1);
		for (std::size_t i=0; i<p.size(); i++)
		{
			if (split[i] == fold)
			{
				v[0] = p[i];
				tmp.push_back(v);
			}
		}
		predictions = createDataFromRange(tmp);
	}

	/// \brief Obtain predictions for a given repetition and fold.
	inline void predictions(std::size_t repetition, std::size_t fold, Data<unsigned int>& predictions) const
	{
		load();
		RANGE_CHECK(repetition < m_predictions.size());
		SHARK_ASSERT(m_predictions.size() == m_task->repetitions());
		std::vector<double> const& p = m_predictions[repetition];
		std::vector<std::size_t> const& split = m_task->splitIndices(repetition);
		SHARK_ASSERT(p.size() == split.size());
		std::vector<unsigned int> tmp;
		for (std::size_t i=0; i<p.size(); i++)
		{
			if (split[i] == fold) tmp.push_back(static_cast<unsigned int>(p[i]));
		}
		predictions = createDataFromRange(tmp);
	}

	/// \brief Store predictions.
	///
	/// Call this function only on new runs. It fails for runs that were
	/// already committed to the server, and also for runs that were
	/// obtained from the server.
	template <typename ContainerType>
	void setPredictions(std::size_t repetition, std::size_t fold, ContainerType const& predictions)
	{
		if (id() != invalidID) throw SHARKEXCEPTION("Cannot set predictions for a committed run.");

		SHARK_ASSERT(repetition < m_task->repetitions());
		SHARK_ASSERT(fold < m_task->folds());

		std::vector<std::size_t> const& split = m_task->splitIndices(repetition);
		std::vector<double>& pred = m_predictions[repetition];
		SHARK_ASSERT(split.size() == pred.size());
		std::size_t j = 0;
		for (std::size_t i=0; i<predictions.size(); i++)
		{
			while (split[j] != fold) j++;
			pred[j] = predictions[i];
			SHARK_ASSERT(j < pred.size());
			j++;
		}
	}

	/// \brief Store regression predictions as Data<LabelType>.
	///
	/// Call this function only on new runs. It fails for runs that were
	/// already committed to the server, and also for runs that were
	/// obtained from the server.
	template <typename LabelType>
	void setPredictions(std::size_t repetition, std::size_t fold, Data<LabelType> const& predictions)
	{
		if (id() != invalidID) throw SHARKEXCEPTION("Cannot set predictions for a committed run.");

		SHARK_ASSERT(repetition < m_task->repetitions());
		SHARK_ASSERT(fold < m_task->folds());

		std::vector<std::size_t> const& split = m_task->splitIndices(repetition);
		std::vector<double>& pred = m_predictions[repetition];
		SHARK_ASSERT(split.size() == pred.size());
		std::size_t j = 0;
		std::size_t batches = predictions.numberOfBatches();
		for (std::size_t b=0; b<batches; b++)
		{
			typename Data<LabelType>::const_batch_reference batch = predictions.batch(b);
			std::size_t size = boost::size(batch);
			for (std::size_t i=0; i<size; i++)
			{
				while (split[j] != fold) j++;
				typename Data<LabelType>::const_element_reference element = shark::get(batch, i);
				pred[j] = arff::detail::label2double<LabelType>(element);
				SHARK_ASSERT(j < pred.size());
				j++;
			}
		}
	}

	/// \brief Commit prediction results to the OpenML service.
	///
	/// \par
	/// Call this function only after filling in the values of all
	/// parameters and the predictions of all combinations of
	/// repetition and fold. It is not possible to commit an already
	/// committed run, or a run that was obtained from the server.
	void commit();

	/// \brief Obtain the underlying ARFF predictions file
	CachedFile const& predictionsfile() const
	{ return m_file; }

private:
	std::shared_ptr<Task> m_task;        ///< task associated with the run
	std::shared_ptr<Flow> m_flow;        ///< flow associated with the run
	std::vector< std::string > m_hyperparameterValue;     ///< values of all hyperparameters defined in the flow
	std::vector< std::vector< double > > m_predictions;   ///< predictions for all repetitions defined in the task

	CachedFile m_file;                   ///< ARFF file storing predictions
};


};  // namespace openML
};  // namespace shark
#endif
