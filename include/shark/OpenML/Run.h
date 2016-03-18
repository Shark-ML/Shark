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
#include "Task.h"
#include "Flow.h"
#include <shark/Data/Dataset.h>
#include <vector>


namespace shark {
namespace openML {


/// \brief Representation of an OpenML run.
SHARK_EXPORT_SYMBOL class Run : public Entity
{
public:
	/// \brief Construct an existing OpenML run from its ID.
	Run(IDType id);

	/// \brief Construct a new OpenML run from task and flow.
	///
	/// The run is created in memory, but not yet in the OpenML service.
	/// Call Run::commit to complete the process.
	Run(std::shared_ptr<Task> task, std::shared_ptr<Flow> flow)
	: Entity()
	, m_task(task)
	, m_flow(flow)
	, m_hyperparameterValue(m_flow->numberOfHyperparameters())
	, m_predictions(m_task->repetitions(), std::vector< std::vector<double> >(m_task->folds()))
	{ }

	/// \brief Add a tag to the entity.
	void tag(std::string const& tagname);

	/// \brief Remove a tag from the entity.
	void untag(std::string const& tagname);

	/// \brief Print a human readable summary of the entity.
	void print(std::ostream& os = std::cout) const;

	template <typename ValueType>
	void setHyperparameterValue(std::string const& name, ValueType const& value)
	{
		// find the index
		std::size_t num = m_flow->numberOfHyperparameters();
		for (std::size_t i=0; i<num; i++)
		{
			Hyperparameter const& p = m_flow->hyperparameter(i);
			if (p.name == name)
			{
				// store the value
				m_hyperparameterValue[i] = boost::lexical_cast<std::string>(value);
				return;
			}
		}
		throw SHARKEXCEPTION("unknown hyperparameter " + name);
	}

	/// \brief Store predictions.
	template <typename ContainerType>
	void storePredictions(std::size_t repetition, std::size_t fold, ContainerType const& predictions)
	{
		SHARK_ASSERT(repetition < m_task->repetitions());
		SHARK_ASSERT(fold < m_task->folds());
		std::vector<double>& p = m_predictions[repetition][fold];
		p.resize(predictions.size());
		for (std::size_t i=0; i<predictions.size(); i++) p[i] = predictions[i];
	}

	/// \brief Store regression predictions as Data<VectorType>.
	template <typename VectorType>
	void storePredictions(std::size_t repetition, std::size_t fold, Data<VectorType> const& predictions)
	{
		SHARK_ASSERT(repetition < m_task->repetitions());
		SHARK_ASSERT(fold < m_task->folds());
		std::vector<double>& p = m_predictions[repetition][fold];
		p.clear();
		std::size_t batches = predictions.numberOfBatches();
		for (std::size_t b=0; b<batches; b++)
		{
			typename Data<VectorType>::const_batch_reference batch = predictions.batch(b);
			std::size_t size = boost::size(batch);
			for (std::size_t i=0; i<size; i++)
			{
				typename Data<VectorType>::const_element_reference element = shark::get(batch, i);
				p.push_back(element(0));
			}
		}
	}

	/// \brief Store classification predictions as Data<unsigned int>.
	inline void storePredictions(std::size_t repetition, std::size_t fold, Data<unsigned int> const& predictions)
	{
		SHARK_ASSERT(repetition < m_task->repetitions());
		SHARK_ASSERT(fold < m_task->folds());
		std::vector<double>& p = m_predictions[repetition][fold];
		p.clear();
		std::size_t batches = predictions.numberOfBatches();
		for (std::size_t b=0; b<batches; b++)
		{
			Data<unsigned int>::const_batch_reference batch = predictions.batch(b);
			std::size_t size = boost::size(batch);
			for (std::size_t i=0; i<size; i++)
			{
				Data<unsigned int>::const_element_reference element = shark::get(batch, i);
				p.push_back(element);
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

private:
	std::shared_ptr<Task> m_task;
	std::shared_ptr<Flow> m_flow;
	std::vector< std::string > m_hyperparameterValue;
	std::vector< std::vector< std::vector< double > > > m_predictions;
};


};  // namespace openML
};  // namespace shark
#endif
