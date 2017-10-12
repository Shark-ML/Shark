//===========================================================================
/*!
 * 
 *
 * \brief       Definition of an OpenML Task.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016-2017
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

#ifndef SHARK_OPENML_TASK_H
#define SHARK_OPENML_TASK_H


#include "PooledEntity.h"
#include "CachedFile.h"
#include "Dataset.h"
#include <shark/Data/Arff.h>
#include <shark/Data/CVDatasetTools.h>

#include <vector>
#include <memory>


namespace shark {
namespace openML {


class Dataset;


/// \brief Representation of an OpenML task.
SHARK_EXPORT_SYMBOL class Task : public PooledEntity<Task>
{
private:
	friend class PooledEntity<Task>;

	/// \brief Construct an existing OpenML task from its ID.
	Task(IDType id, bool downloadSplits = false);

public:
	/// \brief Add a tag to the entity.
	void tag(std::string const& tagname) override;

	/// \brief Remove a tag from the entity.
	void untag(std::string const& tagname) override;

	/// \brief Print a human readable summary of the entity.
	void print(std::ostream& os = std::cout) const override;

	/// \brief Load the splits into memory.
	void load() const;

	/// \brief Obtain the name of the task.
	std::string name() const
	{ return m_name; }

	/// \brief Obtain the type of the task.
	std::string type() const;

	/// \brief Obtain the underlying data set.
	std::shared_ptr<Dataset> dataset()
	{ return m_dataset; }

	/// \brief Obtain the underlying data set.
	std::shared_ptr<const Dataset> dataset() const
	{ return m_dataset; }

	/// \brief Obtain the name of the target attribute to be predicted.
	std::string const& targetAttribute() const
	{ return m_targetAttribute; }

	/// \brief Obtain the number of (cross validation) repetitions.
	std::size_t repetitions() const
	{ return m_repetitions; }

	/// \brief Obtain the number of cross validation folds.
	std::size_t folds() const
	{ return m_folds; }

	/// \brief Load the data set of the task into a LabeledData container.
	template <typename InputT, typename LabelT>
	void loadData(LabeledData<InputT, LabelT>& data)
	{
		CachedFile const& f = m_dataset->datafile();
		f.download();
		importARFF(f.filename().string(), m_targetAttribute, data);
	}

	/// \brief Obtain the assignment of data points to folds corresponding to a repetition.
	std::vector<std::size_t> const& splitIndices(std::size_t repetition) const
	{
		load();
		return m_split[repetition];
	}

	/// \brief Obtain the data split corresponding to a repetition.
	///
	/// \par
	/// NOTE: This function modifies the internal batch structure of the
	/// data set. Hence, calling the same function with a different
	/// repetition index invalidates previously obtained CVFolds objects.
	template <typename InputT, typename LabelT>
	CVFolds< LabeledData<InputT, LabelT> > split(std::size_t repetition, LabeledData<InputT, LabelT>& data) const
	{
		return createCVIndexed(data, m_folds, splitIndices(repetition));
	}

	/// \brief Obtain the underlying ARFF splits file
	CachedFile const& splitsfile() const
	{ return m_file; }

private:
	IDType m_type;                                     ///< task type ID

	// free meta data
	std::string m_name;                                ///< task name

	// data set spec
	std::shared_ptr<Dataset> m_dataset;                ///< associated data set
	std::string m_targetAttribute;                     ///< attribute of the data set acting as the label to be predicted

	// estimation procedure
	std::size_t m_repetitions;                         ///< number of independent repetitions of cross validation
	std::size_t m_folds;                               ///< number of cross validation folds
	std::vector< std::vector<std::size_t> > m_split;   ///< assignment of points to fold-wise test subsets in the form m_split[repetition][index] = fold

	// expected output
	std::string m_outputFormat;                        ///< file format for results upload
	std::vector<AttributeDescription> m_outputAttribute;   ///< attribute encoding for results upload

	// external file
	CachedFile m_file;                                 ///< ARFF file defining the data splits
};


};  // namespace openML
};  // namespace shark
#endif
