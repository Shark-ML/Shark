//===========================================================================
/*!
 * 
 *
 * \brief  Entry point to the OpenML interface and definition of free functions.
 * 
 * 
 * \par
 * This file provides methods and classes for easy access to the OpenML
 * platform for open machine learning research.
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

#ifndef SHARK_OPENML_OPENML_H
#define SHARK_OPENML_OPENML_H


#include "Base.h"
#include "Connection.h"
#include "Dataset.h"
#include "Task.h"
#include "Flow.h"
#include "Run.h"


namespace shark {
namespace openML {


////////////////////////////////////////////////////////////
// create functions
//


// \brief Upload a new data set to OpenML.
//
// The function uploads a data set, which must exist in ARFF format in
// memory. ARFF is specified here:
//
//   https://www.cs.waikato.ac.nz/ml/weka/arff.html
//
// Minimal meta data like the data set name and a description must
// be provided.
//
// In case of failure, the function throws an exception.
//
// \param  arff         contents of an ARFF file in memory
// \param  name         name of the data set as it should appear in OpenML; should most often coincide with the @relation field in the ARFF file
// \param  description  human readable description of the data set
// \param  target       name of the default target attribute, can be empty
// \return  ID of the newly created data set
// 
IDType createDataset(
		std::string const& arff,
		std::string const& name,
		std::string const& description,
		std::string const& target = "");

// \brief Upload a new data set to OpenML.
//
// The function uploads a data set, which must exist as an ARFF file on
// disk. ARFF is specified here:
//
//   https://www.cs.waikato.ac.nz/ml/weka/arff.html
//
// Minimal meta data like the data set name and a description must
// be provided.
//
// In case of failure, the function throws an exception.
//
// \param  arffFile     path to a valid ARFF data set file
// \param  name         name of the data set as it should appear in OpenML; should most often coincide with the @relation field in the ARFF file
// \param  description  human readable description of the data set
// \param  target       name of the default target attribute, can be empty
// \return  ID of the newly created data set
// 
IDType createDataset(
		PathType const& arffFile,
		std::string const& name,
		std::string const& description,
		std::string const& target = "");

// \brief Create a new task in OpenML.
//
// The function creates a new task. A task specifies what to do with a
// data set, see https://www.openml.org/guide/bootcamp
// For the most common task types refer to the functions
// createSupervisedClassificationTask and createSupervisedRegressionTask,
// which come with good defaults.
// In case of failure, the function throws an exception.
//
// \param  type                 see queryTaskTypes() for details
// \param  name                 name of the task
// \param  ds                   data set underlying the task
// \param  target               target attribute, can be empty if the data set defines a default target attribute
// \param  estimationProcedure  see queryEstimationProcedures() for details; default is 10-fold cross-validation
// \return  ID of the newly created task
// 
IDType createTask(
		IDType type,
		std::shared_ptr<Dataset> ds,
		std::string const& target = "",
		IDType estimationProcedure = 1,
		std::string const& evaluationMeasure = "");

// \brief Create a new supervised classification task in OpenML.
//
// The function creates a new supervised classification task.
// Its name is "supervised classification on " + <data set name>.
// Predictive accuracy is defined as the primary evaluation measure.
// In case of failure, the function throws an exception.
//
// \param  ds                   data set underlying the task
// \param  target               target attribute, can be empty if the data set defines a default target attribute
// \param  estimationProcedure  see queryEstimationProcedures() for details; default is 10-fold cross-validation
// \return  ID of the newly created task
//
IDType createSupervisedClassificationTask(
		std::shared_ptr<Dataset> ds,
		std::string const& target = "",
		IDType estimationProcedure = 1);

// \brief Create a new supervised regression task in OpenML.
//
// The function creates a new supervised regression task.
// Its name is "supervised regression on " + <data set name>.
// Mean squared error is defined as the primary evaluation measure.
// In case of failure, the function throws an exception.
//
// \param  ds                   data set underlying the task
// \param  target               target attribute, can be empty if the data set defines a default target attribute
// \param  estimationProcedure  see queryEstimationProcedures() for details; default is 10-fold cross-validation
// \return  ID of the newly created task
// 
IDType createSupervisedRegressionTask(
		std::shared_ptr<Dataset> ds,
		std::string const& target = "",
		IDType estimationProcedure = 1);


////////////////////////////////////////////////////////////
// delete functions
//


void deleteDataset(IDType id);
void deleteDataset(std::shared_ptr<Dataset> dataset);

void deleteTask(IDType id);
void deleteTask(std::shared_ptr<Task> task);

void deleteFlow(IDType id);
void deleteFlow(std::shared_ptr<Flow> flow);

void deleteRun(IDType id);
void deleteRun(Run const& run);


////////////////////////////////////////////////////////////
// query functions
//


/// \brief Obtain data sets filtered by properties.
///
/// Filters follow the format /property1/value1/property2/value2 etc.
/// Values can be numbers, lists like '1,2,3' or ranges like '0..10'.
/// Possible properties are:
///
///  * limit: maximal number of data sets to return
///  * offset: index of the first data set to consider
///  * tag: only report data sets with the given tag
///  * status: possible values are 'active', 'deactivated', and 'in_preparation'
///  * data_name: only report data sets with the given name, e.g., 'iris'
///  * number_instances: filter on number of data points
///  * number_features: filter on number of features (data dimension)
///  * number_classes: filter on number of classes for classification problems
///  * number_missing_values: filter on number of missing feature values
///
SHARK_EXPORT_SYMBOL QueryResult queryDatasets(std::string const& filters);

/// \brief Obtain tasks filtered by properties.
///
/// Filters follow the format /property1/value1/property2/value2 etc.
/// Values can be numbers, lists like '1,2,3' or ranges like '0..10'.
/// Possible properties are:
///
///  * limit: maximal number of tasks to return
///  * offset: index of the first task to consider
///  * type: 1 for classification, 2 for regression; see also queryTaskTypes().
///  * status: possible values are 'active', 'deactivated', and 'in_preparation'
///  * tag: only report tasks with the given tag
///  * data_tag: only report tasks relating to a data set with the given tag
///  * data_id: only report tasks related to the given data set
///  * data_name: only report tasks related to data sets with the given name, e.g., 'iris'
///  * number_instances: filter on number of data points
///  * number_features: filter on number of features (data dimension)
///  * number_classes: filter on number of classes for classification problems
///  * number_missing_values: filter on number of missing feature values
///
SHARK_EXPORT_SYMBOL QueryResult queryTasks(std::string const& filters);

/// \brief Obtain flows filtered by properties.
///
/// Filters follow the format /property1/value1/property2/value2 etc.
/// Values can be numbers, lists like '1,2,3' or ranges like '0..10'.
/// Possible properties are:
///
///  * limit: maximal number of flows to return
///  * offset: index of the first flow to consider
///  * tag: only report flows with the given tag
///  * uploader: only report flows owned by the given uploader (ID or list of IDs)
///
SHARK_EXPORT_SYMBOL QueryResult queryFlows(std::string const& filters);

/// \brief Find a flow by name.
///
/// Obtain the id of a flow by name (and version).
/// If the flow does not exist yet then invalidID is returned.
///
SHARK_EXPORT_SYMBOL IDType findFlow(std::string const& name, std::string const& version = Flow::sharkVersion());

/// \brief Obtain runs filtered by properties.
///
/// Filters follow the format /property1/value1/property2/value2 etc.
/// Values can be numbers, lists like '1,2,3' or ranges like '0..10'.
/// Possible properties are:
///
///  * limit: maximal number of tasks to return
///  * offset: index of the first task to consider
///  * run: only report runs with the given id(s)
///  * tag: only report runs with the given tag
///  * task: only report runs referring to the given task(s)
///  * flow: only report runs referring to the given flow(s)
///  * setup: only report runs following the specified setup
///  * uploader: only report runs created by the given uploader(s)
///  * show_errors: somehow related to errors during offline post-processing of runs on the server...?
///
SHARK_EXPORT_SYMBOL QueryResult queryRuns(std::string const& filters);

/// \brief Obtain a list of all task types defined in OpenML.
///
/// The function returns textual descriptions linked to IDs. An example
/// of a task type is supervised classification.
SHARK_EXPORT_SYMBOL std::map<IDType, std::string> queryTaskTypes();

/// \brief Obtain a list of all estimation procedures defined in OpenML.
///
/// The function returns textual descriptions linked to IDs. An example
/// of an evaluation procedure is 10-fold cross-validation with
/// stratified sampling.
SHARK_EXPORT_SYMBOL std::map<IDType, std::string> queryEstimationProcedures();


};  // namespace openML
};  // namespace shark
#endif
