//===========================================================================
/*!
 * 
 *
 * \brief       Entry point to the OpenML interface.
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
// query functions for data sets, tasks, flows, and runs
//

SHARK_EXPORT_SYMBOL QueryResult allDatasets();                               ///< \brief Obtain the list of all data sets from OpenMP.
SHARK_EXPORT_SYMBOL QueryResult taggedDatasets(std::string const& tagname);  ///< \brief Obtain the list of all data sets with the given tag.

SHARK_EXPORT_SYMBOL QueryResult allTasks();                                  ///< \brief Obtain the list of all taska from OpenMP.
SHARK_EXPORT_SYMBOL QueryResult supervisedClassificationTasks();             ///< \brief Obtain the list of all supervised classification tasks.
SHARK_EXPORT_SYMBOL QueryResult supervisedRegressionTasks();                 ///< \brief Obtain the list of all supervised regression tasks.
SHARK_EXPORT_SYMBOL QueryResult taggedTasks(std::string const& tagname);     ///< \brief Obtain the list of all task with the given tag.

SHARK_EXPORT_SYMBOL QueryResult allFlows();                                  ///< \brief Obtain the list of all flow from OpenML.
SHARK_EXPORT_SYMBOL QueryResult taggedFlows(std::string const& tagname);     ///< \brief Obtain the list of all flow with the given tag.
SHARK_EXPORT_SYMBOL QueryResult myFlows();                                   ///< \brief Obtain the list of all flow owned by the user.
SHARK_EXPORT_SYMBOL IDType getFlow(std::string const& name, std::string const& version);                ///< \brief Obtain the flow id, or invalidID if the flow does not exist.

SHARK_EXPORT_SYMBOL QueryResult taggedRuns(std::string const& tagname);      ///< \brief Obtain the list of all run with the given tag.
SHARK_EXPORT_SYMBOL QueryResult runsByTask(IDType taskID);                   ///< \brief Obtain the list of all run associated with the given task.
SHARK_EXPORT_SYMBOL QueryResult runsByTask(Task const& task);                ///< \brief Obtain the list of all run associated with the given task.
SHARK_EXPORT_SYMBOL QueryResult runsByFlow(IDType flowID);                   ///< \brief Obtain the list of all run associated with the given flow.
SHARK_EXPORT_SYMBOL QueryResult runsByFlow(Flow const& flow);                ///< \brief Obtain the list of all run associated with the given flow.

/// \brief Filter an existing query result set by applying a set of conditions.
///
/// The parameter strCondition holds a semicolon-separated list of conditions.
/// There are two types of conditions:
///  1. check property, syntax: <property> <operator> <value>
///  2. check presence of a tag, syntax: tagged as <tagname>
/// valid operators are == != < <= > >=
/// names and values can be strings or numbers.
SHARK_EXPORT_SYMBOL QueryResult filter(QueryResult const& list, std::string const& strCondition);

};  // namespace openML
};  // namespace shark
#endif
