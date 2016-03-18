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

SHARK_EXPORT_SYMBOL IDList allDatasets();                                            ///< \brief Obtain the list of all data set IDs from OpenMP.
SHARK_EXPORT_SYMBOL IDList taggedDatasets(std::string const& tagname);               ///< \brief Obtain the list of all data set IDs with the given tag.

SHARK_EXPORT_SYMBOL IDList allTasks();                                               ///< \brief Obtain the list of all task IDs from OpenMP.
SHARK_EXPORT_SYMBOL IDList supervisedClassificationTasks();                          ///< \brief Obtain the list of all IDs of supervised classification tasks.
SHARK_EXPORT_SYMBOL IDList supervisedRegressionTasks();                              ///< \brief Obtain the list of all IDs of supervised regression tasks.
SHARK_EXPORT_SYMBOL IDList taggedTasks(std::string const& tagname);                  ///< \brief Obtain the list of all task IDs with the given tag.

SHARK_EXPORT_SYMBOL IDList allFlows();                                               ///< \brief Obtain the list of all flow IDs from OpenML.
SHARK_EXPORT_SYMBOL IDList taggedFlows(std::string const& tagname);                  ///< \brief Obtain the list of all flow IDs with the given tag.
SHARK_EXPORT_SYMBOL IDList myFlows();                                                ///< \brief Obtain the list of all flow IDs owned by the user.
SHARK_EXPORT_SYMBOL IDType getFlow(std::string const& name, std::string const& version);  ///< \brief Obtain the flow id, or invalidID if the flow does not exist.

SHARK_EXPORT_SYMBOL IDList taggedRuns(std::string const& tagname);                   ///< \brief Obtain the list of all run IDs with the given tag.
SHARK_EXPORT_SYMBOL IDList runsByTask(IDType taskID);                                ///< \brief Obtain the list of all run IDs associated with the given task.
SHARK_EXPORT_SYMBOL IDList runsByTask(Task const& task);                             ///< \brief Obtain the list of all run IDs associated with the given task.
SHARK_EXPORT_SYMBOL IDList runsByFlow(IDType flowID);                                ///< \brief Obtain the list of all run IDs associated with the given flow.
SHARK_EXPORT_SYMBOL IDList runsByFlow(Flow const& flow);                             ///< \brief Obtain the list of all run IDs associated with the given flow.


};  // namespace openML
};  // namespace shark
#endif
