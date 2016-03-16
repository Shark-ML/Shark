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


#include "Connection.h"
#include "Dataset.h"
#include "Task.h"
#include "Flow.h"
#include "Run.h"


namespace shark {
namespace openML {


typedef std::vector< std::unique_ptr<Dataset> > DatasetList;     ///< \brief Collection of Dataset objects.
typedef std::vector< std::unique_ptr<Task>    > TaskList;        ///< \brief Collection of Task objects.
typedef std::vector< std::unique_ptr<Flow>    > FlowList;        ///< \brief Collection of Flow objects.
typedef std::vector< std::unique_ptr<Run>     > RunList;         ///< \brief Collection of Run objects.


/// \brief Obtain the list of all data sets from OpenML.
DatasetList datasets();

/// \brief Obtain the list of all tasks from OpenML.
TaskList tasks();

/// \brief Obtain the list of all flows from OpenML.
FlowList flows();

/// \brief Obtain the list of all runs from OpenML.
RunList runs();


std::vector<Dataset*> filter(std::vector<Dataset*> const& list, std::string const& conditions);
std::vector<Task*>    filter(std::vector<Task*>    const& list, std::string const& conditions);
std::vector<Flow*>    filter(std::vector<Flow*>    const& list, std::string const& conditions);
std::vector<Run*>     filter(std::vector<Run*>     const& list, std::string const& conditions);


};  // namespace openML
};  // namespace shark
#endif
