//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of OpenML free functions.
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


#include <shark/OpenML/OpenML.h>
#include <shark/OpenML/detail/Tools.h>
#include <boost/lexical_cast.hpp>


namespace shark {
namespace openML {


////////////////////////////////////////////////////////////


IDList getIDs(std::string const& query, std::string const& level1, std::string const& level2, std::string const& idname)
{
	detail::Json result = connection.get(query);
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json objects = result[level1][level2];
	SHARK_ASSERT(objects.isArray());
	IDList ret(objects.size());
	for (std::size_t i=0; i<objects.size(); i++) ret[i] = detail::json2number<IDType>(objects[i][idname]);
	return ret;
}


IDList allDatasets()
{ return getIDs("/data/list", "data", "dataset", "did"); }

IDList taggedDatasets(std::string const& tagname)
{ return getIDs("/data/list/tag/" + detail::urlencode(tagname), "data", "dataset", "did"); }


IDList allTasks()
{ return getIDs("/task/list", "tasks", "task", "task_id"); }

IDList supervisedClassificationTasks()
{ return getIDs("/task/list/type/1", "tasks", "task", "task_id"); }

IDList supervisedRegressionTasks()
{ return getIDs("/task/list/type/2", "tasks", "task", "task_id"); }

IDList taggedTasks(std::string const& tagname)
{ return getIDs("/task/list/tag/" + detail::urlencode(tagname), "tasks", "task", "task_id"); }


IDList allFlows()
{ return getIDs("/flow/list", "flows", "flow", "id"); }

IDList taggedFlows(std::string const& tagname)
{ return getIDs("/flow/list/tag/" + detail::urlencode(tagname), "flows", "flow", "id"); }

IDList myFlows()
{
	detail::Json result = connection.get("/flow/owned");
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json ids = result["flow_owned"]["id"];
	SHARK_ASSERT(ids.isArray());
	IDList ret(ids.size());
	for (std::size_t i=0; i<ids.size(); i++) ret[i] = detail::json2number<IDType>(ids[i]);
	return ret;
}

IDType getFlow(std::string const& name, std::string const& version)
{
	detail::Json result = connection.get("/flow/exists/" + detail::urlencode(name) + "/" + detail::urlencode(version));
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json f = result["flow_exists"];
	if (detail::json2bool(f["exists"])) return detail::json2number<IDType>(f["id"]);
	else return invalidID;
}


IDList taggedRuns(std::string const& tagname)
{ return getIDs("/run/list/tag/" + detail::urlencode(tagname), "runs", "run", "run_id"); }

IDList runsByTask(IDType taskID)
{ return getIDs("/run/list/task/" + boost::lexical_cast<std::string>(taskID), "runs", "run", "run_id"); }

IDList runsByTask(Task const& task)
{ return getIDs("/run/list/task/" + boost::lexical_cast<std::string>(task.id()), "runs", "run", "run_id"); }

IDList runsByFlow(IDType flowID)
{ return getIDs("/run/list/flow/" + boost::lexical_cast<std::string>(flowID), "runs", "run", "run_id"); }

IDList runsByFlow(Flow const& flow)
{ return getIDs("/run/list/flow/" + boost::lexical_cast<std::string>(flow.id()), "runs", "run", "run_id"); }


};  // namespace openML
};  // namespace shark
