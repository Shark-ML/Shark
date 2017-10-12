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


#include <shark/OpenML/OpenML.h>
#include <shark/OpenML/detail/Tools.h>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <string>
#include <sstream>
#include <istream>


namespace shark {
namespace openML {


////////////////////////////////////////////////////////////


void fillTaggedValueArray(QueryEntry& entry, detail::Json json)
{
	SHARK_ASSERT(json.isArray());
	for (std::size_t i=0; i<json.size(); i++)
	{
		std::string name = detail::json2string(json[i]["name"]);
		std::string value = detail::json2string(json[i]["value"]);
	}
}

void fillProperties(QueryEntry& entry, detail::Json json)
{
	if (json.isArray())
	{
		for (std::size_t i=0; i<json.size(); i++) fillProperties(entry, json[i]);
	}
	else if (json.isObject())
	{
		for (detail::Json::const_object_iterator it=json.object_begin(); it != json.object_end(); ++it)
		{
			if (it->first == "tag")
			{
				detail::Json j = it->second;
				for (std::size_t i=0; i<j.size(); i++) entry.tag.insert(j[i].asString());
			}
			else if (it->first == "quality")
			{
				fillTaggedValueArray(entry, it->second);
			}
			else if (it->first == "input")
			{
				fillTaggedValueArray(entry, it->second);
			}
			else
			{
				detail::Json j = it->second;
				if (j.isString()) entry.property[it->first] = it->second.asString();
				else if (j.isNumber()) entry.property[it->first] = boost::lexical_cast<std::string>(it->second.asNumber());
			}
		}
	}
}


////////////////////////////////////////////////////////////


IDType createDataset(
		std::string const& arff,
		std::string const& name,
		std::string const& description,
		std::string const& target)
{
	// create the XML description
	std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
	xml += "<oml:data_set_description xmlns:oml=\"http://openml.org/openml\">\n"
			"<oml:name>" + detail::xmlencode(name) + "</oml:name>\n"
			"<oml:description>" + detail::xmlencode(description) + "</oml:description>\n"
			"<oml:format>arff</oml:format>\n";
	if (! target.empty()) xml += "<oml:default_target_attribute>" + detail::xmlencode(target) + "</oml:default_target_attribute>\n";
	xml += "</oml:data_set_description>\n";

	// upload the data to the OpenML server
	Connection::ParamType upload {
				{"description|application/xml|description.xml", xml},
				{"dataset|application/octet-stream|dataset.arff", arff}
			};
	openML::detail::Json result = connection.post("/data", upload);

	// check the result
	SHARK_ASSERT(result.isObject());
	SHARK_ASSERT(result.has("upload_data_set"));
	SHARK_ASSERT(result["upload_data_set"].isObject());
	SHARK_ASSERT(result["upload_data_set"].has("id"));

	// return the ID
	return detail::json2number<IDType>(result["upload_data_set"]["id"]);
}

IDType createDataset(
		PathType const& arffFile,
		std::string const& name,
		std::string const& description,
		std::string const& target)
{
	// load the ARFF file
	std::ifstream istr(arffFile.string());
	std::stringstream arff;
	arff << istr.rdbuf();

	// call the in-memory version
	return createDataset(arff.str(), name, description, target);
}


IDType createTask(
		IDType type,
		std::shared_ptr<Dataset> ds,
		std::string const& target,
		IDType estimationProcedure,
		std::string const& evaluationMeasure)
{
	std::string t = target;
	if (t.empty())
	{
		int index = ds->defaultTargetAttribute();
		if (index < 0) throw SHARKEXCEPTION("[createTask] target attribute must be specified, since the data set does not define one");
		t = ds->attribute(index).name;
	}

	// create the XML description
	std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
	xml += "<oml:task_inputs xmlns:oml=\"http://openml.org/openml\">\n"
			"<oml:task_type_id>" + boost::lexical_cast<std::string>(type) + "</oml:task_type_id>\n"
			"<oml:input name=\"source_data\">" + boost::lexical_cast<std::string>(ds->id()) + "</oml:input>\n"
			"<oml:input name=\"target_feature\">" + detail::xmlencode(t) + "</oml:input>\n"
			"<oml:input name=\"estimation_procedure\">" + boost::lexical_cast<std::string>(estimationProcedure) + "</oml:input>\n";
	if (! evaluationMeasure.empty()) xml += "<oml:input name=\"evaluation_measures\">" + detail::xmlencode(evaluationMeasure) + "</oml:input>\n";
	xml += "</oml:task_inputs>\n";

	// upload the data to the OpenML server
	Connection::ParamType upload {
				{"description|application/xml|description.xml", xml},
			};
	openML::detail::Json result = connection.post("/task", upload);

	// check the result
	SHARK_ASSERT(result.isObject());
	SHARK_ASSERT(result.has("upload_task"));
	SHARK_ASSERT(result["upload_task"].isObject());
	SHARK_ASSERT(result["upload_task"].has("id"));

	// return the ID
	return detail::json2number<IDType>(result["upload_task"]["id"]);
}

IDType createSupervisedClassificationTask(
		std::shared_ptr<Dataset> ds,
		std::string const& target,
		IDType estimationProcedure)
{
	IDType type = detail::inverseLookup(queryTaskTypes(), std::string("Supervised Classification"), invalidID);
	if (type == invalidID) throw SHARKEXCEPTION("[createSupervisedClassificationTask] task type 'Supervised Classification' unknown to OpenML");
	return createTask(type, ds, target, estimationProcedure);
}

IDType createSupervisedRegressionTask(
		std::shared_ptr<Dataset> ds,
		std::string const& target,
		IDType estimationProcedure)
{
	IDType type = detail::inverseLookup(queryTaskTypes(), std::string("Supervised Regression"), invalidID);
	if (type == invalidID) throw SHARKEXCEPTION("[createSupervisedRegressionTask] task type 'Supervised Regression' unknown to OpenML");
	return createTask(type, ds, target, estimationProcedure);
}


////////////////////////////////////////////////////////////


void deleteDataset(IDType id)
{
	detail::Json j = connection.del("/data/" + boost::lexical_cast<std::string>(id));

	// check the return value
	SHARK_ASSERT(j.isObject());
	SHARK_ASSERT(j.has("data_delete"));
	SHARK_ASSERT(j["data_delete"].has("id"));
	SHARK_ASSERT(detail::json2number<IDType>(j["data_delete"]["id"]) == id);
}

void deleteDataset(std::shared_ptr<Dataset> dataset)
{ deleteDataset(dataset->id()); }


void deleteTask(IDType id)
{
	detail::Json j = connection.del("/task/" + boost::lexical_cast<std::string>(id));

	// check the return value
	SHARK_ASSERT(j.isObject());
	SHARK_ASSERT(j.has("task_delete"));
	SHARK_ASSERT(j["task_delete"].has("id"));
	SHARK_ASSERT(detail::json2number<IDType>(j["task_delete"]["id"]) == id);
}

void deleteTask(std::shared_ptr<Task> task)
{ deleteTask(task->id()); }


void deleteFlow(IDType id)
{
	detail::Json j = connection.del("/flow/" + boost::lexical_cast<std::string>(id));

	// check the return value
	SHARK_ASSERT(j.isObject());
	SHARK_ASSERT(j.has("flow_delete"));
	SHARK_ASSERT(j["flow_delete"].has("id"));
	SHARK_ASSERT(detail::json2number<IDType>(j["flow_delete"]["id"]) == id);
}

void deleteFlow(std::shared_ptr<Flow> flow)
{ deleteFlow(flow->id()); }


void deleteRun(IDType id)
{
	detail::Json j = connection.del("/run/" + boost::lexical_cast<std::string>(id));

	// check the return value
	SHARK_ASSERT(j.isObject());
	SHARK_ASSERT(j.has("run_delete"));
	SHARK_ASSERT(j["run_delete"].has("id"));
	SHARK_ASSERT(detail::json2number<IDType>(j["run_delete"]["id"]) == id);
}

void deleteRun(Run const& run)
{ deleteRun(run.id()); }


////////////////////////////////////////////////////////////


QueryResult getIDs(std::string const& query, std::string const& level1, std::string const& level2, std::string const& idname)
{
	detail::Json result = connection.get(query);
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json objects = result[level1][level2];
	SHARK_ASSERT(objects.isArray());
	QueryResult ret(objects.size());
	for (std::size_t i=0; i<objects.size(); i++)
	{
		ret[i].id = detail::json2number<IDType>(objects[i][idname]);
		fillProperties(ret[i], objects[i]);
	}
	return ret;
}

QueryResult queryDatasets(std::string const& filters)
{
	return getIDs("/data/list" + filters, "data", "dataset", "did");
}

QueryResult queryTasks(std::string const& filters)
{
	return getIDs("/task/list" + filters, "tasks", "task", "task_id");
}

QueryResult queryFlows(std::string const& filters)
{
	return getIDs("/flow/list" + filters, "flows", "flow", "id");
}

IDType findFlow(std::string const& name, std::string const& version)
{
	std::string url = "/flow/exists/" + detail::urlencode(name) + "/" + detail::urlencode(version);
	detail::Json result = connection.get(url);
	detail::Json f = result["flow_exists"];
	if (detail::json2bool(f["exists"])) return detail::json2number<IDType>(f["id"]);
	else return invalidID;
}

QueryResult queryRuns(std::string const& filters)
{
	return getIDs("/run/list" + filters, "runs", "run", "run_id");
}

std::map<IDType, std::string> queryTaskTypes()
{
	detail::Json result = connection.get("/tasktype/list");
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json objects = result["task_types"]["task_type"];
	SHARK_ASSERT(objects.isArray());
	std::map<IDType, std::string> ret;
	for (std::size_t i=0; i<objects.size(); i++)
	{
		ret[detail::json2number<IDType>(objects[i]["id"])] = objects[i]["name"].asString();
	}
	return ret;
}

std::map<IDType, std::string> queryEstimationProcedures()
{
	detail::Json result = connection.get("/evaluationprocedure/list");
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json objects = result["estimationprocedures"]["estimationprocedure"];
	SHARK_ASSERT(objects.isArray());
	std::map<IDType, std::string> ret;
	for (std::size_t i=0; i<objects.size(); i++)
	{
		ret[(int)objects[i]["id"].asNumber()] = objects[i]["name"].asString();
	}
	return ret;
}


};  // namespace openML
};  // namespace shark
