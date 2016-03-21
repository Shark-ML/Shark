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
#include <boost/algorithm/string/trim.hpp>

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
				fillTaggedValueArray(entry, detail::Json j = it->second);
			}
			else if (it->first == "input")
			{
				fillTaggedValueArray(entry, detail::Json j = it->second);
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

QueryResult allDatasets()
{
	return getIDs("/data/list", "data", "dataset", "did");
}

QueryResult taggedDatasets(std::string const& tagname)
{
	return getIDs("/data/list/tag/" + detail::urlencode(tagname), "data", "dataset", "did");
}

QueryResult allTasks()
{
	return getIDs("/task/list", "tasks", "task", "task_id");
}

QueryResult supervisedClassificationTasks()
{
	return getIDs("/task/list/type/1", "tasks", "task", "task_id");
}

QueryResult supervisedRegressionTasks()
{
	return getIDs("/task/list/type/2", "tasks", "task", "task_id");
}

QueryResult taggedTasks(std::string const& tagname)
{
	return getIDs("/task/list/tag/" + detail::urlencode(tagname), "tasks", "task", "task_id");
}


QueryResult allFlows()
{
	return getIDs("/flow/list", "flows", "flow", "id");
}

QueryResult taggedFlows(std::string const& tagname)
{
	return getIDs("/flow/list/tag/" + detail::urlencode(tagname), "flows", "flow", "id");
}

QueryResult myFlows()
{
	detail::Json result = connection.get("/flow/owned");
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	detail::Json ids = result["flow_owned"]["id"];
	SHARK_ASSERT(ids.isArray());
	QueryResult ret(ids.size());
	for (std::size_t i=0; i<ids.size(); i++)
	{
		ret[i].id = detail::json2number<IDType>(ids[i]);
		// TODO: fill in qualities and tags!
	}
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


QueryResult taggedRuns(std::string const& tagname)
{
	return getIDs("/run/list/tag/" + detail::urlencode(tagname), "runs", "run", "run_id");
}

QueryResult runsByTask(IDType taskID)
{
	return getIDs("/run/list/task/" + boost::lexical_cast<std::string>(taskID), "runs", "run", "run_id");
}

QueryResult runsByTask(Task const& task)
{
	return getIDs("/run/list/task/" + boost::lexical_cast<std::string>(task.id()), "runs", "run", "run_id");
}

QueryResult runsByFlow(IDType flowID)
{
	return getIDs("/run/list/flow/" + boost::lexical_cast<std::string>(flowID), "runs", "run", "run_id");
}

QueryResult runsByFlow(Flow const& flow)
{
	return getIDs("/run/list/flow/" + boost::lexical_cast<std::string>(flow.id()), "runs", "run", "run_id");
}


struct Condition
{
	Condition(std::string const& name_, std::string const& op_, std::string const& value_)
	: name(boost::trim_copy(name_))
	, op(op_)
	, value(boost::trim_copy(value_))
	{ }

	bool operator () (QueryEntry const& entry) const
	{
		if (name == "tag" && op == "tag")
		{
			// check whether value is present as a tag
			std::set<std::string>::const_iterator it = entry.tag.find(value);
			return (it != entry.tag.end());
		}
		else
		{
			// check a condition on a property
			std::map<std::string, std::string>::const_iterator it = entry.property.find(name);
			if (it == entry.property.end()) return false;

			try
			{
				// compare as numbers
		 		double lhs = boost::lexical_cast<double>(value);
		 		double rhs = boost::lexical_cast<double>(it->second);
				if (op == "==") return (lhs == rhs);
				if (op == "!=") return (lhs != rhs);
				if (op == "<") return (lhs < rhs);
				if (op == "<=") return (lhs <= rhs);
				if (op == ">") return (lhs > rhs);
				if (op == ">=") return (lhs >= rhs);
				throw SHARKEXCEPTION("[filter] invalid operator");
			}
			catch (...)
			{
				// compare as strings
				if (op == "==") return (value == it->second);
				if (op == "!=") return (value != it->second);
				throw SHARKEXCEPTION("[filter] invalid operator, non-numerical values cannot be ordered");
			}
		}
	}

	std::string name;
	std::string op;
	std::string value;
};

SHARK_EXPORT_SYMBOL QueryResult filter(QueryResult const& list, std::string const& strCondition)
{
	// parse condition string
	std::vector<Condition> conditions;
	std::size_t start = 0;
	while (start >= strCondition.size())
	{
		std::size_t semicolon = strCondition.find(';', start);
		if (semicolon == std::string::npos) semicolon = strCondition.size();
		if (conditions.size() >= start + 10 && strCondition.substr(start, 10) == "tagged as ")
		{
			conditions.push_back(Condition("tag", "tag", strCondition.substr(start + 10, semicolon - start - 10)));
		}
		else
		{
			std::size_t pos = strCondition.find_first_of("<=>!", start);
			std::size_t end = pos + 1;
			if (strCondition[end] == '=') end++;
			std::string strOP = strCondition.substr(pos, end - pos);
			if (strOP != "==" && strOP != "!=" && strOP != "<" && strOP != "<=" && strOP != ">" && strOP != ">=") throw SHARKEXCEPTION("[filter] invalid comparison operator: " + strOP);
			conditions.push_back(Condition(strCondition.substr(start, pos - start), strOP, strCondition.substr(end, semicolon - end)));
		}
		start = semicolon + 1;
	}

	// apply conditions to list
	QueryResult ret;
	for (std::size_t i=0; i<list.size(); i++)
	{
		bool add = true;
		for (std::size_t j=0; j<conditions.size(); j++)
		{
			if (! conditions[j](list[i]))
			{
				add = false;
				break;
			}
		}
		if (add) ret.push_back(list[i]);
	}

	return ret;
}


};  // namespace openML
};  // namespace shark
