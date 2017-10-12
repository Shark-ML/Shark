//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of an OpenML Flow.
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
#include <shark/OpenML/detail/Json.h>
#include <shark/Core/Shark.h>


namespace shark {
namespace openML {


Flow::Flow(IDType id)
: PooledEntity<Flow>(id)
{
	obtainFromServer();
}

Flow::Flow(std::string const& name, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties)
: PooledEntity<Flow>()
{
	std::string version = sharkVersion();

	// check whether the flow already exists
	SHARK_ASSERT(findFlow(name, version) == invalidID);

	// upload a new flow
	std::string xml = "<oml:flow xmlns:oml=\"http://openml.org/openml\">"
			"<oml:name>" + detail::xmlencode(name) + "</oml:name>"
			"<oml:external_version>" + detail::xmlencode(version) + "</oml:external_version>"
			"<oml:description>" + detail::xmlencode(description) + "</oml:description>"
			"<oml:language>English</oml:language>"
			"<oml:dependencies>Shark machine learning library</oml:dependencies>";
	for (std::size_t i=0; i<hyperparameters.size(); i++)
	{
		Hyperparameter const& p = hyperparameters[i];
		xml += "<oml:parameter>"
				"<oml:name>" + detail::xmlencode(p.name) + "</oml:name>"
				"<oml:data_type>" + detail::xmlencode(p.datatype) + "</oml:data_type>"
				"<oml:default_value>" + detail::xmlencode(p.defaultValue) + "</oml:default_value>"
				"<oml:description>" + detail::xmlencode(p.description) + "</oml:description>"
				"</oml:parameter>";
	}
	xml += "</oml:flow>";


	Connection::ParamType param;
	param.push_back(std::make_pair("description|application/xml", xml));
	detail::Json result = connection.post("/flow", param);
	IDType id = detail::json2number<IDType>(result["upload_flow"]["id"]);

	setID(id);

	// obtain the flow data back from the server
	obtainFromServer();
}

//static
std::string Flow::sharkVersion()
{
	return "shark_version_"
			+ boost::lexical_cast<std::string>(Shark::version_type::MAJOR())
			+ "."
			+ boost::lexical_cast<std::string>(Shark::version_type::MINOR())
			+ "."
			+ boost::lexical_cast<std::string>(Shark::version_type::PATCH());
}

void Flow::tag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("flow_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/flow/tag", param);
	Entity::tag(tagname);
}

void Flow::untag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("flow_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/flow/untag", param);
	Entity::untag(tagname);
}

void Flow::print(std::ostream& os) const
{
	os << "Flow:" << std::endl;
	Entity::print(os);
	os << " name: " << m_name << std::endl;
	os << " (external) version: " << m_version << std::endl;
	os << " description: " << m_description << std::endl;
	for (std::map<std::string, std::string>::const_iterator it = m_properties.begin(); it != m_properties.end(); ++it)
	{
		os << " " << it->first << ": " << it->second << std::endl;
	}
	for (std::size_t i=0; i<m_hyperparameter.size(); i++)
	{
		Hyperparameter const& p = m_hyperparameter[i];
		os << "  parameter " << i << ": " << p.name << "; " << p.description << " (" << p.datatype << ") default: " << p.defaultValue << std::endl;
	}
}

void Flow::obtainFromServer()
{
	detail::Json result = connection.get("/flow/" + boost::lexical_cast<std::string>(id()));

	detail::Json desc = result["flow"];
	detail::Json param = desc["parameter"];

	m_name = desc["name"].asString();
	m_version = desc["external_version"].asString();
	m_description = desc["description"].asString();

	if (desc.has("parameter"))
	{
		detail::Json param = desc["parameter"];
		if (param.isArray())
		{
			for (std::size_t i=0; i<param.size(); i++)
			{
				detail::Json jp = param[i];
				Hyperparameter p;
				p.name = jp["name"].asString();
				p.datatype = jp["data_type"].asString();
				if (jp.has("defaul_value") && jp["default_value"].isString()) p.defaultValue = jp["default_value"].asString();
				p.description = jp["description"].asString();
				m_hyperparameter.push_back(p);
			}
		}
	}

	// TODO: populate properties

	if (desc.has("tag")) setTags(desc["tag"]);
}


};  // namespace openML
};  // namespace shark
