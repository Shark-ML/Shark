//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of an OpenML Dataset.
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
#include <shark/OpenML/detail/HttpResponse.h>
#include <shark/Core/Exception.h>

#include <boost/lexical_cast.hpp>

#include <fstream>


namespace shark {
namespace openML {


Dataset::Dataset(IDType id, bool downloadData)
: PooledEntity<Dataset>(id)
, CachedFile("", "dataset_" + boost::lexical_cast<std::string>(id) + ".arff")
{
	{
		detail::Json result = connection.get("/data/" + boost::lexical_cast<std::string>(id));
		if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("failed to query OpenML data set");

		detail::Json desc = result["data_set_description"];
		if (desc.has("tag")) setTags(desc["tag"]);

		m_name = desc["name"].asString();
		m_description = desc["description"].asString();
		m_defaultTargetFeature = desc["default_target_attribute"].asString();
		m_format = desc["format"].asString();
		m_licence = desc["licence"].asString();
		m_status = desc["status"].asString();
		m_url = desc["url"].asString();
		m_visibility = desc["visibility"].asString();
	}

	{
		detail::Json result = connection.get("/data/features/" + boost::lexical_cast<std::string>(id));
		if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("failed to query OpenML data set");

		detail::Json features = result["data_features"]["feature"];
		for (std::size_t i=0; i<features.size(); i++)
		{
			detail::Json feature = features[i];
			FeatureDescription fd;
			std::string type = feature["data_type"].asString();
			detail::ASCIItoLowerCase(type);
			if (type == "binary") fd.type = BINARY;
			else if (type == "integer") fd.type = INTEGER;
			else if (type == "numeric") fd.type = NUMERIC;
			else if (type == "nominal") fd.type = NOMINAL;
			else if (type == "string") fd.type = STRING;
			else if (type == "date") fd.type = DATE;
			else throw SHARKEXCEPTION("unknown feature type in dataset definition");
			fd.name = feature["name"].asString();
			fd.target = detail::json2bool(feature["is_target"]);
			fd.ignore = detail::json2bool(feature["is_ignore"]);
			fd.rowIdentifier = detail::json2bool(feature["is_row_identifier"]);
			m_feature.push_back(fd);
		}
	}

	if (downloadData)
	{
		if (! download()) throw SHARKEXCEPTION("failed to download OpenML data set");
	}
}


void Dataset::tag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("data_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/data/tag", param);
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	Entity::tag(tagname);
}

void Dataset::untag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("data_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/data/untag", param);
	if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("OpenML request failed");
	Entity::untag(tagname);
}

void Dataset::print(std::ostream& os) const
{
	os << "Dataset:" << std::endl;
	Entity::print(os);
	os << " name: " << m_name << std::endl;
//	os << " description: " << m_description << std::endl;
	os << " default target feature: " << m_defaultTargetFeature << std::endl;
	os << " format: " << m_format << std::endl;
	os << " license: " << m_licence << std::endl;
	os << " status: " << m_status << std::endl;
	os << " url: " << url() << std::endl;
	os << " visibility: " << m_visibility << std::endl;
	os << " file status: ";
	if (downloaded()) os << "in cache at " << filename().string();
	else os << "not in cache";
	os << std::endl;
	os << " " << m_feature.size() << " features:" << std::endl;
	for (std::size_t i=0; i<m_feature.size(); i++)
	{
		FeatureDescription const& fd = m_feature[i];
		os << "  feature " << i << ": " << fd.name << " (" << featureTypeName[(unsigned int)fd.type] << ")";
		if (fd.target) os << " [target]";
		if (fd.ignore) os << " [ignore]";
		if (fd.rowIdentifier) os << " [row-identifier]";
		os << std::endl;
	}
}


};  // namespace openML
};  // namespace shark
