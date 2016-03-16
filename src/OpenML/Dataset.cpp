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

#include <fstream>


namespace shark {
namespace openML {


static const char* featureTypeName[] = { "numeric", "nominal", "string", "date", "unknown" };


Dataset::Dataset(IDType id)
: Entity(id)
{
	{
		detail::Json result = connection.get("/data/" + boost::lexical_cast<std::string>(id));
		if (result.isNull() || result.isNumber()) throw SHARKEXCEPTION("failed to query OpenML data set");

		detail::Json desc = result["data_set_description"];
		setTags(desc["tag"]);

		m_name = desc["name"].asString();
		m_description = desc["description"].asString();
		m_defaultTargetAttribute = desc["default_target_attribute"].asString();
		m_format = desc["format"].asString();
		m_licence = desc["licence"].asString();
		m_status = desc["status"].asString();
		m_uploadDate = desc["upload_date"].asString();
		m_url = desc["url"].asString();
		m_version = desc["version"].asString();
		m_versionLabel = desc["version_label"].asString();
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
			if (type == "numeric") fd.type = NUMERIC;
			else if (type == "nominal") fd.type = NOMINAL;
			else if (type == "string") fd.type = STRING;
			else if (type == "date") fd.type = DATE;
			else fd.type = UNKNOWN;
			fd.name = feature["name"].asString();
			fd.target = detail::json2bool(feature["is_target"]);
			fd.ignore = detail::json2bool(feature["is_ignore"]);
			fd.rowIdentifier = detail::json2bool(feature["is_row_identifier"]);
			m_feature.push_back(fd);
		}
	}

	std::string filename = "dataset-" + boost::lexical_cast<std::string>(id);
	m_arffFile = connection.cacheDirectory() / filename;
}

bool Dataset::download()
{
printf("[download] m_arffFile=%s\n", m_arffFile.string().c_str());
	if (downloaded()) return true;

	std::string host;
	std::string resource;
	if (m_url.size() >= 7 && m_url.substr(0, 7) == "http://")
	{
		std::size_t slash = m_url.find('/', 7);
		if (slash == std::string::npos) return false;
		host = m_url.substr(7, slash - 7);
		resource = m_url.substr(slash);
	}
	else
	{
		std::size_t slash = m_url.find('/');
		if (slash == std::string::npos) return false;
		host = m_url.substr(0, slash);
		resource = m_url.substr(slash);
	}

	Connection conn(host);
	detail::HttpResponse response = conn.getHTTP(resource);
	if (response.statusCode() != 200) return false;

	std::ofstream os(m_arffFile.string());
	os << response.body();
	os.close();
	return true;
}

void Dataset::print(std::ostream& os) const
{
	os << "Dataset:" << std::endl;
	Entity::print(os);
	os << " name: " << m_name << std::endl;
//	os << " description: " << m_description << std::endl;
	os << " default target attribute: " << m_defaultTargetAttribute << std::endl;
	os << " format: " << m_format << std::endl;
	os << " license: " << m_licence << std::endl;
	os << " format: " << m_format << std::endl;
	os << " status: " << m_status << std::endl;
	os << " upload date: " << m_uploadDate << std::endl;
	os << " url: " << m_url << std::endl;
	os << " version: " << m_version << std::endl;
	os << " version label: " << m_versionLabel << std::endl;
	os << " visibility: " << m_visibility << std::endl;
	os << " file status: ";
	if (downloaded()) os << "in cache at " << m_arffFile.string();
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
