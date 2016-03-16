//===========================================================================
/*!
 * 
 *
 * \brief       Definition of an OpenML Dataset.
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

#ifndef SHARK_OPENML_DATASET_H
#define SHARK_OPENML_DATASET_H

#include "Entity.h"


namespace shark {
namespace openML {


enum FeatureType
{
	NUMERIC = 0,
	NOMINAL = 1,
	STRING = 2,
	DATE = 3,
	UNKNOWN = 4,
};


struct FeatureDescription
{
	FeatureType type;
	std::string name;
	bool target;
	bool ignore;
	bool rowIdentifier;
};


/// \brief Representation of an OpenML data set.
class Dataset : public Entity
{
public:
	Dataset(IDType id);

	////////////////////////////////////////////////////////////
	// property getters
	//

	std::string const& name() const
	{ return m_name; }

	std::string const& description() const
	{ return m_description; }

	std::string const& defaultTargetAttribute() const
	{ return m_defaultTargetAttribute; }

	std::string const& labelname() const
	{ return m_defaultTargetAttribute; }

	std::string const& format() const
	{ return m_format; }

	std::string const& status() const
	{ return m_status; }

	std::string const& uploadDate() const
	{ return m_uploadDate; }

	std::string const& url() const
	{ return m_url; }

	std::string const& version() const
	{ return m_version; }

	std::string const& versionLabel() const
	{ return m_versionLabel; }

	std::string const& visibility() const
	{ return m_visibility; }

	std::size_t numberOfFeatures() const
	{ return m_feature.size(); }

	FeatureDescription const& feature(std::size_t index) const
	{ return m_feature[index]; }

//	double namedProperty(std::string const& property) const;

	////////////////////////////////////////////////////////////
	// file cache
	//

	bool download();

	bool downloaded() const
	{ return boost::filesystem::exists(m_arffFile); }

	PathType const& filename() const
	{ return m_arffFile; }

	////////////////////////////////////////////////////////////
	// dump information
	//
	void print(std::ostream& os = std::cout) const;

private:
	// properties
	std::string m_name;
	std::string m_description;
	std::string m_defaultTargetAttribute;
	std::string m_format;
	std::string m_licence;
	std::string m_status;
	std::string m_uploadDate;
	std::string m_url;
	std::string m_version;
	std::string m_versionLabel;
	std::string m_visibility;

	// features
	std::vector<FeatureDescription> m_feature;

	// path to cached ARFF file
	PathType m_arffFile;
};


};  // namespace openML
};  // namespace shark
#endif
