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

#include "PooledEntity.h"
#include "CachedFile.h"


namespace shark {
namespace openML {


/// \brief Representation of an OpenML data set.
SHARK_EXPORT_SYMBOL class Dataset : public PooledEntity<Dataset>, public CachedFile
{
private:
	friend class PooledEntity<Dataset>;

	/// \brief Construct an existing dataset from its ID.
	Dataset(IDType id, bool downloadData = true);

public:
	////////////////////////////////////////////////////////////
	// tagging
	//

	/// \brief Add a tag to the entity.
	void tag(std::string const& tagname);

	/// \brief Remove a tag from the entity.
	void untag(std::string const& tagname);

	////////////////////////////////////////////////////////////
	// printing
	//

	/// \brief Print a human readable summary of the entity.
	void print(std::ostream& os = std::cout) const;

	////////////////////////////////////////////////////////////
	// property getters
	//

	std::string const& name() const
	{ return m_name; }

	std::string const& description() const
	{ return m_description; }

	std::string const& defaultTargetFeature() const
	{ return m_defaultTargetFeature; }

	std::string const& format() const
	{ return m_format; }

	std::string const& status() const
	{ return m_status; }

	std::string const& uploadDate() const
	{ return m_uploadDate; }

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

private:
	// properties
	std::string m_name;
	std::string m_description;
	std::string m_defaultTargetFeature;
	std::string m_format;
	std::string m_licence;
	std::string m_status;
	std::string m_uploadDate;
	std::string m_version;
	std::string m_versionLabel;
	std::string m_visibility;

	// features
	std::vector<FeatureDescription> m_feature;
};


};  // namespace openML
};  // namespace shark
#endif
