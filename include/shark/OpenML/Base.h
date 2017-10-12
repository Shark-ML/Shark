//===========================================================================
/*!
 * 
 *
 * \brief       This file collects basic definitions in a central place.
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

#ifndef SHARK_OPENML_BASE_H
#define SHARK_OPENML_BASE_H


#include <shark/Core/DLLSupport.h>

#include <boost/filesystem.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <map>


namespace shark {
namespace openML {


SHARK_EXPORT_SYMBOL typedef boost::filesystem::path PathType;    ///< \brief Path type, e.g., for specifying the cache directory.
SHARK_EXPORT_SYMBOL typedef std::uint64_t IDType;                ///< \brief An ID is an unsigned integer.

SHARK_EXPORT_SYMBOL static const IDType invalidID = 0;           ///< \brief Invalid ID, marker for default constructed objects.
SHARK_EXPORT_SYMBOL static const double invalidValue = -1e100;   ///< \brief Invalid value, e.g., undefined prediction


/// \brief Result of querying OpenML objects.
SHARK_EXPORT_SYMBOL struct QueryEntry
{
	QueryEntry()
	: id(invalidID)
	{ }

	IDType id;
	std::map<std::string, std::string> property;
	std::set<std::string> tag;
};

SHARK_EXPORT_SYMBOL typedef std::vector<QueryEntry> QueryResult; ///< \brief Collection of descriptions of entities of unspecified type.


/// \brief Attribute types known to OpenML (and ARFF).
SHARK_EXPORT_SYMBOL enum AttributeType
{
	BINARY = 0,
	INTEGER = 1,
	NUMERIC = 2,
	NOMINAL = 3,
	STRING = 4,
	DATE = 5,
};


/// \brief Names of attribute types known to OpenML (and ARFF).
static const char* attributeTypeName[] = { "binary", "integer", "numeric", "nominal", "string", "date" };


/// \brief Meta data for an OpenML attribute.
SHARK_EXPORT_SYMBOL struct AttributeDescription
{
	AttributeType type;
	std::string name;
	bool target;
	bool ignore;
	bool rowIdentifier;
};


/// \brief Meta data for an OpenML parameter as found in flows and runs.
SHARK_EXPORT_SYMBOL struct Hyperparameter
{
	Hyperparameter()
	: defaultValue("none")
	{ }

	Hyperparameter(std::string const& name_, std::string const& description_, std::string const& datatype_, std::string const& defaultValue_ = "none")
	: name(name_)
	, description(description_)
	, datatype(datatype_)
	, defaultValue(defaultValue_)
	{ }

	std::string name;
	std::string description;
	std::string datatype;
	std::string defaultValue;
};


};  // namespace openML
};  // namespace shark
#endif
