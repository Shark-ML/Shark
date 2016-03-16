//===========================================================================
/*!
 * 
 *
 * \brief       Little tools used internally by the OpenML wrapper.
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

#ifndef SHARK_OPENML_DETAIL_TOOLS_H
#define SHARK_OPENML_DETAIL_TOOLS_H


#include <string>
#include "Json.h"


namespace shark {
namespace openML {
namespace detail {


inline void ASCIItoLowerCase(std::string& s)
{
	for (std::size_t i=0; i<s.size(); i++) if (s[i] >= 65 && s[i] <= 90) s[i] += 32;
}

inline bool json2bool(detail::Json const& json)
{
	if (json.isBoolean()) return json.asBoolean();
	else if (json.isString())
	{
		std::string s = json.asString();
		ASCIItoLowerCase(s);
		if (s == "false") return false;
		if (s == "true") return true;
	}
	throw SHARKEXCEPTION("failed to convert json value to bool: " + json.stringify());
}


};  // namespace detail
};  // namespace openML
};  // namespace shark
#endif
