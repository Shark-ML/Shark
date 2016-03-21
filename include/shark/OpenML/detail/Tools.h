//===========================================================================
/*!
 * 
 *
 * \brief       Little tools used internally by the OpenML module.
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


#include "../Base.h"
#include "Json.h"
#include <Shark/Core/Exception.h>
#include <boost/lexical_cast.hpp>
#include <string>


namespace shark {
namespace openML {
namespace detail {


inline std::string urlencode(std::string const& s)
{
	static const char hex[] = "0123456789abcdef";

	std::string ret;
	for (std::size_t i=0; i<s.size(); i++)
	{
		unsigned char c = s[i];
		if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (c == '_') || (c == '-') || (c == '.') || (c == '~')) ret += (char)c;
		else if (c == ' ') ret += '+';
		else
		{
			ret += '%';
			ret += hex[c / 16];
			ret += hex[c % 16];
		}
	}
	return ret;
}

inline std::string xmlencode(std::string const& s)
{
	std::string ret;
	for (std::size_t i=0; i<s.size(); i++)
	{
		char c = s[i];
		if (c == '\"') ret += "&quot;";
		else if (c == '\'') ret += "&apos;";
		else if (c == '<') ret += "&lt;";
		else if (c == '>') ret += "&gt;";
		else if (c == '&') ret += "&amp;";
		else ret += c;
	}
	return ret;
}

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

template <typename T>
T json2number(detail::Json const& json)
{
	if (json.isNumber()) return (IDType)json.asNumber();
	else if (json.isString()) return boost::lexical_cast<T>(json.asString());
	throw SHARKEXCEPTION("failed to convert json value to number: " + json.stringify());
}

inline str::string json2string(detail::Json const& json)
{
	if (json.isString()) return json.asString();
	if (json.isNumber()) return boost::lexical_cast<std::string>(json.asNumber());
	if (json.isBoolean()) return json.asBoolean() ? "true" : "false";
	if (json.isNull()) return "null";
	throw SHARKEXCEPTION("failed to convert json value to string: " + json.stringify());
}


};  // namespace detail
};  // namespace openML
};  // namespace shark
#endif
