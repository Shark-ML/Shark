//===========================================================================
/*!
 * 
 *
 * \brief       HTTP response used internally by the OpenML module.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
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

#ifndef SHARK_OPENML_DETAIL_HTTPRESPONSE_H
#define SHARK_OPENML_DETAIL_HTTPRESPONSE_H


#include <string>
#include <map>


namespace shark {
namespace openML {

class Connection;

namespace detail {


//
// \brief HTTP response used internally by the OpenML wrapper.
//
class HttpResponse
{
public:
	friend class shark::openML::Connection;

	typedef std::map<std::string, std::string> DictionaryType;

	/// \brief Default constructor, sets the status code to 0.
	HttpResponse()
	: m_statusCode(0)
	{ }

	/// \brief Return the http status code.
	inline unsigned int statusCode() const
	{ return m_statusCode; }

	/// \brief Return the (human readable) http "return phrase" corresponding to the status code.
	inline std::string const& returnPhrase() const
	{ return m_returnPhrase; }

	/// \brief Return the set of all http response header fields.
	inline DictionaryType const& headers() const
	{ return m_header; }

	/// \brief Check for the existence of a header field.
	///
	/// \par
	/// Note: header names are lower-case, and header values are stripped off leading spaces.
	inline bool hasHeaderField(std::string const& name) const
	{ return (m_header.count(name) > 0); }

	/// \brief Return the value of an existing header field or the empty string.
	///
	/// \par
	/// Note: header names are lower-case, and header values are stripped off leading spaces.
	inline std::string header(std::string const& name) const
	{
		DictionaryType::const_iterator it = m_header.find(name);
		if (it == m_header.end()) return std::string();
		return it->second;
	}

	/// \brief Return the http body.
	inline std::string const& body() const
	{ return m_body; }

private:
	unsigned int m_statusCode;      ///< HTTP status code
	std::string m_returnPhrase;     ///< HTTP return phrase
	DictionaryType m_header;        ///< collection of all HTTP headers
	std::string m_body;             ///< decoded HTTP message body
};


};  // namespace detail
};  // namespace openML
};  // namespace shark
#endif
