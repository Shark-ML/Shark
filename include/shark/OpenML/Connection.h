//===========================================================================
/*!
 * 
 *
 * \brief       Definition of an OpenML Connection.
 * 
 * 
 * \par
 * This file provides methods and classes for easy access to the OpenML
 * platform for open machine learning research.
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

#ifndef SHARK_OPENML_CONNECTION_H
#define SHARK_OPENML_CONNECTION_H

#include "Entity.h"
#include "detail/Socket.h"
#include "detail/Json.h"
#include "detail/HttpResponse.h"

#include <string>
#include <utility>
#include <vector>


namespace shark {
namespace openML {


class CachedFile;


/// \brief OpenML management class.
///
/// The Connection class handles the communication with the OpenML
/// webservice through the JSON-based endpoint of its REST API.
SHARK_EXPORT_SYMBOL class Connection
{
public:
	friend class CachedFile;

	typedef std::vector< std::pair<std::string, std::string> > ParamType;

	Connection();
	Connection(std::string const& host, unsigned short port = 80, std::string const& prefix = "");
	~Connection();

	/// \brief Obtain the currently set api_key.
	std::string const& key() const
	{ return m_key; }

	/// \brief Set the OpenML api_key.
	void setKey(std::string const& api_key)
	{ m_key = api_key; }

	/// \brief Send an http GET request, expecting a JSON object back.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   parameters   tagged-values sent as URL-encoded parameters
	/// \return  The function returns the JSON reply sent by the server. If the connection is not established it returns a JSON null object. In case of an unsuccessful query it returns the status code as a JSON number.
	detail::Json get(std::string const& request, ParamType const& parameters = ParamType());

	/// \brief Send an http POST request, expecting a JSON object back.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   body         (unencoded) message body
	/// \param   parameters   tagged-values sent as URL-encoded form data
	/// \return  The function returns the JSON reply sent by the server. If the connection is not established it returns a JSON null object. In case of an unsuccessful query it returns the status code as a JSON number.
	detail::Json post(std::string const& request, ParamType const& parameters = ParamType());

private:
	// non-copyable
	Connection(Connection const&);
	Connection& operator = (Connection const&);

	/// \brief Send an http GET request.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   parameters   tagged-values sent as URL-encoded parameters
	detail::HttpResponse getHTTP(std::string const& request, ParamType const& parameters = ParamType());

	/// \brief Send an http POST request.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   parameters   tagged-values sent as URL-encoded form data
	detail::HttpResponse postHTTP(std::string const& request, ParamType const& parameters = ParamType());

	/// \brief Download some more data from the socket and append it to the buffer.
	std::size_t download();

	/// \brief Download a full HTTP response from the socket.
	bool receiveResponse(detail::HttpResponse& response);

	std::string m_host;
	unsigned short m_port;
	std::string m_key;
	std::string m_prefix;
	detail::Socket m_socket;
	std::string m_readbuffer;
};


SHARK_EXPORT_SYMBOL extern Connection connection;


};  // namespace openML
};  // namespace shark
#endif
