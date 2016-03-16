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


namespace shark {
namespace openML {


class Dataset;


/// \brief OpenML management class.
///
/// The Connection class handles severl low-level tasks.
/// Its primary purpose is to encapsulate the OpenML service through its
/// REST API. In addition, the class provides a caching mechanisms for
/// possibly large downloaded dataset files.
class Connection
{
public:
	friend class Dataset;

	Connection();
	Connection(std::string const& host, unsigned short port = 80, std::string const& prefix = "");
	~Connection();

	/// \brief Obtain the currently set api_key.
	std::string const& key() const
	{ return m_key; }

	/// \brief Set the OpenML api_key.
	void setKey(std::string const& api_key)
	{ m_key = api_key; }

	/// \brief Obtain the path of the directory where dataset files are stored.
	PathType const& cacheDirectory() const
	{ return m_cacheDirectory; }

	/// \brief Set the path of the directory where dataset files are stored.
	void setCacheDirectory(PathType const& cacheDirectory)
	{ m_cacheDirectory = cacheDirectory; }

	/// \brief Send an http GET request, expecting a JSON object back.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   parameters   url-encoded string containing parameters in the format "param1&param2&param3..."
	/// \return  The function returns the JSON reply sent by the server. If the connection is not established it returns a JSON null object. In case of an unsuccessful query it returns the status code as a JSON number.
	detail::Json get(std::string const& request, std::string const& parameters = "");

	/// \brief Send an http POST request, expecting a JSON object back.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   body         (unencoded) message body
	/// \param   parameters   url-encoded string containing parameters in the format "param1&param2&param3..."
	/// \return  The function returns the JSON reply sent by the server. If the connection is not established it returns a JSON null object. In case of an unsuccessful query it returns the status code as a JSON number.
	detail::Json post(std::string const& request, std::string const& body, std::string const& parameters = "");

private:
	// non-copyable
	Connection(Connection const&);
	Connection& operator = (Connection const&);

	/// \brief Send an http GET request.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   parameters   url-encoded string containing parameters in the format "param1&param2&param3..."
	detail::HttpResponse getHTTP(std::string const& request, std::string const& parameters = "");

	/// \brief Send an http POST request.
	///
	/// \param   request      REST url, e.g., "/data/list"
	/// \param   body         (unencoded) message body
	/// \param   parameters   url-encoded string containing parameters in the format "param1&param2&param3..."
	detail::HttpResponse postHTTP(std::string const& request, std::string const& body, std::string const& parameters = "");

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
	PathType m_cacheDirectory;
};


extern Connection connection;


};  // namespace openML
};  // namespace shark
#endif
