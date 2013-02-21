/**
 *
 *  \brief Connection of the http server to a client.
 *
 *  \author  T. Glasmachers
 *  \date    2013
 *
 *  \par
 *  The connection object holds different object required for
 *  communication with the browser, including a socket, a request
 *  description, and a request parser. It provides methods for
 *  sending replies to the client.
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *  
 */
#ifndef SHARK_NETWORK_CONNECTION_H
#define SHARK_NETWORK_CONNECTION_H


#include <shark/Network/Socket.h>
#include <shark/Network/Request.h>
#include <shark/Network/RequestParser.h>


namespace shark {
namespace http {


class Server;


///
/// \brief Connection of the http server to a client.
///
/// \par
/// The Connection class offers everything the user program needs for
/// communication with the client.
/// It maintains the underlying socket. Its constructor takes over the
/// ownership of this socket, which is deleted upon destruction of the
/// connection object. It also holds a request object describing the
/// current incoming request, and it provides methods for sending back a
/// response, either a document or an error message.
///
class Connection
{
public:
	/// \brief The constructor takes ownership of the socket!
	Connection(Socket* socket);

	/// \brief The destructor deletes the socket.
	~Connection();


	/// \brief Check whether the connection is working properly.
	///
	/// \par
	/// A return value of false indicates a failure of a previous
	/// operation. Then the connection is closed and it will be deleted
	/// at the next occasion.
	inline bool isGood() const
	{ return m_socket->isGood(); }

	/// \brief Access to the connection's request object.
	Request& request()
	{ return m_request; }

	/// \brief Access to the connection's request object.
	Request const& request() const
	{ return m_request; }

	/// \brief Append data for writing.
	///
	/// \par
	/// This method is used internally by
	/// sendDocument and by sendError. It should rarely be necessary
	/// to use this method directly, however, it is provided for
	/// flexibility.
	void sendRawData(std::string content);

	/// \brief Send a document via http to the client.
	///
	/// \par
	/// The method sends the content as the body of a proper http request
	/// including the following headers: content-type, cache-control,
	/// date, expires, and content-length. Additional headers can be
	/// included at need.
	/// expires: validity of the document in seconds
	///          (indicated by the date and expires
	///          header fields); in case of expires=0
	///          the additional header field
	///          "cache-control: no-cache" is sent.
	/// additionalHeaders: string with format
	///          * ( <name> ":" <value> "\r\n" )
	void sendDocument(std::string const& content, std::string mime, unsigned int expires = 0, std::string additionalHeaders = "");

	/// \brief Send an http error message to the client.
	///
	/// \par
	/// The message contains the error code with the textual
	/// description, both in the http header and in the
	/// html body. A typical use case is
	///    sendError(404, "file not found");
	void sendError(unsigned int code, std::string message);

	/// \brief Close the connection immediately.
	///
	/// \par
	/// This method closes the connection without even sending an error
	/// message or pending data. This may be appropriate as a reaction to
	/// malicious requests or in case of a severe problem.
	void close();

private:
	friend class Server;

	/// \brief Does the connection have data ready for sending?
	inline bool canWrite() const
	{ return (m_sendOffset < m_sendBuffer.size()); }

	/// \brief Check whether the request is ready for processing.
	///
	/// This method checks whether all data concerning the request have
	/// arrived. If the method returns false then the request is in an
	/// invalid intermediate state and the server needs to wait for the
	/// arrival of more data, which is to be fed into the corresponding
	/// request parser.
	inline bool isRequestReady() const
	{ return m_parser.isReady(); }

	/// \brief Invalidate the request (typically because it has been handled) and start parsing a new request from the client.
	inline void resetRequest()
	{ m_parser.reset(); }

	/// \brief Read data from the socket into an internal buffer
	bool processRead();

	/// \brief Write buffered data to the socket.
	bool processWrite();

	Socket* m_socket;                           ///< socket underlying the connection
	Request m_request;                          ///< currently handled request
	RequestParser m_parser;                     ///< incremental request parser

	char m_readBuffer[4096];                    ///< buffer for incoming data
	unsigned int m_readUsed;                    ///< number of bytes already processed
	std::string m_sendBuffer;                   ///< buffer for data to be sent
	std::size_t m_sendOffset;                   ///< already sent bytes in m_sendBuffer
};


}}
#endif
