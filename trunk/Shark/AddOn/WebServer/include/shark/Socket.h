/**
 *
 *  \brief OS independent TCP/IP socket abstraction.
 *
 *  \author  T. Glasmachers
 *  \date    2013
 *
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
#ifndef SHARK_NETWORK_SOCKET_H
#define SHARK_NETWORK_SOCKET_H


// OS dependent headers
#include <errno.h>
#include <sys/types.h>
#ifdef _WIN32
	#include <windows.h>
	typedef int socklen_t;
	#ifndef MSG_NOSIGNAL
		#define MSG_NOSIGNAL 0
	#endif
#else
	#include <netinet/tcp.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <sys/socket.h>
	#include <netdb.h>
	#include <unistd.h>
#endif

#include <string>


namespace shark {
namespace http {


//
// Rather OS independent socket abstraction.
//
class Socket
{
public:
	// connect to a remort host
	Socket(std::string remote_url, unsigned short remote_port);

	// listen on a local port
	explicit Socket(unsigned short local_port);

	// close the connection
	~Socket();


	// check whether the connection is fine
	inline bool isGood() const
	{ return (m_handle > 0); }

	// close the connection (makes isGood() return false)
	void close();

	// check whether the socket has data ready for an asynchronous read operation
	bool hasData();

	// asynchronous read, may return size zero if no data is pending
	std::size_t read(char* buffer, std::size_t buffersize);

	// elementary write operations are always synchronous
	// (even if some library tries to convince you of the contrary)
	std::size_t write(const char* buffer, std::size_t buffersize);

	// Accept a pending connection (if hasData() is false then this call is blocking)
	// and return a newly allocated socket object representing this connection.
	// The caller is responsible for deleting the socket object.
	Socket* accept();

	// add the socket's descriptor to the set
	void populateSet(fd_set& set) const;

	// check whether the socket is listed in the set
	bool isInSet(fd_set& set) const;

private:
	explicit Socket(int handle, bool disambiguate);

	// POSIX socket handle
	int m_handle;
};


}}
#endif
