//===========================================================================
/*!
 * 
 *
 * \brief       Socket with TLS encryption, used internally by the OpenML module.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2017
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

#ifndef SHARK_OPENML_DETAIL_SECURESOCKET_H
#define SHARK_OPENML_DETAIL_SECURESOCKET_H


#include <string>
#include <memory>
#include <vector>
#include <cstdint>


namespace shark {
namespace openML {
namespace detail {


//
// Socket with SSL encryption.
// The class allows to operate a TLS connection over a socket. It aims
// for widest possible interface compatibility with the Socket class.
//
// Right now the class implements only client sockets. For a server side
// implementation check the singlethreadedserver library.
//
// WARNING!!!
// The class will happily accept any TLS connection without any serious
// checks of the server certificate. It does *not* check the server's
// identity at all. Hence the class allows to talk to https servers,
// however, without being secure against serious attacks.
//
class SecureSocket
{
public:
	// Connect to a remote host; this makes the socket a client endpoint.
	SecureSocket();

	// close the connection
	~SecureSocket();


	/// \brief Connect to a remote host.
	bool connect(std::string const& remoteUrl, std::uint16_t remotePort = 443);

	// check whether the connection is fine
	bool connected() const;

	// to whom is this socket connected?
	bool peer(std::vector<std::uint8_t>& ip, std::uint16_t& port);

	// close the connection (makes isGood() return false)
	void close();

	// check whether the socket has data ready for a read operation
	bool hasData(unsigned int timeoutSeconds = 0, unsigned int timeoutUSeconds = 0);

	// synchronous read operation, non-blocking if hasData() returns true
	std::size_t read(char* buffer, std::size_t buffersize);

	// write data to the socket
	std::size_t write(const char* buffer, std::size_t buffersize);

	// write the whole buffer (blocking operation)
	bool writeAll(const char* buffer, std::size_t buffersize);

	// access to the OS level socket handle
	int handle() const;

	// hide implementation details
	struct PIMPL;

private:
	// hide implementation details
	std::unique_ptr<PIMPL> pimpl;
};


};  // namespace detail
};  // namespace openML
};  // namespace shark
#endif
