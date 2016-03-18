//===========================================================================
/*!
 * 
 *
 * \brief       Socket class used internally by the OpenML module.
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

#ifndef SHARK_OPENML_DETAIL_SOCKET_H
#define SHARK_OPENML_DETAIL_SOCKET_H


#ifdef _WIN32
	#include <winsock2.h>
	#include <windows.h>
#else
	#include <sys/types.h>
#endif

#include <memory>
#include <string>
#include <vector>


namespace shark {
namespace openML {
namespace detail {


/// \brief Socket class used internally by the OpenML wrapper.
class Socket
{
public:
	/// \brief Construct an uninitialized (unconnected) socket.
	Socket();

	/// \brief Construct a socket connected to the remote host at the given port.
	Socket(std::string const& remoteHost, unsigned short remotePort);

	/// \brief Listen on a local port, possibly restricted to certain IP (e.g. 127.0.0.1).
	explicit Socket(unsigned short localPort, std::string const& remoteIP = "");

	/// \brief Destruct the socket, close the connection.
	~Socket();


	/// \brief Connect to a remote host.
	bool connect(std::string const& remoteUrl, unsigned short remotePort);

	/// \brief Start listening on a port.
	bool listen(unsigned short localPort, std::string const& remoteIP = "");

	/// \brief Attach a socket to an OS socket handle
	bool setHandle(int handle);

	/// \brief Close the connection.
	void close();


	/// \brief Check whether the connection is fine.
	inline bool connected() const
	{ return (m_handle > 0); }

	/// \brief Obtain IP address and port of the other end.
	///
	/// The function returns false on error. In this case no information is filled in.
	///
	/// \param  ip    remote IP address of the connection
	/// \param  port  remove port of the connection
	bool peer(std::vector<unsigned char>& ip, unsigned short& port);

	/// \brief Check whether the socket has data ready for an asynchronous read operation.
	bool hasData(unsigned int timeoutSeconds = 0, unsigned int timeoutUSeconds = 0);

	/// \brief Read raw data from the socket.
	std::size_t read(char* buffer, std::size_t buffersize);

	/// \brief Write raw data to the socket.
	///
	/// \return  The function returns the number of bytes written, which may be less than buffersize.
	std::size_t write(const char* buffer, std::size_t buffersize);

	/// \brief Write raw data to the socket (blocking).
	///
	/// \return  The function returns true on success and false on failure.
	bool writeAll(const char* buffer, std::size_t buffersize);

	/// \brief Accept an incoming connection.
	///
	/// \par
	/// Accept a pending connection (if hasData() is false then this call is blocking)
	/// and return a newly allocated socket object representing this connection.
	/// The caller is responsible for deleting the returned socket object.
	std::shared_ptr<Socket> accept();

private:
	/// \brief POSIX socket handle
	int m_handle;

	/// \brief Socket initialization
	static void initialize();
	static bool m_initialized;
};


};  // namespace detail
};  // namespace openML
};  // namespace shark
#endif
