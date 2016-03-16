//===========================================================================
/*!
 * 
 *
 * \brief       Socket class used internally by the OpenML wrapper.
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

#include <shark/OpenML/detail/Socket.h>
#include <shark/Core/Exception.h>

#include <cstring>
#include <cerrno>
#include <csignal>

#ifdef _WIN32
	#include <winsock2.h>
	typedef int socklen_t;

	#ifndef MSG_NOSIGNAL
		#define MSG_NOSIGNAL 0
	#endif

	#pragma comment(lib, "ws2_32.lib")
#else
	#include <netinet/tcp.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <sys/socket.h>
	#include <sys/time.h>
	#include <netdb.h>
	#include <unistd.h>
	#ifndef MSG_NOSIGNAL
		#define MSG_NOSIGNAL SO_NOSIGPIPE
	#endif
#endif

#define DISABLE_NAGLE


using namespace std;


namespace shark {
namespace openML {
namespace detail {


// static
bool Socket::m_initialized = false;


Socket::Socket()
: m_handle(0)
{
	initialize();
}

Socket::Socket(string const& remoteHost, unsigned short remotePort)
: m_handle(0)
{
	initialize();

	connect(remoteHost, remotePort);
}

Socket::Socket(unsigned short localPort, std::string const& remoteIP)
: m_handle(0)
{
	initialize();

	listen(localPort, remoteIP);
}

Socket::~Socket()
{
	close();
}

// static
void Socket::initialize()
{
	if (! m_initialized)
	{
#ifdef _WIN32
		WSADATA info;
		int result = WSAStartup(2 /* API version 2.0 */, &info);
		if (result != 0) throw SHARKEXCEPTION("[Socket::initialize] windows socket API startup failed");
#else
		signal(SIGPIPE, SIG_IGN);
#endif
		m_initialized = true;
	}
}


bool Socket::connect(string const& remoteHost, unsigned short remotePort)
{
	// refuse if socket is already connected
	if (m_handle > 0) return false;

	m_handle = ::socket(AF_INET, SOCK_STREAM, 0);
	if (m_handle <= 0) return false;

#ifdef DISABLE_NAGLE
	int flag = 1;
	setsockopt(m_handle, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag));
#endif

	hostent *host = gethostbyname(remoteHost.c_str());
	if (host == NULL)
	{
		close();
		return false;
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(remotePort);
	addr.sin_addr.s_addr = ((in_addr *)host->h_addr)->s_addr;
	memset(&(addr.sin_zero), 0, 8);

	int ret = ::connect(m_handle, (sockaddr*) & addr, sizeof(sockaddr));
	if (ret < 0) close();

	return connected();
}

bool Socket::listen(unsigned short localPort, std::string const& remoteIP)
{
	// refuse if socket is already connected
	if (m_handle > 0) return false;

	m_handle = ::socket(AF_INET, SOCK_STREAM, 0);
	if (m_handle <= 0) return false;

#ifdef _WIN32
	char yes = 1;
#else
	int yes = 1;
#endif
	if (setsockopt(m_handle, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
	{
		close();
		return false;
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(localPort);
	if (remoteIP.empty()) addr.sin_addr.s_addr = INADDR_ANY;
	else
	{
		if (inet_aton(remoteIP.c_str(), &addr.sin_addr) == 0)
		{
			close();
			return false;
		}
	}
	memset(&(addr.sin_zero), 0, 8);

	if (::bind(m_handle, (sockaddr*)&addr, sizeof(sockaddr)) == -1)
	{
		close();
		return false;
	}

	if (::listen(m_handle, 5) == -1)
	{
		close();
		return false;
	}

	return true;
}

bool Socket::setHandle(int handle)
{
	if (m_handle) return false;
	m_handle = handle;
	return true;
}

void Socket::close()
{
#ifdef _WIN32
	if (m_handle > 0) ::closesocket(m_handle);
#else
	if (m_handle > 0) ::close(m_handle);
#endif
	m_handle = 0;
}

bool Socket::peer(std::vector<unsigned char>& ip, unsigned short& port)
{
	sockaddr_in s;
	socklen_t len = sizeof(sockaddr_in);
	int err = getpeername(m_handle, (sockaddr*)(&s), &len);
	if (err == 0)
	{
		unsigned int ip4 = ntohl(s.sin_addr.s_addr);
		ip.resize(4);
		for (std::size_t i=0; i<4; i++) ip[3-i] = (ip4 >> (8*i)) & 0xff;
		port = ntohs(s.sin_port);
		return true;
	}
	else
	{
		close();
		return false;
	}
}

bool Socket::hasData(unsigned int timeoutSeconds, unsigned int timeoutUSeconds)
{
	if (m_handle == 0) return false;

	timeval timeout;
	timeout.tv_sec = timeoutSeconds;
	timeout.tv_usec = timeoutUSeconds;

	fd_set read_socks;
	FD_ZERO(&read_socks);
	FD_SET(m_handle, &read_socks);

	int count = select(m_handle + 1, &read_socks, NULL, NULL, &timeout);
	// bool ret = (FD_ISSET(m_handle, &read_socks));
	bool ret = (count == 1);

	FD_ZERO(&read_socks);
	if (count == -1)
	{
		close();
		return false;
	}

	return ret;
}

std::size_t Socket::read(char* buffer, std::size_t buffersize)
{
	int ret = recv(m_handle, buffer, buffersize, 0);
	if (ret <= 0)
	{
		ret = 0;
		close();
	}
	return ret;
}

std::size_t Socket::write(const char* buffer, std::size_t buffersize)
{
	if (buffersize == 0) return 0;
	int ret = send(m_handle, buffer, buffersize, MSG_NOSIGNAL);
	if (ret <= 0)
	{
		ret = 0;
		close();
	}
	return ret;
}

bool Socket::writeAll(const char* buffer, std::size_t buffersize)
{
	std::size_t pos = 0;
	while (pos < buffersize)
	{
		std::size_t n = write(buffer + pos, buffersize - pos);
		if (n == 0) return false;
		pos += n;
	}
	return true;
}

std::shared_ptr<Socket> Socket::accept()
{
	int handle = ::accept(m_handle, NULL, NULL);
	if (handle < 0) return std::shared_ptr<Socket>();
	std::shared_ptr<Socket> ret = make_shared<Socket>();
	ret->setHandle(handle);
	return ret;
}


};  // namespace detail
};  // namespace openML
};  // namespace shark
