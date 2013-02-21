
#include <shark/Network/Socket.h>

#include <cstring>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL SO_NOSIGPIPE
#endif


namespace shark {
namespace http {


Socket::Socket(std::string remote_url, unsigned short remote_port)
{
	m_handle = socket(AF_INET, SOCK_STREAM, 0);
	if (m_handle <= 0) return;

	hostent *host = gethostbyname(remote_url.c_str());
	if (host == NULL) return;

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(remote_port);
	addr.sin_addr.s_addr = ((in_addr *)host->h_addr)->s_addr;
	memset(&(addr.sin_zero), 0, 8);

	int ret = connect(m_handle, (sockaddr*) & addr, sizeof(sockaddr));
	if (ret < 0) return;
}

Socket::Socket(unsigned short port)
{
	m_handle = socket(AF_INET, SOCK_STREAM, 0);
	if (m_handle <= 0) return;

	int yes = 1;
	if (setsockopt(m_handle, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
	{
		close();
		return;
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	addr.sin_addr.s_addr = INADDR_ANY;
	memset(&(addr.sin_zero), 0, 8);

	if (bind(m_handle, (sockaddr*)&addr, sizeof(sockaddr)) == -1)
	{
		close();
		return;
	}

	if (listen(m_handle, 5) == -1)
	{
		close();
		return;
	}
}

Socket::Socket(int handle, bool disambiguate)
: m_handle(handle)
{
	(void)disambiguate;
}

Socket::~Socket()
{
	close();
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

bool Socket::hasData()
{
	if (m_handle == 0) return false;

	timeval timeout;
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;

	fd_set read_socks;
	FD_ZERO(&read_socks);
	FD_SET(m_handle, &read_socks);

	int count = select(m_handle + 1, &read_socks, NULL, NULL, &timeout);
	if (count == -1)
	{
		close();
		return false;
	}

	// bool ret = (FD_ISSET(m_handle, &read_socks));
	bool ret = (count == 1);

	FD_ZERO(&read_socks);
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
	int ret = send(m_handle, buffer, buffersize, MSG_NOSIGNAL);
	if (ret <= 0)
	{
		ret = 0;
		close();
	}
	return ret;
}

Socket* Socket::accept()
{
	int comm = ::accept(m_handle, NULL, NULL);
	if (comm < 0) return NULL;
	return new Socket(comm, false);
}

void Socket::populateSet(fd_set& set) const
{
	FD_SET(m_handle, &set);
}

bool Socket::isInSet(fd_set& set) const
{
	return FD_ISSET(m_handle, &set);
}


}}
