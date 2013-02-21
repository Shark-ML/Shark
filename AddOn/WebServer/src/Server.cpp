
#include <shark/Network/Server.h>
#include <shark/Network/Socket.h>
#include <shark/Network/Connection.h>
#include <shark/Network/Tools.h>

#include <vector>


namespace shark {
namespace http {


Server::Server()
: m_stop(false)
{ }

Server::~Server()
{ }


bool Server::run(unsigned short port)
{
	m_stop = false;

	Socket listener(port);
	std::vector<Connection*> conn;

	// main loop -- single threaded http server with concurrency
	while (listener.isGood() && ! m_stop)
	{
		// determine sockets with operations pending
		fd_set read_socks, write_socks;
		FD_ZERO(&read_socks);
		FD_ZERO(&write_socks);
		listener.populateSet(read_socks);
		for (std::size_t i=0; i<conn.size(); i++)
		{
			conn[i]->m_socket->populateSet(read_socks);
			if (conn[i]->canWrite()) conn[i]->m_socket->populateSet(write_socks);
		}
		timeval timeout;
		timeout.tv_sec = 0;
		timeout.tv_usec = 100000;
		int count = select(FD_SETSIZE, &read_socks, &write_socks, NULL, &timeout);

		if (count < 0)
		{
			// something went terribly wrong
			break;
		}
		else if (count > 0)
		{
			for (std::size_t i=0; i<conn.size(); i++)
			{
				Connection* c = conn[i];
				if (c->m_socket->isInSet(read_socks))
				{
					if (c->processRead())
					{
						// check status
						if (c->isRequestReady())
						{
							// process the request
							try
							{
								// invoke the central callback
								handleRequest(*c);
							}
							catch (...)
							{
								// close the connection, don't affect the others
								c->close();
								continue;
							}

							// prepare the next request
							c->resetRequest();
						}
					}
				}
				if (c->m_socket->isInSet(write_socks))
				{
					c->processWrite();
				}
			}

			if (listener.isInSet(read_socks))
			{
				Socket* s = listener.accept();
				if (s != NULL) conn.push_back(new Connection(s));
			}
		}

		// check for dead connections
		for (int i=conn.size()-1; i>=0; i--)
		{
			if (! conn[i]->m_socket->isGood())
			{
				delete conn[i];
				conn.erase(conn.begin() + i);
			}
		}
	}

	// clean up
	for (std::size_t i=0; i<conn.size(); i++) delete conn[i];

	// did we exit cleanly or with a problem?
	return m_stop;
}

void Server::stop()
{ m_stop = true; }


}}
