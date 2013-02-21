/**
 *
 *  \brief Low-level implementation of a concurrent http server.
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
#ifndef SHARK_NETWORK_SERVER_H
#define SHARK_NETWORK_SERVER_H


#include <shark/Network/Connection.h>
#include <shark/Network/Tools.h>


namespace shark {
namespace http {


//
// Minimal HTTP Server.
// This server is not well suited for serving a large number of clients.
// In Shark, we typically serve only a single client.
//
// This server is single threaded. However, it supports concurrent
// reading and writing to and from an indefinite number of connections.
// This design is simple and efficient; the only price to pay is some
// memory overhead for the storage of complete messages.
//
// The server is started by invoking its run() method. This method is
// running the server's main loop. It does not return as long as the
// server is up and running. The server can be stopped cleanly from
// within a request handler of from another thread by invoking its
// stop() method. It may also stop on a severe error, such as a problem
// with its listening socket.
//
// An application communicates with the server through the virtual
// function handleRequest(connection). This function is called as soon
// as a complete request has arrived at the server. The handler is
// supposed to send a reply back to the client by means of the functions
// connection.sendDocument(...) or connection.sendError(...).
// This reply needs to be assembled timely, since the callback blocks
// the single threaded server. There is no problem with big reply
// messages though, since these are sent piece by piece by the server
// without blocking other connections.
//
class Server
{
public:
	Server();
	virtual ~Server();

	// Listen to the socket. Return false on error and true on clean
	// exit. The method may be called multiple times in a row.
	bool run(unsigned short port = 80);

	// Make the run() method quit cleanly. This happens with a delay of
	// about 0.1 seconds.
	void stop();

	// Request handling is left to sub classes.
	virtual void handleRequest(Connection& connection) = 0;

private:
	// should the loop stop?
	bool m_stop;
};


}}
#endif
