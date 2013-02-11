//===========================================================================
/*!
*
*  \brief HttpServer example program
*
*  \author  T. Voss, T. Glasmachers
*  \date  2011, 2013
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
//===========================================================================

#include <shark/Network/HttpServer.h>
#include <shark/Network/Handlers/AboutHandler.h>
#include <shark/Network/Handlers/FileHandler.h>

#include <shark/Rng/GlobalRng.h>

#include <iostream>
#include <sstream>


using namespace std;
using namespace shark;
using namespace shark::http;


class RandomHandler : public HttpServer::AbstractRequestHandler
{
public:
	RandomHandler()
	: HttpServer::AbstractRequestHandler("RandomHandler")
	{
		m_pattern.push_back("/ProbeManager");
	}

	void handleRequest(Connection& connection)
	{
		Request& request = connection.request();
		if (request.method() != "GET")
		{
			connection.sendError(405, "method not allowed");
			return;
		}

		std::string content = "{\"probes\": [\n";

		std::size_t n = 5 + (rand() % 25);
		for (std::size_t i=0; i<n; i++)
		{
			if (i != 0) content += ", ";

			std::string name, value, timestamp;
			for (int i=0; i<5; i++) name += (char)(rand() % 26) + 'A';
			std::stringstream ss1;
			ss1 << (double)rand() / (double)RAND_MAX;
			value = ss1.str();
			std::stringstream ss2;
			ss2 << 1e6 * (double)rand() / (double)RAND_MAX;
			timestamp = ss2.str();

			content += "{ \"name\": \"" + name
					+ "\", \"value\": \"" + value
					+ "\", \"timestamp\": \"" + timestamp
					+ "\"}";
		}

		content += "]}\n";

		connection.sendDocument(content, "application/json", 0);
	}
};


int main( int argc, char ** argv ) {
	HttpServer httpServer;

	httpServer.registerHandler(new SharkAboutHandler());
	httpServer.registerHandler(new RandomHandler());
	httpServer.registerHandler(new FileHandler(boost::filesystem::initial_path()));

	cout << "Connect your browser to http://localhost:9090/index.html" << endl;
	httpServer.run(9090);

	return( EXIT_SUCCESS );
}
