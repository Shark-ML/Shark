/**
 *
 *  \brief Implements a custom HTTP/HTML Shark about handler.
 *
 *  \author  T. Voss
 *  \date    2011
 *
 *  \par Copyright (c) 2007-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
#ifndef SHARK_NETWORK_ABOUT_HANDLER_H
#define SHARK_NETWORK_ABOUT_HANDLER_H

#include <shark/Core/Shark.h>
#include <shark/Network/HttpServer.h>

#include <boost/bimap.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/noncopyable.hpp>

namespace shark {
    
    /**
     * \brief Implements an http server request providing information on the
     * shark library.
     */
    struct SharkAboutHandler : public HttpServer::AbstractRequestHandler {
        
        /** \brief Default HTML format for assembling an http response. */
        static const char * HTML_FORMAT() {
            return(
                   "\
                   <html>\
                   <body>\
                   <h1>Welcome to the <a href=\"http://shark-project.sourceforge.net\">Shark Machine Learning Library</a></h1>\
                   <table>\
                   <tr><td>Version:</td>			<td>%1%</td></tr>\
                   <tr><td>Official Release:</td>	<td>%2%</td></tr>\
                   <tr><td>Platform:</td>			<td>%3%</td></tr>\
                   <tr><td>Compiler:</td>			<td>%4%</td></tr>\
                   <tr><td>C++ Std. Lib.:</td>		<td>%5%</td></tr>\
                   <tr><td>Boost Version:</td>		<td>%6%</td></tr>\
                   <tr><td>Build Type:</td>		<td>%7%</td></tr>\
                   <tr><td>OpenMP Enabled:</td>	<td>%8%</td></tr>\
                   </table>\
                   </body>\
                   </html>\
                   "
                   );
        }
        
        /**
         * \brief Handles the supplied request, generates a response containing 
         * information on the shark library and sends it to the requesting client.
         */
        void handle( const HttpServer::request_type & request, HttpServer::connection_ptr_type connection ) {
            
            if( request.method != "GET" )
                return;
            
            std::stringstream ss;
            Shark::info( ss );
            boost::format htmlFormat( SharkAboutHandler::HTML_FORMAT() );
            htmlFormat = 
            htmlFormat % 
            (	boost::format( Shark::version_type::DEFAULT_FORMAT() ) 
             % Shark::version_type::MAJOR() 
             % Shark::version_type::MINOR() 
             % Shark::version_type::PATCH() 
             ).str() %
            ( Shark::isOfficialRelease() ? "true" : "false" ) %
            BOOST_PLATFORM %
            BOOST_COMPILER %
            BOOST_STDLIB %
            ( boost::format( Shark::boost_version_type::DEFAULT_FORMAT() ) 
             % Shark::boost_version_type::MAJOR() 
             % Shark::boost_version_type::MINOR() 
             % Shark::boost_version_type::PATCH() 
             ).str() %
            Shark::buildType() %
            (Shark::hasOpenMp() ? "true" : "false");
            
            HttpServer::async_server_type::response_header headers[] = {
                {"Content-Type", "text/html"},
                {"Content-Length", "0"}
            };
            
            std::string content = htmlFormat.str();
            headers[ 1 ].value = boost::lexical_cast< std::string >( content.size() );
            
            connection->set_status( HttpServer::connection_type::ok );
            connection->set_headers( boost::make_iterator_range(headers, headers+2) );
            connection->write( content );
        }
    };
}

#endif
