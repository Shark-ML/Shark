/**
 *
 *  \brief Implements a generic log handler for arbitrary streams.
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
#ifndef SHARK_CORE_LOG_HANDLERS_STREAM_HANDLER_H
#define SHARK_CORE_LOG_HANDLERS_STREAM_HANDLER_H

#include <shark/Core/Logger.h>

#include <ostream>

namespace shark {

	/**
	* \brief Implements a generic log handler for arbitrary streams.
	* \tparam Stream Needs to provide operator<< for char.
	*/
	template<typename Stream>
	class StreamHandlerBase : public Logger::AbstractHandler {
	public:

		static const char * CLASS_NAME() {
			return( "StreamHandlerBase< Stream >" );
		}

		/** \brief Constructs a stream handler with the supplied stream. */
		StreamHandlerBase( Stream & s = std::cout ) : m_stream( s ) {
			m_name = CLASS_NAME();
		}

		/** \brief Virtual d'tor. */
		virtual ~StreamHandlerBase() {}

		/**
		* \brief Handles the supplied log record.
		* 
		* If the formatter is empty, this method returns immediately.
		*/
		void handle( const Logger::Record & record ) {
			if( !mp_formatter )
				return;
			
			std::string line = mp_formatter->handle( record );
			std::copy( 
				line.begin(), 
				line.end(), 
				std::ostream_iterator< std::string::value_type >( m_stream ) 
			);
			m_stream << "\n";
		}

		/**
		* \brief Accesses a mutable reference to the associated stream of this handler.
		*/
		Stream & stream() {
			return( m_stream );
		}

	protected:
		Stream & m_stream; ///< Mutable reference to the stream.
	};

	/** \brief Tags a stream handler for STL ostreams. */
	typedef StreamHandlerBase< std::ostream > StlStreamHandler;

	namespace tag {

		/** \brief Tags the std::cout stream. */
		struct cout {};

		/** \brief Tags the std::clog stream. */
		struct clog {};

		/** \brief Tags the std::cerr stream. */
		struct cerr {};
	}

	/** \brief Defines a console handler based on tag dispatching. */
	template<typename Tag>
	struct ConsoleHandler : public StlStreamHandler {

		/** \brief Default c'tor. Constructs the stream handler for std::cout */
		ConsoleHandler() : StlStreamHandler( std::cout ) {}

	};

	/** \brief Template specialization for std::cout. */
	template<>
	struct ConsoleHandler< tag::cout > : public StlStreamHandler {

		ConsoleHandler() : StlStreamHandler( std::cout ) {}

	};

	/** \brief Template specialization for std::clog. */
	template<>
	struct ConsoleHandler< tag::clog > : public StlStreamHandler {

		ConsoleHandler() : StlStreamHandler( std::clog ) {}

	};

	/** \brief Template specialization for std::cerr. */
	template<>
	struct ConsoleHandler< tag::cerr > : public StlStreamHandler {

		ConsoleHandler() : StlStreamHandler( std::cerr ) {}

	};
	
	/** \brief Log handler outputting to std::cout. */
	typedef ConsoleHandler< tag::cout > CoutLogHandler;

	/** \brief Log handler outputting to std::cerr. */
	typedef ConsoleHandler< tag::cerr > CerrLogHandler;

	/** \brief Log handler outputting to std::clog. */
	typedef ConsoleHandler< tag::clog > ClogLogHandler;

	/** \brief Make the std::cout log handler known to the factory. */
	ANNOUNCE_LOG_HANDLER( CoutLogHandler, LogHandlerFactory );	

	/** \brief Make the std::cerr log handler known to the factory. */
	ANNOUNCE_LOG_HANDLER( CerrLogHandler, LogHandlerFactory );

	/** \brief Make the std::clog log handler known to the factory. */
	ANNOUNCE_LOG_HANDLER( ClogLogHandler, LogHandlerFactory );
}

#endif