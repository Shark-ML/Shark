/**
 *
 *  \brief Implements a generic logging facility.
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
#ifndef SHARK_CORE_LOGGER_H
#define SHARK_CORE_LOGGER_H

#include <shark/Core/Exception.h>
#include <shark/Core/Factory.h>

#include <boost/assign.hpp>
#include <boost/bimap.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/thread.hpp>

#include <ostream>

namespace shark {

	/**
	* \brief Implements a generic logging facility.
	*
	* Simple logging mechanism and a singleton logger pool for logger management:
	* \code
	* boost::shared_ptr<
	*	shark::Logger
	* > sharkLogger = shark::LoggerPool::instance().registerLogger( "edu.rub.ini.shark" );
	* sharkLogger << shark::Logger::Record( shark::Logger::INFO, "Info message.", "Test source" );
	* \endcode
	*/
	class Logger {
	public:

		/**
		* \brief Default time quantity.
		*/
		typedef boost::posix_time::ptime time_type;

		/**
		* \brief Models the different log levels known to the system.
		*/
		enum Level {
			//it's a bad idea to call somethind DEBUG when we diefine DEBUG in debug mode...
			DEBUG_LEVEL		= 1, ///< Models debugging/tracing information.
			INFO_LEVEL		= 2, ///< Models general information for reporting system states etc..
			WARNING_LEVEL		= 4, ///< Models warning conditions.
			ERROR_LEVEL		= 8	 ///< Models actual errors.
		};

		/** \brief Look-up-table type for converting between string and log-level. */
		typedef boost::bimap< 
			Level,
			std::string
		> lut_type;

		/** \brief Default look-up table for Level <-> string conversion. */
		static lut_type LEVEL_LUT;

		/**
		* \brief Converts the supplied log level to a human-readable string representation.
		* \throws shark::Exception if log level is not known.
		*/
		static const std::string & toString( Level level ) {
			lut_type::left_const_iterator it = LEVEL_LUT.left.find( level );
			if( it == LEVEL_LUT.left.end() )
				throw( SHARKEXCEPTION( "Logger::toString: No such log level." ) );
			return( it->second );
		}

		/**
		* \brief Converts the supplied log level to an element of enumeration log level.
		* \throws shark::Exception if the log level is not known.
		*/
		static Level fromString( const std::string & level ) {
			lut_type::right_const_iterator it = LEVEL_LUT.right.find( level );
			if( it == LEVEL_LUT.right.end() )
				throw( SHARKEXCEPTION( "Logger::toString: No such log level." ) );
			return( it->second );
		}

		/**
		* \brief Models a single log record passed to the logging framework.
		* 
		* A log record consists of:
		*   - a log level
		*   - a human-readable message
		*   - the source of the message
		*   - the filename that the message originated from.
		*   - the line in the file that the message originated from.
		*   - A posix-compliant timestamp.
		*/
		class Record {
		public:

			/**
			* \brief Constructs an empty log record.
			*
			* The log level is initialized to INFO and the timestamp is
			* initialized to now.
			*/
			Record() : m_logLevel( INFO_LEVEL ),
				m_line( 0 ),
				m_timeStamp( boost::posix_time::microsec_clock::local_time() ) {}

			/**
			* \brief Constructs a log record with the given values.
			*/
			Record( Level logLevel,
				const std::string & message,
				const std::string & source,
				const std::string & file = std::string( __FILE__ ),
				std::size_t line = std::size_t( __LINE__ )
				) : m_logLevel( logLevel ),
				m_message( message ),
				m_source( source ),
				m_file( file ),
				m_line( line ),
				m_timeStamp( boost::posix_time::microsec_clock::local_time() ) {
			}

			/**
			* \brief Accesses the log level of this record.
			*/
			Level logLevel() const {
				return( m_logLevel );
			}

			/**
			* \brief Accesses a non-const reference to the log level of this record.
			*/
			Level & logLevel() {
				return( m_logLevel );
			}

			/**
			* \brief Accesses the message associated with this record.
			*/
			const std::string & message() const {
				return( m_message );
			}

			/**
			* \brief Accesses a non-const reference to the message associated with this record.
			*/
			std::string & message() {
				return( m_message );
			}

			/**
			* \brief Accesses the source of this log record.
			*/
			const std::string & source() const {
				return( m_source );
			}

			/**
			* \brief Accesses a non-const reference to the source of this log record.
			*/
			std::string & source() {
				return( m_source );
			}

			/**
			* \brief Accesses the file this record originates from.
			*/
			const std::string & file() const {
				return( m_file );
			}

			/**
			* \brief Accesses a non-const reference to the file this record originates from.
			*/
			std::string & file() {
				return( m_file );
			}

			/**
			* \brief Accesses the line in the file this record originates from.
			*/
			std::size_t line() const {
				return( m_line );
			}

			/**
			* \brief Accesses a non-const reference to the line in the file this record originates from.
			*/
			std::size_t & line() {
				return( m_line );
			}

			/**
			* \brief Accesses the timestamp of this message.
			*/
			time_type timestamp() const {
				return( m_timeStamp );
			}

		protected:
			Level m_logLevel; ///< Log level of the record.
			std::string m_message; ///< Message of the record.
			std::string m_source; ///< Source of the record.
			std::string m_file; ///< File from which the record originates.
			std::size_t m_line; ///< Line of file from which the record originates.
			time_type m_timeStamp; ///< Timestamp of the message.

		};

		/**
		* \brief Entrypoint for extending the logging framework with custom format handlers.
		*/
		class AbstractFormatter {
		public:

			/**
			* \brief Virtual d'tor.
			*/
			virtual ~AbstractFormatter() {}

			/**
			* \brief Accesses the name of the formatter.
			*/
			virtual const std::string & name() const {
				return( m_name );
			}

			/**
			* \brief Called by handler, needs to be implemented by subclasses.
			*/
			virtual std::string handle( const Record & record ) = 0;

		protected:
			std::string m_name; ///< Models the name of the formatter.
		};

		/**
		* \brief Entrypoint for extending the logging framework with custom log handlers.
		*/
		class AbstractHandler {
		public:

			/**
			* \brief Virtual d'tor.
			*/
			virtual ~AbstractHandler() {}

			/**
			* \brief Accesses the name of the handler.
			*/
			virtual const std::string & name() const {
				return( m_name );
			}

			virtual void setFormatter( const boost::shared_ptr< AbstractFormatter > & formatter ) {
				mp_formatter = formatter;
			}
			/**
			* \brief Called by logger, needs to be implemented by subclasses.
			*/
			virtual void handle( const Record & record ) = 0;

		protected:
			std::string m_name; ///< Models the name of the handler.
			boost::shared_ptr< AbstractFormatter > mp_formatter; ///< Stores the current formatter.
		};

		Logger() : m_logLevel( INFO_LEVEL ) {}

		template<typename PropertyTree>
		void configure( const PropertyTree & root ) {
			
			m_logLevel = LEVEL_LUT.right.find( root.template get< std::string >( "LogLevel" ) )->second;

		}

		/**
		* \brief Processes the supplied record and passes it to the registered handlers.
		*/
		void operator()( const Record & record ) {

			if( record.logLevel() < logLevel() )
				return;

			boost::lock_guard< boost::mutex > lg( m_registeredHandlersGuard );
			BOOST_FOREACH( boost::shared_ptr< AbstractHandler > handler, m_registeredHandlers ) {
				handler->handle( record );
			}
		}

		/**
		* \brief Accesses the log level of this logger.
		*/
		Level logLevel() const {
			return( m_logLevel );
		}

		/**
		* \brief Adjusts the log level of this logger.
		*/
		void setLogLevel( Level logLevel ) {
			m_logLevel = logLevel;
		}

		/**
		* \brief Registers a handler instance with this logger.
		*/
		void registerHandler( const boost::shared_ptr< AbstractHandler > & handler ) {
			boost::lock_guard< boost::mutex > lg( m_registeredHandlersGuard );
			m_registeredHandlers.insert( handler );
		}

		/**
		* \brief Deregisters a handler instance from this logger.
		*/
		void unregisterHandler( const boost::shared_ptr< AbstractHandler > & handler ) {
			boost::lock_guard< boost::mutex > lg( m_registeredHandlersGuard );
			m_registeredHandlers.erase( handler );
		}

		/**
		* \brief Queries this logger instance if the handler is installed.
		*/
		bool hasHandler( const boost::shared_ptr< AbstractHandler > & handler ) const {
			return( m_registeredHandlers.count( handler ) > 0 );
		}

	protected:
		Level m_logLevel; ///< Log level of the logger instance.
		boost::unordered_set< 
			boost::shared_ptr< 
			AbstractHandler 
			> 
		> m_registeredHandlers; ///< Hash-set of registered handlers.
		boost::mutex m_registeredHandlersGuard; ///< Guards the set of registered handlers.

	};

	/**
	* \brief Pushes the record in the specified logger.
	*/
	Logger & operator<<( Logger & logger, const Logger::Record & record ) {
		logger( record );
		return( logger );
	}


	Logger::lut_type Logger::LEVEL_LUT = boost::assign::list_of< Logger::lut_type::relation >
		( Logger::DEBUG_LEVEL, "Debug" )
		( Logger::INFO_LEVEL, "Info" )
		( Logger::WARNING_LEVEL, "Warning" )
		( Logger::ERROR_LEVEL, "Error" );

	/**
	* \brief Singleton that manages named loggers.
	*/
	    class LoggerPool : /** \cond */ private boost::noncopyable /** \endcond */ {
	public:

		/** \brief Type of the name <-> logger registry. */
		typedef boost::bimap<
			std::string,
			boost::shared_ptr<
			Logger
			>
		> registry_type;

		/** \brief Marks the managed pointer type. */
		typedef boost::shared_ptr<
			Logger
		> ptr_type;

		/**
		* \brief Accesses the unique instance of the logger pool.
		*/
		static LoggerPool & instance() {
			static LoggerPool pool;
			return( pool );
		}

		/**
		* \brief Registers a new logger for the supplied name.
		* \returns A managed pointer to the named logger.
		*/
		ptr_type registerLogger( const std::string & name ) {
			boost::lock_guard< boost::mutex > lg( m_registryGuard );

			registry_type::left_iterator it = m_registry.left.find( name );

			if( it != m_registry.left.end() )
				return( it->second );

			boost::shared_ptr< Logger > p( new Logger() );
			m_registry.insert( registry_type::value_type( name, p ) );

			return( p );
		}

		/**
		* \brief Unregisters all loggers for the supplied name.
		*/
		void unregisterLogger( const std::string & name ) {
			boost::lock_guard< boost::mutex > lg( m_registryGuard );
			m_registry.left.erase( name ); 
		}

		/**
		* \brief Unregisters the supplied logger.
		*/
		void unregisterLogger( const boost::shared_ptr< Logger > & logger ) {
			boost::lock_guard< boost::mutex > lg( m_registryGuard );
			m_registry.right.erase( logger ); 
		}

	protected:

		/**
		* \brief Default c'tor.
		*/
		LoggerPool() {}

		registry_type m_registry; ///< Manages named loggers.
		boost::mutex m_registryGuard; ///< Guards the registry.
	};

}

/**
* \brief Convenience macro for logging debug information.
*/
#define SHARK_LOG_DEBUG( logger, message, source ) (*logger) << shark::Logger::Record( shark::Logger::DEBUG_LEVEL, message, source, __FILE__, __LINE__ );

/**
* \brief Convenience macro for logging generic information.
*/
#define SHARK_LOG_INFO( logger, message, source ) (*logger) << shark::Logger::Record( shark::Logger::INFO_LEVEL, message, source, __FILE__, __LINE__ );

/**
* \brief Convenience macro for logging warnings.
*/
#define SHARK_LOG_WARNING( logger, message, source ) (*logger) << shark::Logger::Record( shark::Logger::WARNING_LEVEL, message, source, __FILE__, __LINE__ );

/**
* \brief Convenience macro for logging errors.
*/
#define SHARK_LOG_ERROR( logger, message, source ) (*logger) << shark::Logger::Record( shark::Logger::ERROR_LEVEL, message, source, __FILE__, __LINE__ );

namespace shark {

	/** \brief Defines the default factory type for log handler implementations. */
	typedef shark::Factory< shark::Logger::AbstractHandler, std::string > LogHandlerFactory;
	
	/** \brief Defines the default factory type for log formatter implementations. */
	typedef shark::Factory< shark::Logger::AbstractFormatter, std::string > LogFormatterFactory;
}

/**
* \brief Convenience macro for registering log handler implementations.
*/
#define ANNOUNCE_LOG_HANDLER( Handler, Factory ) namespace Handler ## _namespace {\
			typedef TypeErasedAbstractFactory< Handler, Factory > abstract_factory_type;\
			typedef FactoryRegisterer< Factory > factory_registerer_type;\
			static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Handler, new abstract_factory_type() );\
		}\

/**
* \brief Convenience macro for registering log handler implementations.
*/
#define ANNOUNCE_LOG_FORMATTER( Formatter, Factory ) namespace Formatter ## _namespace {\
	typedef TypeErasedAbstractFactory< Formatter, Factory > abstract_factory_type;\
	typedef FactoryRegisterer< Factory > factory_registerer_type;\
	static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Formatter, new abstract_factory_type() );\
}\
	

#endif // SHARK_CORE_LOGGER_H
