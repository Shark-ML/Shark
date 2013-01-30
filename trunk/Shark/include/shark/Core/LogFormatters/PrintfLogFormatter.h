/**
 *
 *  \brief Implements a generic log formatter that relies on an externally 
 *	defined printf-like format.
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
#ifndef SHARK_CORE_LOG_FORMATTERS_PRINTF_LOG_FORMATTER_H
#define SHARK_CORE_LOG_FORMATTERS_PRINTF_LOG_FORMATTER_H

#include <shark/Core/Logger.h>

#include <boost/format.hpp>

namespace shark {

	namespace detail {

		/**
		* \brief Implements a printf-like format for plain text log stores.
		*/
		struct PlainTextPrintfFormatProvider {
			static const char * format() {
				return( "[%1% %2%]: %3% (source: %4%, file: %5%, line: %6%)\n" );
			}
		};

		/**
		* \brief Implements a printf-like format for XML log stores.
		*/
		struct XmlPrintfFormatProvider {
			static const char * format() {
				return( 
					"\
					<Record level=\"%1%\" timestamp=\"%2%\">\n\
					\t<Message>%3%</Message>\n\
					\t<Source>%4%</Source>\n\
					\t<File>%5%</File>\n\
					\t<Line>%6%</Line>\n\
					\t</Record>\n\
					"					
				);
			}
		};

		/**
		* \brief Implements a printf-like format for JSON log stores.
		*/
		struct JsonPrintfFormatProvider {
			static const char * format() {
				return( 
					"\
					\"record\":\n\
					{\n\
					\t\"level\":\"%1%\",\n\
					\t\"timestamp\":\"%2%\",\n\
					\t\"message\":\"%3%\",\n\
					\t\"source\":\"%4%\",\n\
					\t\"file\":\"%5%\",\n\
					\t\"line\":%6%,\n\
					}\n\
					"					
					);
			}
		};
	}

	/**
	* \brief A log formatter that relies on externally defined printf-like formats.
	*
	* \tparam FormatProvider Needs to provide a static method const char * format() that returns
	* a format-string for _exactly_ 6 parameters, either positional or referenced by its 1-based index:
	*  - First argument: The log type as string.
	*  - Second argument: the timestamp as string.
	*  - Third argument: The human-readable log message.
	*  - Fourth argument: The source of the log message as string.
	*  - Fifth argument: The file that the log message originates from as string.
	*  - Sixth argument: The line that the log message originates from as std::size_t.
	*/
	template<typename FormatProvider>
	class PrintfLogFormatter : public Logger::AbstractFormatter {
	public:

		/**
		* \brief Make the format provider known to the outside world.
		*/
		typedef FormatProvider format_provider;

		/** \brief Default look-up table for Level <-> short string conversion. */
		static Logger::lut_type SHORT_LEVEL_LUT;

		/** \brief Formats the supplied log record as string. */
		std::string handle( const Logger::Record & record ) {
			static boost::format format( FormatProvider::format() );
			return(  
				(	format % 
				SHORT_LEVEL_LUT.left.find( record.logLevel() )->second % 
				boost::posix_time::to_simple_string( record.timestamp() ) % 
				record.message() % 
				( record.source().empty() ? "Unknown" : record.source() ) %
				( record.file().empty() ? "Unknown" : record.file() ) % 
				record.line() 
				).str()
			);
		}
	};

	/**
	* \brief Initializes the lut for log level <-> short string conversion.
	*/
	template<typename T>
	Logger::lut_type PrintfLogFormatter<T>::SHORT_LEVEL_LUT = boost::assign::list_of< Logger::lut_type::relation >
		( Logger::DEBUG_LEVEL, "DD" )
		( Logger::INFO_LEVEL, "II" )
		( Logger::WARNING_LEVEL, "WW" )
		( Logger::ERROR_LEVEL, "EE" );

	/** \brief Tags a plain text formatter. */
	typedef PrintfLogFormatter< detail::PlainTextPrintfFormatProvider > PlainTextLogFormatter;

	/** \brief Tags an XML formatter. */
	typedef PrintfLogFormatter< detail::XmlPrintfFormatProvider > XmlLogFormatter;

	/** \brief Tags a JSON formatter. */
	typedef PrintfLogFormatter< detail::JsonPrintfFormatProvider > JsonLogFormatter;

	/** \brief Make the plain text formatter known to the formatter factory. */
	ANNOUNCE_LOG_FORMATTER( PlainTextLogFormatter, LogFormatterFactory );

	/** \brief Make the xml formatter known to the formatter factory. */
	ANNOUNCE_LOG_FORMATTER( XmlLogFormatter, LogFormatterFactory );

	/** \brief Make the json formatter known to the formatter factory. */
	ANNOUNCE_LOG_FORMATTER( JsonLogFormatter, LogFormatterFactory );

}

#endif