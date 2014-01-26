/*!
 * 
 *
 * \brief       Exception
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef SHARK_CORE_EXCEPTION_H
#define SHARK_CORE_EXCEPTION_H

#include <string>
#include <exception>

namespace shark {

	/**
	* \brief Top-level exception class of the shark library.
	*/
	class Exception : public std::exception {
	public:
		/**
		* \brief Default c'tor.
		* \param [in] what String that describes the exception.
		* \param [in] file Filename the function that has thrown the exception resides in.
		* \param [in] line Line of file that has thrown the exception.
		*/
		Exception( const std::string & what = std::string(), const std::string & file = std::string(), unsigned int line = 0 ) : m_what( what ),
			m_file( file ),
			m_line( line ) {
		}

		/**
		* \brief Default d'tor.
		*/
		~Exception( ) throw() {}

		/**
		* \brief Accesses the description of the exception.
		*/
		inline const char* what() const throw() {
			return m_what.c_str();
		}

		/**
		* \brief Accesses the name of the file the exception occurred in.
		*/
		inline const std::string & file() const {
			return( m_file );
		}

		/**
		* \brief Accesses the line of the file the exception occured in.
		*/
		inline unsigned int line() const {
			return( m_line );
		}

	protected:
		std::string m_what; ///< Description of the exception.
		std::string m_file; ///< File name the exception occurred in.
		unsigned int m_line; ///< Line of file the exception occurred in.
	};

}

/**
* \brief Convenience macro that creates an instance of class shark::exception,
* injecting file and line information automatically.
*/
#define SHARKEXCEPTION(message) shark::Exception(message, __FILE__, __LINE__)

/// Break the execution and throw exception with @a message in case of predefined @a unexpectedCondition is true
/// @note This should not be replaced by SHARK_CHECK as we need always evaluate @a unexpectedCondition
inline void THROW_IF(bool unexpectedCondition, const std::string& message)
{
	if (unexpectedCondition)
		throw SHARKEXCEPTION(message);
}

// some handy macros for special types of checks,
// throwing standard error messages
#if defined(DEBUG) || defined(_DEBUG)|| !(defined(NDEBUG)||defined(RELEASE))
#define RANGE_CHECK(cond) do { if (!(cond)) throw SHARKEXCEPTION("range check error: "#cond); } while (false)
#define SIZE_CHECK(cond) do { if (!(cond)) throw SHARKEXCEPTION("size mismatch: "#cond); } while (false)
#define TYPE_CHECK(cond) do { if (!(cond)) throw SHARKEXCEPTION("type mismatch: "#cond); } while (false)
#define IO_CHECK(cond) do { if (!(cond)) throw SHARKEXCEPTION("I/O error "); } while (false)
#define SHARK_ASSERT(cond) do { if (!(cond)) throw SHARKEXCEPTION("assertion failed: "#cond); } while (false)
#define SHARK_CHECK(cond, error) do { if (!(cond)) throw SHARKEXCEPTION(error); } while (false)
#else
#define RANGE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define SIZE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define TYPE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define IO_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define SHARK_ASSERT(cond) do { (void)sizeof(cond); } while (false)
#define SHARK_CHECK(cond, error) do { (void)sizeof(cond); (void)sizeof(error);} while (false)
#endif

#endif // SHARK_CORE_EXCEPTION_H
