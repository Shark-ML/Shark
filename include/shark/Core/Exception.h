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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
		* \param [in] func Name of the Function the error appeared in.
		*/
		Exception( 
			const std::string & what = "unknown reason",
			const std::string & file = "unknown",
			unsigned int line = 0,
			const std::string & func = "function"
		): m_what( what )
		, m_file( file )
		, m_line( line )
		, m_func( func )
		{
			m_message="["+m_file+"::"+m_func+","+std::to_string(line)+"] " + what; 
		}

		/**
		* \brief Default d'tor.
		*/
		~Exception( ) throw() {}

		/**
		* \brief Accesses the description of the exception.
		*/
		inline const char* what() const throw() {
			return m_message.c_str();
		}

		/**
		* \brief Accesses the name of the file the exception occurred in.
		*/
		inline const std::string & file() const {
			return m_file;
		}

		/**
		* \brief Accesses the line of the file the exception occured in.
		*/
		inline unsigned int line() const {
			return m_line;
		}

	protected:
		std::string m_what; ///< Description of the exception.
		std::string m_file; ///< File name the exception occurred in.
		unsigned int m_line; ///< Line of file the exception occurred in.
		std::string m_func; ///< Function name the exception occured in
		std::string m_message; ///< complete error message
		
	};

}

//MSVC  does not have the __func__ so __FUNCTION__ has to be used instead.
#ifdef _MSC_VER
#define SHARKEXCEPTION(message) shark::Exception(message, __FILE__, __LINE__, __FUNCTION__)
#else
#define SHARKEXCEPTION(message) shark::Exception(message, __FILE__, __LINE__, __func__)
#endif
// some handy macros for special types of checks,
// throwing standard error messages
#ifndef NDEBUG
#define RANGE_CHECK(cond) assert(cond)
#define SIZE_CHECK(cond) assert(cond)
#define SHARK_ASSERT(cond) assert(cond)
#else
#define RANGE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define SIZE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define SHARK_ASSERT(cond) do { (void)sizeof(cond); } while (false)
#endif
#define SHARK_RUNTIME_CHECK(cond, message) do { if (!(cond)) throw SHARKEXCEPTION(message);} while (false)

#endif // SHARK_CORE_EXCEPTION_H
