/**
 * \file FuzzyException.h
 *
 * \brief Implementation of an exception for error handling
 * 
 * \author Marc Nunkesser
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */


#ifndef FUZZYEXCEPTION_H
#define FUZZYEXCEPTION_H

#include <exception>
#include <string>
#include <iostream>


/**
 * \brief Implementation of an exception for error handling.
 * 
 * Use this class to throw Fuzzy-related exceptions. 
 */
class FuzzyException : public std::exception {
public:
	
/**
 * \brief Constructor.
 * @param Number the error number, should be unique
 * @param Text the error text
 */
	FuzzyException(int Number=0, const std::string Text="");
	
/**
 * \brief Returns the error text.
 *
 * @return the error text
 */
     inline std::string getErrorText() {
    	 return ErrorText;
     };

/**
 * \brief Returns the error number.
 *
 * @return the error number
 */
	inline int getErrorNumber() {
		return ErrorNumber;
	};

/**
 * \brief Prints the error.
 */
	inline void printError() {
		std::cerr<<"An Error occured:\n Error Number:"<<ErrorNumber<<std::endl<<"ErrorText:"<<ErrorText<<std::endl;
	};

/**
 * \brief Destructor.
 */
	virtual ~FuzzyException() throw() {};
private:
	int ErrorNumber;
	std::string ErrorText;
};

#endif
