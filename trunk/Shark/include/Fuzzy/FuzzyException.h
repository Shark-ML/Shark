/**
 * \file FuzzyException.h
 *
 * \brief Implementation of an exception for error handling
 * 
 * \author Marc Nunkesser
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
