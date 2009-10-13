
/**
 * \file FuzzyException.cpp
 *
 * \brief Implementation of an exception for error handling
 * 
 * \author Marc Nunkesser
 */

/* $log$ */

#include "Fuzzy/FuzzyException.h"

FuzzyException::FuzzyException(int Number, const std::string Text):
		ErrorNumber(Number), ErrorText(Text) {};

