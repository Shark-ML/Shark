/**
 * \mainpage
 *
 * This module offers strucures and methods needed for the
 * implementation of Fuzzy Logic and Fuzzy Control systems. The base
 * classes implementing fuzzy sets are FuzzySet and NDimFS, which
 * model one and n-dimensional fuzzy sets, respectively. Several
 * different types of fuzzy sets are already implemented. There is
 * possibility to define additional fuzzy sets by combining two fuzzy
 * sets (see: ComposedFS and ComposedNDimFS) or by direct definition
 * of membership function and support by the user is given (see
 * CustomizedFS).
 *
 * The concepts of a linguist variable, like <i>befuddlement</i>, and the
 * associated linguistic terms, like <i>drunken</i> or <i>sober</i>, are implemented
 * in the classes LinguisticVariable and LinguisticTerm. For every
 * subclass of FuzzySet there's a corresponding subclass of linguistic
 * term, for instance BellFS and BellLT (a linguistic term is
 * basically a named fuzzy set).
 *
 * Fuzzy if-then rules can either be of the canonical type Rule (with
 * a fuzzy set on the RHS) or a SugenoRule. Different inference
 * machines (namely the MamdaniIM and SugenoIM) are available as
 * implementation of the general interface InferenceMachine. The class
 * CustomIM provides an inference method that can be defined by the
 * user. In this context the class Implication (and its base class
 * FuzzyRelation) are needed. The RuleBase on which the inference
 * machine works can either be build manually or read from a XML-file
 * (see: RuleBaseGenerator.h).
 */


/**
 * \file FuzzyIncludes.h
 *
 * \brief Mother of all Fuzzy includes
 * 
 * \authors Marc Nunkesser, Copyright (c) 2008, Marc Nunkesser
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



#ifdef __SOLARIS__
#include <climits>
#endif
#ifdef __LINUX__
#include <float.h>
#endif
#ifndef FUZZYINCLUDES_H
#define FUZZYINCLUDES_H
#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/Rule.h>
#include <Fuzzy/RuleBase.h>
#include <Fuzzy/InferenceMachine.h>
#include <Fuzzy/InfinityFS.h>
#include <Fuzzy/InfinityLT.h>
#include <Fuzzy/SugenoIM.h>
#include <Fuzzy/SugenoRule.h>
#include <Fuzzy/SingletonLT.h>
#include <Fuzzy/CustomizedFS.h>
#include <Fuzzy/BellFS.h>
#include <Fuzzy/BellLT.h>
#include <Fuzzy/MamdaniIM.h>
#include <Fuzzy/CustomIM.h>
#include <Fuzzy/TrapezoidFS.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/TriangularLT.h>
#include <Fuzzy/SigmoidalLT.h>
#endif
