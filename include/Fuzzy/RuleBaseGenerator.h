
/**
 * \file RuleBaseGenerator.h
 *
 * \brief Reads and writes rule bases from and to XML-files.
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

#ifndef __RULEBASEGENERATOR_H__
#define __RULEBASEGENERATOR_H__

#include <Fuzzy/RuleBase.h>

/**
 * \brief Reads a RuleBase from XML file
 * 
 * @param descriptionFile the XML file containing the RuleBase
 * @return the read RuleBase
 */
RuleBase build_rule_base_from_xml( const std::string & descriptionFile );

/**
 * \brief Writes a RuleBase into a XML file
 * 
 * @param descriptionFile the output file
 * @param rb the RuleBase to write
 * @return true, in case of success
 */
bool save_rule_base_to_xml( const std::string & descriptionFile, RuleBase & rb );

// Building a rule-base from fcl.
//RuleBase build_rule_base_from_fcl( const std::string & descriptionFile );
#endif
