/**
 * \file TrapezoidLT.cpp
 *
 * \brief LinguisticTerm with trapezoid membership function
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


#include <Fuzzy/TrapezoidLT.h>

TrapezoidLT::TrapezoidLT(const std::string name,
                         const RCPtr<LinguisticVariable>& parent,
                         double                           p1,
                         double                           p2,
                         double                           p3,
                         double                           p4):
		LinguisticTerm(name,parent),TrapezoidFS(p1,p2,p3,p4) {}
