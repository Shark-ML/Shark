//===========================================================================
/*!
 *  \file PVMinterface.h
 *
 *  \brief This file serves as a dummy interface for the use of PVM with the EALib.
 *
 *
 *  \author  S. Wiegand
 *  \date    2003-01-01
 *
 *  \par Copyright (c) 1999-2003:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      EALib
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of EALib. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#ifndef PVMINTERFACE_H
#define PVMINTERFACE_H

#define PvmDataDefault  0

/*! This file serves as a dummy interface for the use of PVM with the EALib. It
    reimplements basic PVM methods with default return error values due to calls
    implemented in Individual.h.
*/

typedef unsigned uint;

#ifdef PVM_EXISTS

	#include <stdio.h>
	#include <pvm3.h>
	
#else

	#include <stdlib.h>
	#include <iostream>
	
	#ifdef __cplusplus
		extern "C"
		{
	#endif
	
	int pvm_joingroup(char*);
	int pvm_mytid();
	int pvm_gettid(char*  , int);
	
	int pvm_initsend(int);
	int pvm_send(int    , int);
	int pvm_recv(int    , int);
	
	int pvm_pkdouble(double*, int, int);
	int pvm_pkuint(uint*  , int, int);
	int pvm_pkint(int*   , int, int);
	int pvm_pkbyte(char*   , int, int);
	
	int pvm_upkdouble(double*, int, int);
	int pvm_upkuint(uint*  , int, int);
	int pvm_upkint(int*   , int, int);
	int pvm_upkbyte(char*   , int, int);
	
	int pvm_probe(int    , int);
	int pvm_barrier(char*  , int);
	int pvm_exit();
	
	int pvm_getinst(char*  , int);
	int pvm_bufinfo(int, int*, int*, int*);
	
	#ifdef __cplusplus
		}
	#endif

#endif

#endif

