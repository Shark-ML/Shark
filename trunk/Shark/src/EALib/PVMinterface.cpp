//===========================================================================
/*!
*  \file PVMinterface.cpp
*
*  \brief This file serves as a dummy interface for the use of PVM with the EALib.
*
*  \throw SharkException
* 
*  \author  S. Wiegand
* 
*  \par Copyright (c) 1999-2003:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*  \par Project:
*      EALib
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


#include <EALib/PVMinterface.h>
#include <SharkDefs.h>


#ifndef PVM_EXISTS


int pvm_joingroup(char* group)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_mytid()
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_gettid(char* group , int inum)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_initsend(int encoding)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_send(int tid     , int msgtag)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_recv(int tid     , int msgtag)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_pkdouble(double *dp  , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_pkuint(unsigned int *up    , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_pkint(int *ip     , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_pkbyte(char *xp     , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_upkdouble(double *dp  , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_upkuint(unsigned int *up    , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_upkint(int *ip     , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_upkbyte(char *xp     , int nitem , int stride)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_probe(int tid     , int msgtag)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_barrier(char* group , int count)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_exit()
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_getinst(char* group , int tid)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}

int pvm_bufinfo(int bufid, int *bytes, int *mstag, int *tid)
{
	fprintf(stderr, "WARNING: Using EALib/PVMinterface\n");
	exit(EXIT_FAILURE);
}


#endif
