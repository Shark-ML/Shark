//===========================================================================
/*!
 *  \file QuadraticProgram.h
 *
 *  \brief Quadratic programming for Support Vector Machines
 *
 *
 *  \par
 *  This file provides the following interfaces:
 *  <ul>
 *	<li>A quadratic matrix interface</li>
 *	<li>Several special matrices based on kernels and other matrices</li>
 *	<li>A quadtaric matrix cache</li>
 *	<li>A quadratic program solver for a special #SVM related family of problems</li>
 *  </ul>
 *  All methods are specifically tuned toward the solution of
 *  quadratic programs occuring in support vector machines and
 *  related kernel methods. There is no support the general
 *  quadratic programs. Refer to the #QuadraticProgram class
 *  documentation for details.
 *
 *
 *  \author  T. Glasmachers
 *  \date	2007
 *
 *  \par Copyright (c) 1999-2007:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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


#ifndef _QuadraticProgram_H_
#define _QuadraticProgram_H_

#include <ReClaM/KernelMatrices.h>
#include <ReClaM/QpSvmDecomp.h>
#include <ReClaM/QpBoxDecomp.h>
#include <ReClaM/QpMcDecomp.h>
#include <ReClaM/QpMcStzDecomp.h>
#include <ReClaM/QpEbCsDecomp.h>

#endif
