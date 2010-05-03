//===========================================================================
/*!
 *  \file UTest.h
 *
 *  \brief   Mann-Whitney U-Test
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#ifndef _UTest_H_
#define _UTest_H_


#include <vector>


//!
//! \brief perform a Mann-Whitney U-Test
//!
//! \par
//! This function performs a Mann-Whitney U-Test.
//! There are one-sided and two-sided tests
//! available. In all cases the null-hypothesis is
//! equality of the distributions of both samples.
//!
//! \par
//! The alternatives are as follows:
//! <table>
//!   <tr>
//!     <td><b>test type</b></th><td><b>alternative</b></td>
//!   </tr>
//!   <tr>
//!     <td>A left of B</th><td>distribution of sample-A produces smaller values than distribution of sample-B</td>
//!     <td>B left of A</th><td>distribution of sample-A produces larger values than distribution of sample-B</td>
//!     <td>two-sided</th><td>distribution of sample-A produces smaller or larger values than distribution of sample-B</td>
//!   </tr>
//! </table>
//!
//! \par
//! As usual, a normal approximation is used for
//! large sample sizes.
//!
//! \param  sampleA       values in the first sample
//! \param  sampleB       values in the second sample
//! \param  alternative   type of alternative (UTEST_LEFTSIDED, UTEST_RIGHTSIDED, UTEST_TWOSIDED)
//! \param  p_twosided    p-value: largest type 1 error for which the two-sided test discards the null-hypothesis
//! \param  p_A_leftOf_B  p-value: largest type 1 error for which the one-sided test discards the null-hypothesis
//! \param  p_B_leftOf_A  p-value: largest type 1 error for which the one-sided test discards the null-hypothesis
//!
void UTest(const std::vector<double>& sampleA, const std::vector<double>& sampleB, double& p_twosided, double& p_A_leftOf_B, double& p_B_leftOf_A);


#endif
