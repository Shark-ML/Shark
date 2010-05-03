//===========================================================================
/*!  
 *  \file eigen.cpp
 * 
 *  \brief eigenvalues of arbitrary matrices
 * 
 *  \par Copyright (c) 1998-2000:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
#include <Array/Array.h>
#include <LinAlg/LinAlg.h>
#include <cmath>





//--------------------------------------------------------------
void Exc(Array<double> &A, Array<double> &D, int j, int k, int l, int m)
{  // used by Balanc
	int ii; double f;
	int n = A.dim(0);
	D(m) = (double) j;
	if (j != m)
	{
		for (ii = 0; ii < k + 1; ii++)
		{
			f = A(ii, j);
			A(ii, j) = A(ii, m);
			A(ii, m) = f;
		}
		//for (ii = l; ii < n + 1; ii++)
    for (ii = 1; ii < n; ii++) // b.roeschies 12/13/2007: changed because ii goes beyond array bounds (ii == A.dim(0) + 1 in the last step).
		{
			f = A(j, ii);
			A(j, ii) = A(m, ii);
			A(m, ii) = f;
		}
	}
}
//--------------------------------------------------------------

/**********************************************************************
* Balanc:   From Pascal version translated from ALGOL by J-P Dumont   *
*                                  Basic version by J-P Moreau        *
* ------------------------------------------------------------------- *
* Ref.: "Handbook for automatic computation Volume II Linear Algebra  *
*        J.H. Wilkinson et C.Reinsch, Springer Verlag, 1971"          *
* ------------------------------------------------------------------- *
*   This procedure balances the elements of a non symmetric square    *
* matrix. Given a matrix n x n stored in a table of size n x n, that  *
* matrix is replaced by a balanced matrix having the same eigenvalues *
* A symmetric matrix is not affected by this procedure.               *
* ------------------------------------------------------------------- *
* Inputs:                                                             *
*       n       Order of non symmetric square matrix A (n x n)        *
*       A       Table n*n storing the  elements of A                  *
* ------------------------------------------------------------------- *
* Outputs:                                                            *
*       A       The original A matrix is replaced by the balanced     *
*               matrix. However, it is possible to recover exactly    *
*               the original matrix.                                  *
*  low,hi       Two integers such as A(i, j) = 0 if                   *
*               (1) i > j  AND                                        *
*               (2) j =1,...,Low-1 OU i = hi+1,...,n                  *
*       D       Vector (1..n) containing enough information to trace  *
*               the done permutations and the used scale factors.     *
**********************************************************************/

//===========================================================================
/*!
 *  \brief This function balances the elements of a non symmetric square    
 *  matrix. Given a matrix n x n stored in a table of size n x n, that  
 *  matrix is replaced by a balanced matrix having the same eigenvalues 
 *  A symmetric matrix is not affected by this procedure
 *
 *
 *      \param  A     \f$ n \times n \f$  matrix
 *      \param  low   first nonzero row. 
 *      \param  high  last nonzero row. 
 *      \param  D     Vector (1..n) containing enough information to trace 
 *                    the done permutations and the used scale factors.
 *
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that
 *             	\em amatA is not a square matrix
 *
 *
 *  \par Changes
 *     previously 'eigensymm', renamed by S. Wiegand 2003/10/01
 *
 *  \par Status
 *      stable
 *
 */
void Balanc(Array<double> &A, Array<double> &D, int &low, int &hi)
{
//LABELS: Iteration(1100), L1(1200), L2(1300), L3(1400), L4(1500)

	int i, j, k, l, m;
	int n = A.dim(0);
	double b, b2, c, f, g, r, s;
	int noconv;  // boolean 0 or 1

	b = 2;       // Floating point basis of used CPU
	b2 = b * b;
	l = 0;
	k = n - 1;
	//Search lines isolating an eigenvalue and shift downwards
L1:
	for (j = k; j >= 0; j--)
	{
		r = 0;
		for (i = 0; i < j; i++)  r = r + fabs(A(j, i));
		for (i = j + 1; i < k + 1; i++)  r = r + fabs(A(j, i));
		if (r == 0)
		{
			m = k; Exc(A, D, j, k, l, m);
			k--;
			goto L1;
		}
	}
	//Search columns isolating an eigenvalue and shift to the left
L2:
	for (j = l; j < k + 1; j++)
	{
		c = 0;
		for (i = l; i < j; i++)  c = c + fabs(A(i, j));
		for (i = j + 1; i < k + 1; i++)  c = c + fabs(A(i, j));
		if (c == 0)
		{
			m = l; Exc(A, D, j, k, l, m);
			l++;
			goto L2;
		}
	}
	//Now balance submatrix from line l to line k
	low = l;
	hi  = k;
	for (i = 0; i < k + 1; i++)  D(i) = 1;
Iterations:  noconv = 0;
	for (i = l; i < k + 1; i++)
	{
		c = 0;
		r = c;
		for (j = l; j < i; j++)
		{
			c = c + fabs(A(j, i));
			r = r + fabs(A(i, j));
		}
		for (j = i + 1; j < k + 1; j++)
		{
			c = c + fabs(A(j, i));
			r = r + fabs(A(i, j));
		}
		g = r / b;
		f = 1;
		s = c + r;
L3: if (c < g)
		{
			f = f * b;
			c = c * b2;
			goto L3;
		}
		g = r * b;
L4: if (c >= g)
		{
			f = f / b;
			c = c / b2;
			goto L4;
		}

		// Balancing the elements of submatrix
		if ((c + r) / f < 0.95*s)
		{
			g = 1. / f;
			D(i) *= f;
			noconv = 1;
			// assuming an error?
			for (j = l; j < n; j++)  A(i, j) *= g;
			for (j = 0; j < k + 1; j++)  A(j, i) *= f;
		}
	}
	if (noconv == 1) goto Iterations;

} //Balanc()

/*
{-------------------------Documentation-------------------------------}
{               VAR A: Square_Matrix; n: INTEGER);                    }
{ Reduction of a non symmetric real matrix to Hessenberg form by the  }
{ elimination method. Matrix A (n x n), stored in a table of size     }
{ Maxc x Maxc, is replaced by a superior Hessenberg matrix having the }
{ same eigenvalues. It is recommanded to call previously the Balanc   }
{ procedure. In output,the Hessenberg matrix has elements A(i, j) with}
{ i<=j+1. The elements for i>j+1, that in theory equal zero, are      }
{ actually filled with random (not used) values.                      }
{---------------------------------------------------------------------}
{ From Pascal version by J.P.DUMONT - Extracted from reference:       }
{       "William H.PRESS, Brian P.FLANNERY, Saul A.TEUKOLSKY AND      }
{        William T.VETTERLING                                         }
{              N U M E R I C A L  R E C I P E S                       }
{              The Art OF Scientific Computing                        }
{              CAMBRIDGE UNIVERSITY PRESS 1987"                       }
{                                         C++ version by J-P Moreau   }
{/////////////////////////////////////////////////////////////////////}
*/

//===========================================================================
/*!
 *  \brief Reduction of a non symmetric real matrix to Hessenberg form by the  
 *  elimination method. Matrix A (n x n), stored in a table of size     
 *  Maxc x Maxc, is replaced by a superior Hessenberg matrix having the 
 *  same eigenvalues. It is recommanded to call previously the Balanc   
 *  procedure. In output,the Hessenberg matrix has elements A(i, j) with
 *  i<=j+1. The elements for i>j+1, that in theory equal zero, are      
 *  actually filled with random (not used) values
 *
 *
 *      \param  A     \f$ n \times n \f$ matrix
 *
 *
 *  \par Status
 *      stable
 *
 */

void ElmHes(Array<double> &A)
{
	int i, j, m;
	double x, y;
	int n = A.dim(0);

	if (n > 2)
	{
		for (m = 1; m < n - 1; m++)
		{
			x = 0;
			i = m;
			for (j = m; j < n; j++)
			{
				if (fabs(A(j, m - 1)) > fabs(x))
				{
					x = A(j, m - 1);
					i = j;
				}
			} // j loop
			if (i != m)
			{
				for (j = m - 1; j < n; j++)
				{
					y = A(i, j);
					A(i, j) = A(m, j);
					A(m, j) = y;
				}
				for (j = 0; j < n; j++)
				{
					y = A(j, i);
					A(j, i) = A(j, m);
					A(j, m) = y;
				}
			} //if i != m
			if (x != 0)
			{
				for (i = m + 1; i < n; i++)
				{
					y = A(i, m - 1);
					if (y != 0)
					{
						y /= x;
						A(i, m - 1) = y;
						for (j = m; j < n; j++)  A(i, j) -= y * A(m, j);
						for (j = 0; j < n; j++)  A(j, m) += y * A(j, i);
					} // if y
				} // i loop
			} // if x
		} // m loop
	} //if n>2
	//Put lower triangle to zero (optional)
	for (i = 0; i < n; i++)
		for (j = 0; j < i - 1; j++)
			A(i, j) = 0;
} // ElmHes()

//===========================================================================
/*!
 *  \brief Computes eigenvalues of an upper Hessenberg matrix.
 *
 *
 *      \param  h     \f$ n \times n \f$ upper Hessenberg matrix
 *      \param  low   first nonzero row. 
 *      \param  high  last nonzero row. 
 *      \param	vr    real parts of eigenvalues
 *	\param  vi    imaginary parts of eigenvalues
 *      \param	cnt    iteration counter
 *
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that
 *             	\em amatA is not a square matrix
 *
 *
 *  \par Changes
 *     previously 'eigensymm', renamed by S. Wiegand 2003/10/01
 *
 *  \par Status
 *      stable
 *
 */
static int hqr           // compute eigenvalues ......................*/
(int     low,         /* first nonzero row ...........*/
 int     high,        /* last nonzero row ............*/
 Array<double>  &h,   /* Hessenberg matrix ...........*/
 Array<double>  &wr,  /* Real parts of eigenvalues....*/
 Array<double>  &wi,  /* Imaginary parts of evalues ..*/
 Array<int>     &cnt  /* Iteration counter ...........*/
)
/*====================================================================*
 *                                                                    *
 *  hqr computes the eigenvalues of an n * n upper Hessenberg matrix  *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Input parameters:                                                *
 *   ================                                                 *
 *      low      int low;                                             *
 *      high     int high; see  balance                               *
 *      h        upper  Hessenberg matrix (output of ElmHes)          *
 *                                                                    *
 *   Output parameters:                                               *
 *   ==================                                               *
 *      wr       double   wr(n);                                      *
 *               double part of the n eigenvalues.                    *
 *      wi       double   wi(n);                                      *
 *               Imaginary parts of the eigenvalues                   *
 *      cnt      int cnt(n);                                          *
 *               vector of iterations used for each eigenvalue.       *
 *               For a complex conjugate eigenvalue pair the second   *
 *               entry is negative.                                   *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      =   0    all ok                                               *
 *      = 4xx    Iteration maximum exceeded when computing evalue xx  *
 *      =  99    zero  matrix                                         *
 *                                                                    *
 *====================================================================*
 * Reference: "Numerical Algorithms with C by G. Engeln-Mueller and   *
 *             F. Uhlig, Springer-Verlag, 1996"                       *
 *====================================================================*/
{
	int  i, j, MAXIT = 30;
	int  na, en, iter, k, l, m;
	double p = 0, q = 0, r = 0, s, t, w, x, y, z;
	int n = h.dim(0);
	double MACH_EPS = 1e-16;


	for (i = 0; i < n; i++)
		if (i < low || i > high)
		{
			wr(i) = h(i, i);
			wi(i) = 0;
			cnt(i) = 0;
		}

	en = high;
	t = 0;

	while (en >= low)
	{
		iter = 0;
		na = en - 1;

		for (; ;)
		{
			for (l = en; l > low; l--)             /* search for small      */
				if (fabs(h(l, l - 1)) <=               /* subdiagonal element   */
						MACH_EPS *(fabs(h(l - 1, l - 1)) + fabs(h(l, l))))  break;

			x = h(en, en);
			if (l == en)                            /* found one evalue     */
			{
				wr(en) = h(en, en) = x + t;
				wi(en) = 0;
				cnt(en) = iter;
				en--;
				break;
			}

			y = h(na, na);
			w = h(en, na) * h(na, en);

			if (l == na)                            /* found two evalues    */
			{
				p = (y - x) * 0.5;
				q = p * p + w;
				z = sqrt(fabs(q));
				x = h(en, en) = x + t;
				h(na, na) = y + t;
				cnt(en) = -iter;
				cnt(na) = iter;
				if (q >= 0)
				{                                     /* double eigenvalues     */
					z = (p < 0) ? (p - z) : (p + z);
					wr(na) = x + z;
					wr(en) = s = x - w / z;
					wi(na) = wi(en) = 0;
					x = h(en, na);
					r = sqrt(x * x + z * z);
				}  /* end if (q >= 0) */
				else                                  /* pair of complex      */
				{                                     /* conjugate evalues    */
					wr(na) = wr(en) = x + p;
					wi(na) =   z;
					wi(en) = - z;
				}

				en -= 2;
				break;
			}  /* end if (l == na) */

			if (iter >= MAXIT)
			{
				cnt(en) = MAXIT + 1;
				return (en);                         /* MAXIT Iterations     */
			}

			if ((iter != 0) && (iter % 10 == 0))
			{
				t += x;
				for (i = low; i <= en; i++) h(i, i) -= x;
				s = fabs(h(en, na)) + fabs(h(na, en - 2));
				x = y = (double)0.75 * s;
				w = - (double)0.4375 * s * s;
			}

			iter ++;

			for (m = en - 2; m >= l; m--)
			{
				z = h(m, m);
				r = x - z;
				s = y - z;
				p = (r * s - w) / h(m + 1, m) + h(m, m + 1);
				q = h(m + 1, m + 1) - z - r - s;
				r = h(m + 2, m + 1);
				s = fabs(p) + fabs(q) + fabs(r);
				p /= s;
				q /= s;
				r /= s;
				if (m == l) break;
				if (fabs(h(m, m - 1)) *(fabs(q) + fabs(r)) <=
						MACH_EPS * fabs(p)
						*(fabs(h(m - 1, m - 1)) + fabs(z) + fabs(h(m + 1, m + 1))))
					break;
			}

			for (i = m + 2; i <= en; i++) h(i, i - 2) = 0;
			for (i = m + 3; i <= en; i++) h(i, i - 3) = 0;

			for (k = m; k <= na; k++)
			{
				if (k != m)             /* double  QR step, for rows l to en  */
				{                       /* and columns m to en                */
					p = h(k, k - 1);
					q = h(k + 1, k - 1);
					r = (k != na) ? h(k + 2, k - 1) : 0;
					x = fabs(p) + fabs(q) + fabs(r);
					if (x == 0) continue;                  /*  next k        */
					p /= x;
					q /= x;
					r /= x;
				}
				s = sqrt(p * p + q * q + r * r);
				if (p < 0) s = -s;

				if (k != m) h(k, k - 1) = -s * x;
				else if (l != m)
					h(k, k - 1) = -h(k, k - 1);
				p += s;
				x = p / s;
				y = q / s;
				z = r / s;
				q /= p;
				r /= p;

				for (j = k; j < n; j++)               /* modify rows          */
				{
					p = h(k, j) + q * h(k + 1, j);
					if (k != na)
					{
						p += r * h(k + 2, j);
						h(k + 2, j) -= p * z;
					}
					h(k + 1, j) -= p * y;
					h(k, j)   -= p * x;
				}

				j = (k + 3 < en) ? (k + 3) : en;
				for (i = 0; i <= j; i++)              /* modify columns       */
				{
					p = x * h(i, k) + y * h(i, k + 1);
					if (k != na)
					{
						p += z * h(i, k + 2);
						h(i, k + 2) -= p * r;
					}
					h(i, k + 1) -= p * q;
					h(i, k)   -= p;
				}
			}    /* end k          */
		}    /* end for ( ; ;) */
	}    /* while (en >= low)                      All evalues found    */
	return (0);
} //hqr()



//===========================================================================
/*!
 *  \brief Calculates the eigenvalues of an arbitrary  matrix "amatA" by 
 *  balancing the elements of the matrix, then reducing it to an upper 
 *  Hessenberg matrix and finally calculating the eigenvalues ot the reduced form. 
 *
 *  Given a \f$ n \times n \f$ matrix \em A, this function
 *  calculates the eigenvalues \f$ \lambda \f$ 
 *  defined as (\em x is the corresponding but not calculated eigenvector)
 *
 *  \f$
 *      Ax = \lambda x
 *  \f$
 *
 *  where \em x is a one-column matrix and the matrix multiplication
 *  is used for \em A and \em x.
 *
 *      \param  amatA \f$ n \times n \f$ matrix. 
 *      \param	vr    real parts of eigenvalues
 *	\param  vi    imaginary parts of eigenvalues
 *
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that
 *             	\em amatA is not a square matrix
 *
 *
 *  \par Changes
 *     previously 'eigensymm', renamed by S. Wiegand 2003/10/01
 *
 *  \par Status
 *      stable
 *
 */
void eigen(Array<double> amatA, Array<double> & vr, Array<double> & vi)
{
	// eigen computes the eigenvalues of an arbitrary (non-symmetric) matrix amatA and stores the unsorted eigenvalues in vi and vr (vr real parts, vi complex parts)
	int n = amatA.dim(0);
	Array<double> B, D;
	B.resize(n, n, false);
	D.resize(n, false);
	Array<int> niter;
	niter.resize(n, false);
	int low, hi, rc;
	low = 0;
	hi  = n - 1;

	for (int i = 0;i < n;i++)
		for (int j = 0;j < n;j++)
			B(i, j) = amatA(i, j);

	//balance matrix
	Balanc(B, D, low, hi);
	//transform in Hessenberg matrix
	ElmHes(B);
	//   call QR algorithm, calculate eigenvalues
	rc = hqr(low, hi, B, vr, vi, niter);
}
