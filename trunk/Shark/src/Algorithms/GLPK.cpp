//===========================================================================
/*!
 *  \file GLPK.cpp
 *
 *  \brief Shark adaptation of the GNU Linear Programming Kit (GLPK)
 *
 *
 *  \par
 *  This file contains a minimal adaptation of GLPK, the GNU
 *  Linear Programming Kit, version 4.45. The functionality has
 *  been reduced to the simplex solver, it has been included in
 *  the shark namespace, type casts have been made compatible
 *  with C++, and the error reporting mechanism has been made
 *  compatible with Shark standards. All changes were done by
 *  Tobias Glasmachers, 2010-2011.<br>
 *  The original copyright notice is found below:
 *
 *  \par
 *  This code is part of GLPK (GNU Linear Programming Kit).<br>
 *  <br>
 *  Copyright (C) 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,<br>
 *  2009, 2010 Andrew Makhorin, Department for Applied Informatics,<br>
 *  Moscow Aviation Institute, Moscow, Russia. All rights reserved.<br>
 *  E-mail: &lt;mao@gnu.org&gt;.<br>
 *  <br>
 *  GLPK is free software: you can redistribute it and/or modify it<br>
 *  under the terms of the GNU General Public License as published by<br>
 *  the Free Software Foundation, either version 3 of the License, or<br>
 *  (at your option) any later version.<br>
 *  <br>
 *  GLPK is distributed in the hope that it will be useful, but WITHOUT<br>
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY<br>
 *  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public<br>
 *  License for more details.<br>
 *  <br>
 *  You should have received a copy of the GNU General Public License<br>
 *  along with GLPK. If not, see &lt;http://www.gnu.org/licenses/&gt;.<br>
 */
//===========================================================================

#include<shark/Core/Exception.h>
#include <shark/Algorithms/LP/GLPK.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#include <shark/SharkDefs.h>

#ifdef _WIN32
#define snprintf _snprintf
#endif

namespace shark {
namespace glpk {


#define xmalloc(sz) malloc(sz)
#define xcalloc(sz1, sz2) calloc(sz1, sz2)
#define xfree(ptr) free(ptr)

#ifndef GLP_LONG_DEFINED
#define GLP_LONG_DEFINED
struct glp_long { int lo, hi; };
/* long integer data type */
#endif

void xerror(const char* msg)
{
	throw SHARKEXCEPTION(msg);
}
void xerror(const char* format, int a1)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}
void xerror(const char* format, double a1)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}
void xerror(const char* format, void* a1)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}
void xerror(const char* format, int a1, int a2)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1, a2);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}
void xerror(const char* format, int a1, double a2)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1, a2);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}
void xerror(const char* format, int a1, int a2, int a3)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1, a2, a3);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}
void xerror(const char* format, int a1, int a2, int a3, int a4)
{
	char buffer[1024];
	snprintf(buffer, 1023, format, a1, a2, a3, a4);
	char buffer2[1030];
	strcpy(buffer2, "[LP] ");
	strcat(buffer2, buffer);
	throw SHARKEXCEPTION(buffer2);
}

#define xassert(cond) SHARK_CHECK(cond, "[LP] GLPK error");


glp_long xtime() { glp_long ret; ret.lo = ret.hi = 0; return ret; }
double xdifftime(glp_long t1, glp_long t2) { return 0.0; }


////////// START DMC HEADER

#define DMP_BLK_SIZE 8000
/* size of memory blocks, in bytes, allocated for memory pools */

struct DMP
{     /* dynamic memory pool */
      void *avail[32];
      /* avail[k], 0 <= k <= 31, is a pointer to the first available
         (free) cell of (k+1)*8 bytes long; in the beginning of each
         free cell there is a pointer to another free cell of the same
         length */
      void *block;
      /* pointer to the most recently allocated memory block; in the
         beginning of each allocated memory block there is a pointer to
         the previously allocated memory block */
      int used;
      /* number of bytes used in the most recently allocated memory
         block */
      glp_long count;
      /* number of atoms which are currently in use */
};

#define dmp_create_pool _glp_dmp_create_pool
DMP *dmp_create_pool(void);
/* create dynamic memory pool */

#define dmp_get_atom _glp_dmp_get_atom
void *dmp_get_atom(DMP *pool, int size);
/* get free atom from dynamic memory pool */

#define dmp_free_atom _glp_dmp_free_atom
void dmp_free_atom(DMP *pool, void *atom, int size);
/* return atom to dynamic memory pool */

#define dmp_in_use _glp_dmp_in_use
glp_long dmp_in_use(DMP *pool);
/* determine how many atoms are still in use */

#define dmp_delete_pool _glp_dmp_delete_pool
void dmp_delete_pool(DMP *pool);
/* delete dynamic memory pool */

////////// END DMC HEADER


////////// START DMC

#define align_datasize(size) ((((size) + 7) / 8) * 8)
/* 8 bytes is sufficient in both 32- and 64-bit environments */


DMP *dmp_create_pool(void)
{     DMP *pool;
      int k;
      pool = (DMP*)xmalloc(sizeof(DMP));
      for (k = 0; k <= 31; k++) pool->avail[k] = NULL;
      pool->block = NULL;
      pool->used = DMP_BLK_SIZE;
      pool->count.lo = pool->count.hi = 0;
      return pool;
}

void *dmp_get_atom(DMP *pool, int size)
{     void *atom;
      int k;
      if (!(1 <= size && size <= 256))
         xerror("dmp_get_atom: size = %d; invalid atom size", size);
      /* adjust the size to provide the proper data alignment */
      size = align_datasize(size);
      /* adjust the size to make it multiple of 8 bytes, if needed */
      size = ((size + 7) / 8) * 8;
      /* determine the corresponding list of free cells */
      k = size / 8 - 1;
      xassert(0 <= k && k <= 31);
      /* obtain a free atom */
      if (pool->avail[k] == NULL)
      {  /* the list of free cells is empty */
         if (pool->used + size > DMP_BLK_SIZE)
         {  /* allocate a new memory block */
            void *block = xmalloc(DMP_BLK_SIZE);
            *(void **)block = pool->block;
            pool->block = block;
            pool->used = align_datasize(sizeof(void *));
         }
         /* place the atom in the current memory block */
         atom = (char *)pool->block + pool->used;
         pool->used += size;
      }
      else
      {  /* obtain the atom from the list of free cells */
         atom = pool->avail[k];
         pool->avail[k] = *(void **)atom;
      }
      memset(atom, '?', size);
      /* increase the number of atoms which are currently in use */
      pool->count.lo++;
      if (pool->count.lo == 0) pool->count.hi++;
      return atom;
}

void dmp_free_atom(DMP *pool, void *atom, int size)
{     int k;
      if (!(1 <= size && size <= 256))
         xerror("dmp_free_atom: size = %d; invalid atom size", size);
      if (pool->count.lo == 0 && pool->count.hi == 0)
         xerror("dmp_free_atom: pool allocation error");
      /* adjust the size to provide the proper data alignment */
      size = align_datasize(size);
      /* adjust the size to make it multiple of 8 bytes, if needed */
      size = ((size + 7) / 8) * 8;
      /* determine the corresponding list of free cells */
      k = size / 8 - 1;
      xassert(0 <= k && k <= 31);
      /* return the atom to the list of free cells */
      *(void **)atom = pool->avail[k];
      pool->avail[k] = atom;
      /* decrease the number of atoms which are currently in use */
      pool->count.lo--;
      if ((unsigned int)pool->count.lo == 0xFFFFFFFF) pool->count.hi--;
      return;
}

glp_long dmp_in_use(DMP *pool)
{     return
         pool->count;
}

void dmp_delete_pool(DMP *pool)
{     while (pool->block != NULL)
      {  void *block = pool->block;
         pool->block = *(void **)block;
         xfree(block);
      }
      xfree(pool);
      return;
}

////////// END DMC


////////// START GLPK HEADER

/* optimization direction flag: */
#define GLP_MIN            1  /* minimization */
#define GLP_MAX            2  /* maximization */

/* type of auxiliary/structural variable: */
#define GLP_FR             1  /* free variable */
#define GLP_LO             2  /* variable with lower bound */
#define GLP_UP             3  /* variable with upper bound */
#define GLP_DB             4  /* double-bounded variable */
#define GLP_FX             5  /* fixed variable */

/* status of auxiliary/structural variable: */
#define GLP_BS             1  /* basic variable */
#define GLP_NL             2  /* non-basic variable on lower bound */
#define GLP_NU             3  /* non-basic variable on upper bound */
#define GLP_NF             4  /* non-basic free variable */
#define GLP_NS             5  /* non-basic fixed variable */

/* solution status: */
#define GLP_UNDEF          1  /* solution is undefined */
#define GLP_FEAS           2  /* solution is feasible */
#define GLP_INFEAS         3  /* solution is infeasible */
#define GLP_NOFEAS         4  /* no feasible solution exists */
#define GLP_OPT            5  /* solution is optimal */
#define GLP_UNBND          6  /* solution is unbounded */

/* enable/disable flag: */
#define GLP_ON             1  /* enable something */
#define GLP_OFF            0  /* disable something */

/* return codes: */
#define GLP_EBADB       0x01  /* invalid basis */
#define GLP_ESING       0x02  /* singular matrix */
#define GLP_ECOND       0x03  /* ill-conditioned matrix */
#define GLP_EBOUND      0x04  /* invalid bounds */
#define GLP_EFAIL       0x05  /* solver failed */
#define GLP_EOBJLL      0x06  /* objective lower limit reached */
#define GLP_EOBJUL      0x07  /* objective upper limit reached */
#define GLP_EITLIM      0x08  /* iteration limit exceeded */
#define GLP_ETMLIM      0x09  /* time limit exceeded */
#define GLP_ENOPFS      0x0A  /* no primal feasible solution */
#define GLP_ENODFS      0x0B  /* no dual feasible solution */
#define GLP_EROOT       0x0C  /* root LP optimum not provided */
#define GLP_ESTOP       0x0D  /* search terminated by application */
#define GLP_EMIPGAP     0x0E  /* relative mip gap tolerance reached */
#define GLP_ENOFEAS     0x0F  /* no primal/dual feasible solution */
#define GLP_ENOCVG      0x10  /* no convergence */
#define GLP_EINSTAB     0x11  /* numerical instability */
#define GLP_EDATA       0x12  /* invalid data */
#define GLP_ERANGE      0x13  /* result out of range */

/* kind of structural variable: */
#define GLP_CV             1  /* continuous variable */
#define GLP_IV             2  /* integer variable */
#define GLP_BV             3  /* binary variable */

struct glp_bfcp
{     /* basis factorization control parameters */
      int msg_lev;            /* (reserved) */
      int type;               /* factorization type: */
#define GLP_BF_FT          1  /* LUF + Forrest-Tomlin */
#define GLP_BF_BG          2  /* LUF + Schur compl. + Bartels-Golub */
#define GLP_BF_GR          3  /* LUF + Schur compl. + Givens rotation */
      int lu_size;            /* luf.sv_size */
      double piv_tol;         /* luf.piv_tol */
      int piv_lim;            /* luf.piv_lim */
      int suhl;               /* luf.suhl */
      double eps_tol;         /* luf.eps_tol */
      double max_gro;         /* luf.max_gro */
      int nfs_max;            /* fhv.hh_max */
      double upd_tol;         /* fhv.upd_tol */
      int nrs_max;            /* lpf.n_max */
      int rs_size;            /* lpf.v_size */
      double foo_bar[38];     /* (reserved) */
};

struct glp_smcp
{     /* simplex method control parameters */
      int msg_lev;            /* message level: */
#define GLP_MSG_OFF        0  /* no output */
#define GLP_MSG_ERR        1  /* warning and error messages only */
#define GLP_MSG_ON         2  /* normal output */
#define GLP_MSG_ALL        3  /* full output */
#define GLP_MSG_DBG        4  /* debug output */
      int meth;               /* simplex method option: */
#define GLP_PRIMAL         1  /* use primal simplex */
#define GLP_DUALP          2  /* use dual; if it fails, use primal */
#define GLP_DUAL           3  /* use dual simplex */
      int pricing;            /* pricing technique: */
#define GLP_PT_STD      0x11  /* standard (Dantzig rule) */
#define GLP_PT_PSE      0x22  /* projected steepest edge */
      int r_test;             /* ratio test technique: */
#define GLP_RT_STD      0x11  /* standard (textbook) */
#define GLP_RT_HAR      0x22  /* two-pass Harris' ratio test */
      double tol_bnd;         /* spx.tol_bnd */
      double tol_dj;          /* spx.tol_dj */
      double tol_piv;         /* spx.tol_piv */
      double obj_ll;          /* spx.obj_ll */
      double obj_ul;          /* spx.obj_ul */
      int it_lim;             /* spx.it_lim */
      int tm_lim;             /* spx.tm_lim (milliseconds) */
      int out_frq;            /* spx.out_frq */
      int out_dly;            /* spx.out_dly (milliseconds) */
      int presolve;           /* enable/disable using LP presolver */
      double foo_bar[36];     /* (reserved) */
};

////////// END GLPK HEADER


////////// START LUF HEADER

struct LUF
{     /* LU-factorization of a square matrix */
      int n_max;
      /* maximal value of n (increased automatically, if necessary) */
      int n;
      /* the order of matrices A, F, V, P, Q */
      int valid;
      /* the factorization is valid only if this flag is set */
      /*--------------------------------------------------------------*/
      /* matrix F in row-wise format */
      int *fr_ptr; /* int fr_ptr[1+n_max]; */
      /* fr_ptr[i], i = 1,...,n, is a pointer to the first element of
         i-th row in SVA */
      int *fr_len; /* int fr_len[1+n_max]; */
      /* fr_len[i], i = 1,...,n, is the number of elements in i-th row
         (except unity diagonal element) */
      /*--------------------------------------------------------------*/
      /* matrix F in column-wise format */
      int *fc_ptr; /* int fc_ptr[1+n_max]; */
      /* fc_ptr[j], j = 1,...,n, is a pointer to the first element of
         j-th column in SVA */
      int *fc_len; /* int fc_len[1+n_max]; */
      /* fc_len[j], j = 1,...,n, is the number of elements in j-th
         column (except unity diagonal element) */
      /*--------------------------------------------------------------*/
      /* matrix V in row-wise format */
      int *vr_ptr; /* int vr_ptr[1+n_max]; */
      /* vr_ptr[i], i = 1,...,n, is a pointer to the first element of
         i-th row in SVA */
      int *vr_len; /* int vr_len[1+n_max]; */
      /* vr_len[i], i = 1,...,n, is the number of elements in i-th row
         (except pivot element) */
      int *vr_cap; /* int vr_cap[1+n_max]; */
      /* vr_cap[i], i = 1,...,n, is the capacity of i-th row, i.e.
         maximal number of elements which can be stored in the row
         without relocating it, vr_cap[i] >= vr_len[i] */
      double *vr_piv; /* double vr_piv[1+n_max]; */
      /* vr_piv[p], p = 1,...,n, is the pivot element v[p,q] which
         corresponds to a diagonal element of matrix U = P*V*Q */
      /*--------------------------------------------------------------*/
      /* matrix V in column-wise format */
      int *vc_ptr; /* int vc_ptr[1+n_max]; */
      /* vc_ptr[j], j = 1,...,n, is a pointer to the first element of
         j-th column in SVA */
      int *vc_len; /* int vc_len[1+n_max]; */
      /* vc_len[j], j = 1,...,n, is the number of elements in j-th
         column (except pivot element) */
      int *vc_cap; /* int vc_cap[1+n_max]; */
      /* vc_cap[j], j = 1,...,n, is the capacity of j-th column, i.e.
         maximal number of elements which can be stored in the column
         without relocating it, vc_cap[j] >= vc_len[j] */
      /*--------------------------------------------------------------*/
      /* matrix P */
      int *pp_row; /* int pp_row[1+n_max]; */
      /* pp_row[i] = j means that P[i,j] = 1 */
      int *pp_col; /* int pp_col[1+n_max]; */
      /* pp_col[j] = i means that P[i,j] = 1 */
      /* if i-th row or column of matrix F is i'-th row or column of
         matrix L, or if i-th row of matrix V is i'-th row of matrix U,
         then pp_row[i'] = i and pp_col[i] = i' */
      /*--------------------------------------------------------------*/
      /* matrix Q */
      int *qq_row; /* int qq_row[1+n_max]; */
      /* qq_row[i] = j means that Q[i,j] = 1 */
      int *qq_col; /* int qq_col[1+n_max]; */
      /* qq_col[j] = i means that Q[i,j] = 1 */
      /* if j-th column of matrix V is j'-th column of matrix U, then
         qq_row[j] = j' and qq_col[j'] = j */
      /*--------------------------------------------------------------*/
      /* the Sparse Vector Area (SVA) is a set of locations used to
         store sparse vectors representing rows and columns of matrices
         F and V; each location is a doublet (ind, val), where ind is
         an index, and val is a numerical value of a sparse vector
         element; in the whole each sparse vector is a set of adjacent
         locations defined by a pointer to the first element and the
         number of elements; these pointer and number are stored in the
         corresponding matrix data structure (see above); the left part
         of SVA is used to store rows and columns of matrix V, and its
         right part is used to store rows and columns of matrix F; the
         middle part of SVA contains free (unused) locations */
      int sv_size;
      /* the size of SVA, in locations; all locations are numbered by
         integers 1, ..., n, and location 0 is not used; if necessary,
         the SVA size is automatically increased */
      int sv_beg, sv_end;
      /* SVA partitioning pointers:
         locations from 1 to sv_beg-1 belong to the left part
         locations from sv_beg to sv_end-1 belong to the middle part
         locations from sv_end to sv_size belong to the right part
         the size of the middle part is (sv_end - sv_beg) */
      int *sv_ind; /* sv_ind[1+sv_size]; */
      /* sv_ind[k], 1 <= k <= sv_size, is the index field of k-th
         location */
      double *sv_val; /* sv_val[1+sv_size]; */
      /* sv_val[k], 1 <= k <= sv_size, is the value field of k-th
         location */
      /*--------------------------------------------------------------*/
      /* in order to efficiently defragment the left part of SVA there
         is a doubly linked list of rows and columns of matrix V, where
         rows are numbered by 1, ..., n, while columns are numbered by
         n+1, ..., n+n, that allows uniquely identifying each row and
         column of V by only one integer; in this list rows and columns
         are ordered by ascending their pointers vr_ptr and vc_ptr */
      int sv_head;
      /* the number of leftmost row/column */
      int sv_tail;
      /* the number of rightmost row/column */
      int *sv_prev; /* int sv_prev[1+n_max+n_max]; */
      /* sv_prev[k], k = 1,...,n+n, is the number of a row/column which
         precedes k-th row/column */
      int *sv_next; /* int sv_next[1+n_max+n_max]; */
      /* sv_next[k], k = 1,...,n+n, is the number of a row/column which
         succedes k-th row/column */
      /*--------------------------------------------------------------*/
      /* working segment (used only during factorization) */
      double *vr_max; /* int vr_max[1+n_max]; */
      /* vr_max[i], 1 <= i <= n, is used only if i-th row of matrix V
         is active (i.e. belongs to the active submatrix), and is the
         largest magnitude of elements in i-th row; if vr_max[i] < 0,
         the largest magnitude is not known yet and should be computed
         by the pivoting routine */
      /*--------------------------------------------------------------*/
      /* in order to efficiently implement Markowitz strategy and Duff
         search technique there are two families {R[0], R[1], ..., R[n]}
         and {C[0], C[1], ..., C[n]}; member R[k] is the set of active
         rows of matrix V, which have k non-zeros, and member C[k] is
         the set of active columns of V, which have k non-zeros in the
         active submatrix (i.e. in the active rows); each set R[k] and
         C[k] is implemented as a separate doubly linked list */
      int *rs_head; /* int rs_head[1+n_max]; */
      /* rs_head[k], 0 <= k <= n, is the number of first active row,
         which has k non-zeros */
      int *rs_prev; /* int rs_prev[1+n_max]; */
      /* rs_prev[i], 1 <= i <= n, is the number of previous row, which
         has the same number of non-zeros as i-th row */
      int *rs_next; /* int rs_next[1+n_max]; */
      /* rs_next[i], 1 <= i <= n, is the number of next row, which has
         the same number of non-zeros as i-th row */
      int *cs_head; /* int cs_head[1+n_max]; */
      /* cs_head[k], 0 <= k <= n, is the number of first active column,
         which has k non-zeros (in the active rows) */
      int *cs_prev; /* int cs_prev[1+n_max]; */
      /* cs_prev[j], 1 <= j <= n, is the number of previous column,
         which has the same number of non-zeros (in the active rows) as
         j-th column */
      int *cs_next; /* int cs_next[1+n_max]; */
      /* cs_next[j], 1 <= j <= n, is the number of next column, which
         has the same number of non-zeros (in the active rows) as j-th
         column */
      /* (end of working segment) */
      /*--------------------------------------------------------------*/
      /* working arrays */
      int *flag; /* int flag[1+n_max]; */
      /* integer working array */
      double *work; /* double work[1+n_max]; */
      /* floating-point working array */
      /*--------------------------------------------------------------*/
      /* control parameters */
      int new_sva;
      /* new required size of the sparse vector area, in locations; set
         automatically by the factorizing routine */
      double piv_tol;
      /* threshold pivoting tolerance, 0 < piv_tol < 1; element v[i,j]
         of the active submatrix fits to be pivot if it satisfies to the
         stability criterion |v[i,j]| >= piv_tol * max |v[i,*]|, i.e. if
         it is not very small in the magnitude among other elements in
         the same row; decreasing this parameter gives better sparsity
         at the expense of numerical accuracy and vice versa */
      int piv_lim;
      /* maximal allowable number of pivot candidates to be considered;
         if piv_lim pivot candidates have been considered, the pivoting
         routine terminates the search with the best candidate found */
      int suhl;
      /* if this flag is set, the pivoting routine applies a heuristic
         proposed by Uwe Suhl: if a column of the active submatrix has
         no eligible pivot candidates (i.e. all its elements do not
         satisfy to the stability criterion), the routine excludes it
         from futher consideration until it becomes column singleton;
         in many cases this allows reducing the time needed for pivot
         searching */
      double eps_tol;
      /* epsilon tolerance; each element of the active submatrix, whose
         magnitude is less than eps_tol, is replaced by exact zero */
      double max_gro;
      /* maximal allowable growth of elements of matrix V during all
         the factorization process; if on some eliminaion step the ratio
         big_v / max_a (see below) becomes greater than max_gro, matrix
         A is considered as ill-conditioned (assuming that the pivoting
         tolerance piv_tol has an appropriate value) */
      /*--------------------------------------------------------------*/
      /* some statistics */
      int nnz_a;
      /* the number of non-zeros in matrix A */
      int nnz_f;
      /* the number of non-zeros in matrix F (except diagonal elements,
         which are not stored) */
      int nnz_v;
      /* the number of non-zeros in matrix V (except its pivot elements,
         which are stored in a separate array) */
      double max_a;
      /* the largest magnitude of elements of matrix A */
      double big_v;
      /* the largest magnitude of elements of matrix V appeared in the
         active submatrix during all the factorization process */
      int rank;
      /* estimated rank of matrix A */
};

/* return codes: */
#define LUF_ESING    1  /* singular matrix */
#define LUF_ECOND    2  /* ill-conditioned matrix */

#define luf_create_it _glp_luf_create_it
LUF *luf_create_it(void);
/* create LU-factorization */

#define luf_defrag_sva _glp_luf_defrag_sva
void luf_defrag_sva(LUF *luf);
/* defragment the sparse vector area */

#define luf_enlarge_row _glp_luf_enlarge_row
int luf_enlarge_row(LUF *luf, int i, int cap);
/* enlarge row capacity */

#define luf_enlarge_col _glp_luf_enlarge_col
int luf_enlarge_col(LUF *luf, int j, int cap);
/* enlarge column capacity */

#define luf_factorize _glp_luf_factorize
int luf_factorize(LUF *luf, int n, int (*col)(void *info, int j,
      int ind[], double val[]), void *info);
/* compute LU-factorization */

#define luf_f_solve _glp_luf_f_solve
void luf_f_solve(LUF *luf, int tr, double x[]);
/* solve system F*x = b or F'*x = b */

#define luf_v_solve _glp_luf_v_solve
void luf_v_solve(LUF *luf, int tr, double x[]);
/* solve system V*x = b or V'*x = b */

#define luf_a_solve _glp_luf_a_solve
void luf_a_solve(LUF *luf, int tr, double x[]);
/* solve system A*x = b or A'*x = b */

#define luf_delete_it _glp_luf_delete_it
void luf_delete_it(LUF *luf);
/* delete LU-factorization */

////////// END LUF HEADER


////////// START LUF

/* CAUTION: DO NOT CHANGE THE LIMIT BELOW */

#define N_MAX 100000000 /* = 100*10^6 */
/* maximal order of the original matrix */

LUF *luf_create_it(void)
{     LUF *luf;
      luf = (LUF*)xmalloc(sizeof(LUF));
      luf->n_max = luf->n = 0;
      luf->valid = 0;
      luf->fr_ptr = luf->fr_len = NULL;
      luf->fc_ptr = luf->fc_len = NULL;
      luf->vr_ptr = luf->vr_len = luf->vr_cap = NULL;
      luf->vr_piv = NULL;
      luf->vc_ptr = luf->vc_len = luf->vc_cap = NULL;
      luf->pp_row = luf->pp_col = NULL;
      luf->qq_row = luf->qq_col = NULL;
      luf->sv_size = 0;
      luf->sv_beg = luf->sv_end = 0;
      luf->sv_ind = NULL;
      luf->sv_val = NULL;
      luf->sv_head = luf->sv_tail = 0;
      luf->sv_prev = luf->sv_next = NULL;
      luf->vr_max = NULL;
      luf->rs_head = luf->rs_prev = luf->rs_next = NULL;
      luf->cs_head = luf->cs_prev = luf->cs_next = NULL;
      luf->flag = NULL;
      luf->work = NULL;
      luf->new_sva = 0;
      luf->piv_tol = 0.10;
      luf->piv_lim = 4;
      luf->suhl = 1;
      luf->eps_tol = 1e-15;
      luf->max_gro = 1e+10;
      luf->nnz_a = luf->nnz_f = luf->nnz_v = 0;
      luf->max_a = luf->big_v = 0.0;
      luf->rank = 0;
      return luf;
}

void luf_defrag_sva(LUF *luf)
{     int n = luf->n;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vr_cap = luf->vr_cap;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *vc_cap = luf->vc_cap;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int *sv_next = luf->sv_next;
      int sv_beg = 1;
      int i, j, k;
      /* skip rows and columns, which do not need to be relocated */
      for (k = luf->sv_head; k != 0; k = sv_next[k])
      {  if (k <= n)
         {  /* i-th row of the matrix V */
            i = k;
            if (vr_ptr[i] != sv_beg) break;
            vr_cap[i] = vr_len[i];
            sv_beg += vr_cap[i];
         }
         else
         {  /* j-th column of the matrix V */
            j = k - n;
            if (vc_ptr[j] != sv_beg) break;
            vc_cap[j] = vc_len[j];
            sv_beg += vc_cap[j];
         }
      }
      /* relocate other rows and columns in order to gather all unused
         locations in one continuous extent */
      for (k = k; k != 0; k = sv_next[k])
      {  if (k <= n)
         {  /* i-th row of the matrix V */
            i = k;
            memmove(&sv_ind[sv_beg], &sv_ind[vr_ptr[i]],
               vr_len[i] * sizeof(int));
            memmove(&sv_val[sv_beg], &sv_val[vr_ptr[i]],
               vr_len[i] * sizeof(double));
            vr_ptr[i] = sv_beg;
            vr_cap[i] = vr_len[i];
            sv_beg += vr_cap[i];
         }
         else
         {  /* j-th column of the matrix V */
            j = k - n;
            memmove(&sv_ind[sv_beg], &sv_ind[vc_ptr[j]],
               vc_len[j] * sizeof(int));
            memmove(&sv_val[sv_beg], &sv_val[vc_ptr[j]],
               vc_len[j] * sizeof(double));
            vc_ptr[j] = sv_beg;
            vc_cap[j] = vc_len[j];
            sv_beg += vc_cap[j];
         }
      }
      /* set new pointer to the beginning of the free part */
      luf->sv_beg = sv_beg;
      return;
}

int luf_enlarge_row(LUF *luf, int i, int cap)
{     int n = luf->n;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vr_cap = luf->vr_cap;
      int *vc_cap = luf->vc_cap;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int *sv_prev = luf->sv_prev;
      int *sv_next = luf->sv_next;
      int ret = 0;
      int cur, k, kk;
      xassert(1 <= i && i <= n);
      xassert(vr_cap[i] < cap);
      /* if there are less than cap free locations, defragment SVA */
      if (luf->sv_end - luf->sv_beg < cap)
      {  luf_defrag_sva(luf);
         if (luf->sv_end - luf->sv_beg < cap)
         {  ret = 1;
            goto done;
         }
      }
      /* save current capacity of the i-th row */
      cur = vr_cap[i];
      /* copy existing elements to the beginning of the free part */
      memmove(&sv_ind[luf->sv_beg], &sv_ind[vr_ptr[i]],
         vr_len[i] * sizeof(int));
      memmove(&sv_val[luf->sv_beg], &sv_val[vr_ptr[i]],
         vr_len[i] * sizeof(double));
      /* set new pointer and new capacity of the i-th row */
      vr_ptr[i] = luf->sv_beg;
      vr_cap[i] = cap;
      /* set new pointer to the beginning of the free part */
      luf->sv_beg += cap;
      /* now the i-th row starts in the rightmost location among other
         rows and columns of the matrix V, so its node should be moved
         to the end of the row/column linked list */
      k = i;
      /* remove the i-th row node from the linked list */
      if (sv_prev[k] == 0)
         luf->sv_head = sv_next[k];
      else
      {  /* capacity of the previous row/column can be increased at the
            expense of old locations of the i-th row */
         kk = sv_prev[k];
         if (kk <= n) vr_cap[kk] += cur; else vc_cap[kk-n] += cur;
         sv_next[sv_prev[k]] = sv_next[k];
      }
      if (sv_next[k] == 0)
         luf->sv_tail = sv_prev[k];
      else
         sv_prev[sv_next[k]] = sv_prev[k];
      /* insert the i-th row node to the end of the linked list */
      sv_prev[k] = luf->sv_tail;
      sv_next[k] = 0;
      if (sv_prev[k] == 0)
         luf->sv_head = k;
      else
         sv_next[sv_prev[k]] = k;
      luf->sv_tail = k;
done: return ret;
}

int luf_enlarge_col(LUF *luf, int j, int cap)
{     int n = luf->n;
      int *vr_cap = luf->vr_cap;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *vc_cap = luf->vc_cap;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int *sv_prev = luf->sv_prev;
      int *sv_next = luf->sv_next;
      int ret = 0;
      int cur, k, kk;
      xassert(1 <= j && j <= n);
      xassert(vc_cap[j] < cap);
      /* if there are less than cap free locations, defragment SVA */
      if (luf->sv_end - luf->sv_beg < cap)
      {  luf_defrag_sva(luf);
         if (luf->sv_end - luf->sv_beg < cap)
         {  ret = 1;
            goto done;
         }
      }
      /* save current capacity of the j-th column */
      cur = vc_cap[j];
      /* copy existing elements to the beginning of the free part */
      memmove(&sv_ind[luf->sv_beg], &sv_ind[vc_ptr[j]],
         vc_len[j] * sizeof(int));
      memmove(&sv_val[luf->sv_beg], &sv_val[vc_ptr[j]],
         vc_len[j] * sizeof(double));
      /* set new pointer and new capacity of the j-th column */
      vc_ptr[j] = luf->sv_beg;
      vc_cap[j] = cap;
      /* set new pointer to the beginning of the free part */
      luf->sv_beg += cap;
      /* now the j-th column starts in the rightmost location among
         other rows and columns of the matrix V, so its node should be
         moved to the end of the row/column linked list */
      k = n + j;
      /* remove the j-th column node from the linked list */
      if (sv_prev[k] == 0)
         luf->sv_head = sv_next[k];
      else
      {  /* capacity of the previous row/column can be increased at the
            expense of old locations of the j-th column */
         kk = sv_prev[k];
         if (kk <= n) vr_cap[kk] += cur; else vc_cap[kk-n] += cur;
         sv_next[sv_prev[k]] = sv_next[k];
      }
      if (sv_next[k] == 0)
         luf->sv_tail = sv_prev[k];
      else
         sv_prev[sv_next[k]] = sv_prev[k];
      /* insert the j-th column node to the end of the linked list */
      sv_prev[k] = luf->sv_tail;
      sv_next[k] = 0;
      if (sv_prev[k] == 0)
         luf->sv_head = k;
      else
         sv_next[sv_prev[k]] = k;
      luf->sv_tail = k;
done: return ret;
}

void reallocate(LUF *luf, int n)
{     int n_max = luf->n_max;
      luf->n = n;
      if (n <= n_max) goto done;
      if (luf->fr_ptr != NULL) xfree(luf->fr_ptr);
      if (luf->fr_len != NULL) xfree(luf->fr_len);
      if (luf->fc_ptr != NULL) xfree(luf->fc_ptr);
      if (luf->fc_len != NULL) xfree(luf->fc_len);
      if (luf->vr_ptr != NULL) xfree(luf->vr_ptr);
      if (luf->vr_len != NULL) xfree(luf->vr_len);
      if (luf->vr_cap != NULL) xfree(luf->vr_cap);
      if (luf->vr_piv != NULL) xfree(luf->vr_piv);
      if (luf->vc_ptr != NULL) xfree(luf->vc_ptr);
      if (luf->vc_len != NULL) xfree(luf->vc_len);
      if (luf->vc_cap != NULL) xfree(luf->vc_cap);
      if (luf->pp_row != NULL) xfree(luf->pp_row);
      if (luf->pp_col != NULL) xfree(luf->pp_col);
      if (luf->qq_row != NULL) xfree(luf->qq_row);
      if (luf->qq_col != NULL) xfree(luf->qq_col);
      if (luf->sv_prev != NULL) xfree(luf->sv_prev);
      if (luf->sv_next != NULL) xfree(luf->sv_next);
      if (luf->vr_max != NULL) xfree(luf->vr_max);
      if (luf->rs_head != NULL) xfree(luf->rs_head);
      if (luf->rs_prev != NULL) xfree(luf->rs_prev);
      if (luf->rs_next != NULL) xfree(luf->rs_next);
      if (luf->cs_head != NULL) xfree(luf->cs_head);
      if (luf->cs_prev != NULL) xfree(luf->cs_prev);
      if (luf->cs_next != NULL) xfree(luf->cs_next);
      if (luf->flag != NULL) xfree(luf->flag);
      if (luf->work != NULL) xfree(luf->work);
      luf->n_max = n_max = n + 100;
      luf->fr_ptr = (int*)xcalloc(1+n_max, sizeof(int));
      luf->fr_len = (int*)xcalloc(1+n_max, sizeof(int));
      luf->fc_ptr = (int*)xcalloc(1+n_max, sizeof(int));
      luf->fc_len = (int*)xcalloc(1+n_max, sizeof(int));
      luf->vr_ptr = (int*)xcalloc(1+n_max, sizeof(int));
      luf->vr_len = (int*)xcalloc(1+n_max, sizeof(int));
      luf->vr_cap = (int*)xcalloc(1+n_max, sizeof(int));
      luf->vr_piv = (double*)xcalloc(1+n_max, sizeof(double));
      luf->vc_ptr = (int*)xcalloc(1+n_max, sizeof(int));
      luf->vc_len = (int*)xcalloc(1+n_max, sizeof(int));
      luf->vc_cap = (int*)xcalloc(1+n_max, sizeof(int));
      luf->pp_row = (int*)xcalloc(1+n_max, sizeof(int));
      luf->pp_col = (int*)xcalloc(1+n_max, sizeof(int));
      luf->qq_row = (int*)xcalloc(1+n_max, sizeof(int));
      luf->qq_col = (int*)xcalloc(1+n_max, sizeof(int));
      luf->sv_prev = (int*)xcalloc(1+n_max+n_max, sizeof(int));
      luf->sv_next = (int*)xcalloc(1+n_max+n_max, sizeof(int));
      luf->vr_max = (double*)xcalloc(1+n_max, sizeof(double));
      luf->rs_head = (int*)xcalloc(1+n_max, sizeof(int));
      luf->rs_prev = (int*)xcalloc(1+n_max, sizeof(int));
      luf->rs_next = (int*)xcalloc(1+n_max, sizeof(int));
      luf->cs_head = (int*)xcalloc(1+n_max, sizeof(int));
      luf->cs_prev = (int*)xcalloc(1+n_max, sizeof(int));
      luf->cs_next = (int*)xcalloc(1+n_max, sizeof(int));
      luf->flag = (int*)xcalloc(1+n_max, sizeof(int));
      luf->work = (double*)xcalloc(1+n_max, sizeof(double));
done: return;
}

int initialize(LUF *luf, int (*col)(void *info, int j, int rn[],
      double aj[]), void *info)
{     int n = luf->n;
      int *fc_ptr = luf->fc_ptr;
      int *fc_len = luf->fc_len;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vr_cap = luf->vr_cap;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *vc_cap = luf->vc_cap;
      int *pp_row = luf->pp_row;
      int *pp_col = luf->pp_col;
      int *qq_row = luf->qq_row;
      int *qq_col = luf->qq_col;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int *sv_prev = luf->sv_prev;
      int *sv_next = luf->sv_next;
      double *vr_max = luf->vr_max;
      int *rs_head = luf->rs_head;
      int *rs_prev = luf->rs_prev;
      int *rs_next = luf->rs_next;
      int *cs_head = luf->cs_head;
      int *cs_prev = luf->cs_prev;
      int *cs_next = luf->cs_next;
      int *flag = luf->flag;
      double *work = luf->work;
      int ret = 0;
      int i, i_ptr, j, j_beg, j_end, k, len, nnz, sv_beg, sv_end, ptr;
      double big, val;
      /* free all locations of the sparse vector area */
      sv_beg = 1;
      sv_end = luf->sv_size + 1;
      /* (row-wise representation of the matrix F is not initialized,
         because it is not used at the factorization stage) */
      /* build the matrix F in column-wise format (initially F = I) */
      for (j = 1; j <= n; j++)
      {  fc_ptr[j] = sv_end;
         fc_len[j] = 0;
      }
      /* clear rows of the matrix V; clear the flag array */
      for (i = 1; i <= n; i++)
         vr_len[i] = vr_cap[i] = 0, flag[i] = 0;
      /* build the matrix V in column-wise format (initially V = A);
         count non-zeros in rows of this matrix; count total number of
         non-zeros; compute largest of absolute values of elements */
      nnz = 0;
      big = 0.0;
      for (j = 1; j <= n; j++)
      {  int *rn = pp_row;
         double *aj = work;
         /* obtain j-th column of the matrix A */
         len = col(info, j, rn, aj);
         if (!(0 <= len && len <= n))
            xerror("luf_factorize: j = %d; len = %d; invalid column len"
               "gth", j, len);
         /* check for free locations */
         if (sv_end - sv_beg < len)
         {  /* overflow of the sparse vector area */
            ret = 1;
            goto done;
         }
         /* set pointer to the j-th column */
         vc_ptr[j] = sv_beg;
         /* set length of the j-th column */
         vc_len[j] = vc_cap[j] = len;
         /* count total number of non-zeros */
         nnz += len;
         /* walk through elements of the j-th column */
         for (ptr = 1; ptr <= len; ptr++)
         {  /* get row index and numerical value of a[i,j] */
            i = rn[ptr];
            val = aj[ptr];
            if (!(1 <= i && i <= n))
               xerror("luf_factorize: i = %d; j = %d; invalid row index", i, j);
            if (flag[i])
               xerror("luf_factorize: i = %d; j = %d; duplicate element not allowed", i, j);
            if (val == 0.0)
               xerror("luf_factorize: i = %d; j = %d; zero element not allowed", i, j);
            /* add new element v[i,j] = a[i,j] to j-th column */
            sv_ind[sv_beg] = i;
            sv_val[sv_beg] = val;
            sv_beg++;
            /* big := max(big, |a[i,j]|) */
            if (val < 0.0) val = - val;
            if (big < val) big = val;
            /* mark non-zero in the i-th position of the j-th column */
            flag[i] = 1;
            /* increase length of the i-th row */
            vr_cap[i]++;
         }
         /* reset all non-zero marks */
         for (ptr = 1; ptr <= len; ptr++) flag[rn[ptr]] = 0;
      }
      /* allocate rows of the matrix V */
      for (i = 1; i <= n; i++)
      {  /* get length of the i-th row */
         len = vr_cap[i];
         /* check for free locations */
         if (sv_end - sv_beg < len)
         {  /* overflow of the sparse vector area */
            ret = 1;
            goto done;
         }
         /* set pointer to the i-th row */
         vr_ptr[i] = sv_beg;
         /* reserve locations for the i-th row */
         sv_beg += len;
      }
      /* build the matrix V in row-wise format using representation of
         this matrix in column-wise format */
      for (j = 1; j <= n; j++)
      {  /* walk through elements of the j-th column */
         j_beg = vc_ptr[j];
         j_end = j_beg + vc_len[j] - 1;
         for (k = j_beg; k <= j_end; k++)
         {  /* get row index and numerical value of v[i,j] */
            i = sv_ind[k];
            val = sv_val[k];
            /* store element in the i-th row */
            i_ptr = vr_ptr[i] + vr_len[i];
            sv_ind[i_ptr] = j;
            sv_val[i_ptr] = val;
            /* increase count of the i-th row */
            vr_len[i]++;
         }
      }
      /* initialize the matrices P and Q (initially P = Q = I) */
      for (k = 1; k <= n; k++)
         pp_row[k] = pp_col[k] = qq_row[k] = qq_col[k] = k;
      /* set sva partitioning pointers */
      luf->sv_beg = sv_beg;
      luf->sv_end = sv_end;
      /* the initial physical order of rows and columns of the matrix V
         is n+1, ..., n+n, 1, ..., n (firstly columns, then rows) */
      luf->sv_head = n+1;
      luf->sv_tail = n;
      for (i = 1; i <= n; i++)
      {  sv_prev[i] = i-1;
         sv_next[i] = i+1;
      }
      sv_prev[1] = n+n;
      sv_next[n] = 0;
      for (j = 1; j <= n; j++)
      {  sv_prev[n+j] = n+j-1;
         sv_next[n+j] = n+j+1;
      }
      sv_prev[n+1] = 0;
      sv_next[n+n] = 1;
      /* clear working arrays */
      for (k = 1; k <= n; k++)
      {  flag[k] = 0;
         work[k] = 0.0;
      }
      /* initialize some statistics */
      luf->nnz_a = nnz;
      luf->nnz_f = 0;
      luf->nnz_v = nnz;
      luf->max_a = big;
      luf->big_v = big;
      luf->rank = -1;
      /* initially the active submatrix is the entire matrix V */
      /* largest of absolute values of elements in each active row is
         unknown yet */
      for (i = 1; i <= n; i++) vr_max[i] = -1.0;
      /* build linked lists of active rows */
      for (len = 0; len <= n; len++) rs_head[len] = 0;
      for (i = 1; i <= n; i++)
      {  len = vr_len[i];
         rs_prev[i] = 0;
         rs_next[i] = rs_head[len];
         if (rs_next[i] != 0) rs_prev[rs_next[i]] = i;
         rs_head[len] = i;
      }
      /* build linked lists of active columns */
      for (len = 0; len <= n; len++) cs_head[len] = 0;
      for (j = 1; j <= n; j++)
      {  len = vc_len[j];
         cs_prev[j] = 0;
         cs_next[j] = cs_head[len];
         if (cs_next[j] != 0) cs_prev[cs_next[j]] = j;
         cs_head[len] = j;
      }
done: /* return to the factorizing routine */
      return ret;
}

int find_pivot(LUF *luf, int *_p, int *_q)
{     int n = luf->n;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      double *vr_max = luf->vr_max;
      int *rs_head = luf->rs_head;
      int *rs_next = luf->rs_next;
      int *cs_head = luf->cs_head;
      int *cs_prev = luf->cs_prev;
      int *cs_next = luf->cs_next;
      double piv_tol = luf->piv_tol;
      int piv_lim = luf->piv_lim;
      int suhl = luf->suhl;
      int p, q, len, i, i_beg, i_end, i_ptr, j, j_beg, j_end, j_ptr,
         ncand, next_j, min_p, min_q, min_len;
      double best, cost, big, temp;
      /* initially no pivot candidates have been found so far */
      p = q = 0, best = DBL_MAX, ncand = 0;
      /* if in the active submatrix there is a column that has the only
         non-zero (column singleton), choose it as pivot */
      j = cs_head[1];
      if (j != 0)
      {  xassert(vc_len[j] == 1);
         p = sv_ind[vc_ptr[j]], q = j;
         goto done;
      }
      /* if in the active submatrix there is a row that has the only
         non-zero (row singleton), choose it as pivot */
      i = rs_head[1];
      if (i != 0)
      {  xassert(vr_len[i] == 1);
         p = i, q = sv_ind[vr_ptr[i]];
         goto done;
      }
      /* there are no singletons in the active submatrix; walk through
         other non-empty rows and columns */
      for (len = 2; len <= n; len++)
      {  /* consider active columns that have len non-zeros */
         for (j = cs_head[len]; j != 0; j = next_j)
         {  /* the j-th column has len non-zeros */
            j_beg = vc_ptr[j];
            j_end = j_beg + vc_len[j] - 1;
            /* save pointer to the next column with the same length */
            next_j = cs_next[j];
            /* find an element in the j-th column, which is placed in a
               row with minimal number of non-zeros and satisfies to the
               stability condition (such element may not exist) */
            min_p = min_q = 0, min_len = INT_MAX;
            for (j_ptr = j_beg; j_ptr <= j_end; j_ptr++)
            {  /* get row index of v[i,j] */
               i = sv_ind[j_ptr];
               i_beg = vr_ptr[i];
               i_end = i_beg + vr_len[i] - 1;
               /* if the i-th row is not shorter than that one, where
                  minimal element is currently placed, skip v[i,j] */
               if (vr_len[i] >= min_len) continue;
               /* determine the largest of absolute values of elements
                  in the i-th row */
               big = vr_max[i];
               if (big < 0.0)
               {  /* the largest value is unknown yet; compute it */
                  for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
                  {  temp = sv_val[i_ptr];
                     if (temp < 0.0) temp = - temp;
                     if (big < temp) big = temp;
                  }
                  vr_max[i] = big;
               }
               /* find v[i,j] in the i-th row */
               for (i_ptr = vr_ptr[i]; sv_ind[i_ptr] != j; i_ptr++);
               xassert(i_ptr <= i_end);
               /* if v[i,j] doesn't satisfy to the stability condition,
                  skip it */
               temp = sv_val[i_ptr];
               if (temp < 0.0) temp = - temp;
               if (temp < piv_tol * big) continue;
               /* v[i,j] is better than the current minimal element */
               min_p = i, min_q = j, min_len = vr_len[i];
               /* if Markowitz cost of the current minimal element is
                  not greater than (len-1)**2, it can be chosen right
                  now; this heuristic reduces the search and works well
                  in many cases */
               if (min_len <= len)
               {  p = min_p, q = min_q;
                  goto done;
               }
            }
            /* the j-th column has been scanned */
            if (min_p != 0)
            {  /* the minimal element is a next pivot candidate */
               ncand++;
               /* compute its Markowitz cost */
               cost = (double)(min_len - 1) * (double)(len - 1);
               /* choose between the minimal element and the current
                  candidate */
               if (cost < best) p = min_p, q = min_q, best = cost;
               /* if piv_lim candidates have been considered, there are
                  doubts that a much better candidate exists; therefore
                  it's time to terminate the search */
               if (ncand == piv_lim) goto done;
            }
            else
            {  /* the j-th column has no elements, which satisfy to the
                  stability condition; Uwe Suhl suggests to exclude such
                  column from the further consideration until it becomes
                  a column singleton; in hard cases this significantly
                  reduces a time needed for pivot searching */
               if (suhl)
               {  /* remove the j-th column from the active set */
                  if (cs_prev[j] == 0)
                     cs_head[len] = cs_next[j];
                  else
                     cs_next[cs_prev[j]] = cs_next[j];
                  if (cs_next[j] == 0)
                     /* nop */;
                  else
                     cs_prev[cs_next[j]] = cs_prev[j];
                  /* the following assignment is used to avoid an error
                     when the routine eliminate (see below) will try to
                     remove the j-th column from the active set */
                  cs_prev[j] = cs_next[j] = j;
               }
            }
         }
         /* consider active rows that have len non-zeros */
         for (i = rs_head[len]; i != 0; i = rs_next[i])
         {  /* the i-th row has len non-zeros */
            i_beg = vr_ptr[i];
            i_end = i_beg + vr_len[i] - 1;
            /* determine the largest of absolute values of elements in
               the i-th row */
            big = vr_max[i];
            if (big < 0.0)
            {  /* the largest value is unknown yet; compute it */
               for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
               {  temp = sv_val[i_ptr];
                  if (temp < 0.0) temp = - temp;
                  if (big < temp) big = temp;
               }
               vr_max[i] = big;
            }
            /* find an element in the i-th row, which is placed in a
               column with minimal number of non-zeros and satisfies to
               the stability condition (such element always exists) */
            min_p = min_q = 0, min_len = INT_MAX;
            for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
            {  /* get column index of v[i,j] */
               j = sv_ind[i_ptr];
               /* if the j-th column is not shorter than that one, where
                  minimal element is currently placed, skip v[i,j] */
               if (vc_len[j] >= min_len) continue;
               /* if v[i,j] doesn't satisfy to the stability condition,
                  skip it */
               temp = sv_val[i_ptr];
               if (temp < 0.0) temp = - temp;
               if (temp < piv_tol * big) continue;
               /* v[i,j] is better than the current minimal element */
               min_p = i, min_q = j, min_len = vc_len[j];
               /* if Markowitz cost of the current minimal element is
                  not greater than (len-1)**2, it can be chosen right
                  now; this heuristic reduces the search and works well
                  in many cases */
               if (min_len <= len)
               {  p = min_p, q = min_q;
                  goto done;
               }
            }
            /* the i-th row has been scanned */
            if (min_p != 0)
            {  /* the minimal element is a next pivot candidate */
               ncand++;
               /* compute its Markowitz cost */
               cost = (double)(len - 1) * (double)(min_len - 1);
               /* choose between the minimal element and the current
                  candidate */
               if (cost < best) p = min_p, q = min_q, best = cost;
               /* if piv_lim candidates have been considered, there are
                  doubts that a much better candidate exists; therefore
                  it's time to terminate the search */
               if (ncand == piv_lim) goto done;
            }
            else
            {  /* this can't be because this can never be */
               xassert(min_p != min_p);
            }
         }
      }
done: /* bring the pivot to the factorizing routine */
      *_p = p, *_q = q;
      return (p == 0);
}

int eliminate(LUF *luf, int p, int q)
{     int n = luf->n;
      int *fc_ptr = luf->fc_ptr;
      int *fc_len = luf->fc_len;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vr_cap = luf->vr_cap;
      double *vr_piv = luf->vr_piv;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *vc_cap = luf->vc_cap;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int *sv_prev = luf->sv_prev;
      int *sv_next = luf->sv_next;
      double *vr_max = luf->vr_max;
      int *rs_head = luf->rs_head;
      int *rs_prev = luf->rs_prev;
      int *rs_next = luf->rs_next;
      int *cs_head = luf->cs_head;
      int *cs_prev = luf->cs_prev;
      int *cs_next = luf->cs_next;
      int *flag = luf->flag;
      double *work = luf->work;
      double eps_tol = luf->eps_tol;
      /* at this stage the row-wise representation of the matrix F is
         not used, so fr_len can be used as a working array */
      int *ndx = luf->fr_len;
      int ret = 0;
      int len, fill, i, i_beg, i_end, i_ptr, j, j_beg, j_end, j_ptr, k,
         p_beg, p_end, p_ptr, q_beg, q_end, q_ptr;
      double fip, val, vpq, temp;
      xassert(1 <= p && p <= n);
      xassert(1 <= q && q <= n);
      /* remove the p-th (pivot) row from the active set; this row will
         never return there */
      if (rs_prev[p] == 0)
         rs_head[vr_len[p]] = rs_next[p];
      else
         rs_next[rs_prev[p]] = rs_next[p];
      if (rs_next[p] == 0)
         ;
      else
         rs_prev[rs_next[p]] = rs_prev[p];
      /* remove the q-th (pivot) column from the active set; this column
         will never return there */
      if (cs_prev[q] == 0)
         cs_head[vc_len[q]] = cs_next[q];
      else
         cs_next[cs_prev[q]] = cs_next[q];
      if (cs_next[q] == 0)
         ;
      else
         cs_prev[cs_next[q]] = cs_prev[q];
      /* find the pivot v[p,q] = u[k,k] in the p-th row */
      p_beg = vr_ptr[p];
      p_end = p_beg + vr_len[p] - 1;
      for (p_ptr = p_beg; sv_ind[p_ptr] != q; p_ptr++) /* nop */;
      xassert(p_ptr <= p_end);
      /* store value of the pivot */
      vpq = (vr_piv[p] = sv_val[p_ptr]);
      /* remove the pivot from the p-th row */
      sv_ind[p_ptr] = sv_ind[p_end];
      sv_val[p_ptr] = sv_val[p_end];
      vr_len[p]--;
      p_end--;
      /* find the pivot v[p,q] = u[k,k] in the q-th column */
      q_beg = vc_ptr[q];
      q_end = q_beg + vc_len[q] - 1;
      for (q_ptr = q_beg; sv_ind[q_ptr] != p; q_ptr++) /* nop */;
      xassert(q_ptr <= q_end);
      /* remove the pivot from the q-th column */
      sv_ind[q_ptr] = sv_ind[q_end];
      vc_len[q]--;
      q_end--;
      /* walk through the p-th (pivot) row, which doesn't contain the
         pivot v[p,q] already, and do the following... */
      for (p_ptr = p_beg; p_ptr <= p_end; p_ptr++)
      {  /* get column index of v[p,j] */
         j = sv_ind[p_ptr];
         /* store v[p,j] to the working array */
         flag[j] = 1;
         work[j] = sv_val[p_ptr];
         /* remove the j-th column from the active set; this column will
            return there later with new length */
         if (cs_prev[j] == 0)
            cs_head[vc_len[j]] = cs_next[j];
         else
            cs_next[cs_prev[j]] = cs_next[j];
         if (cs_next[j] == 0)
            ;
         else
            cs_prev[cs_next[j]] = cs_prev[j];
         /* find v[p,j] in the j-th column */
         j_beg = vc_ptr[j];
         j_end = j_beg + vc_len[j] - 1;
         for (j_ptr = j_beg; sv_ind[j_ptr] != p; j_ptr++) /* nop */;
         xassert(j_ptr <= j_end);
         /* since v[p,j] leaves the active submatrix, remove it from the
            j-th column; however, v[p,j] is kept in the p-th row */
         sv_ind[j_ptr] = sv_ind[j_end];
         vc_len[j]--;
      }
      /* walk through the q-th (pivot) column, which doesn't contain the
         pivot v[p,q] already, and perform gaussian elimination */
      while (q_beg <= q_end)
      {  /* element v[i,q] should be eliminated */
         /* get row index of v[i,q] */
         i = sv_ind[q_beg];
         /* remove the i-th row from the active set; later this row will
            return there with new length */
         if (rs_prev[i] == 0)
            rs_head[vr_len[i]] = rs_next[i];
         else
            rs_next[rs_prev[i]] = rs_next[i];
         if (rs_next[i] == 0)
            ;
         else
            rs_prev[rs_next[i]] = rs_prev[i];
         /* find v[i,q] in the i-th row */
         i_beg = vr_ptr[i];
         i_end = i_beg + vr_len[i] - 1;
         for (i_ptr = i_beg; sv_ind[i_ptr] != q; i_ptr++) /* nop */;
         xassert(i_ptr <= i_end);
         /* compute gaussian multiplier f[i,p] = v[i,q] / v[p,q] */
         fip = sv_val[i_ptr] / vpq;
         /* since v[i,q] should be eliminated, remove it from the i-th
            row */
         sv_ind[i_ptr] = sv_ind[i_end];
         sv_val[i_ptr] = sv_val[i_end];
         vr_len[i]--;
         i_end--;
         /* and from the q-th column */
         sv_ind[q_beg] = sv_ind[q_end];
         vc_len[q]--;
         q_end--;
         /* perform gaussian transformation:
            (i-th row) := (i-th row) - f[i,p] * (p-th row)
            note that now the p-th row, which is in the working array,
            doesn't contain the pivot v[p,q], and the i-th row doesn't
            contain the eliminated element v[i,q] */
         /* walk through the i-th row and transform existing non-zero
            elements */
         fill = vr_len[p];
         for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
         {  /* get column index of v[i,j] */
            j = sv_ind[i_ptr];
            /* v[i,j] := v[i,j] - f[i,p] * v[p,j] */
            if (flag[j])
            {  /* v[p,j] != 0 */
               temp = (sv_val[i_ptr] -= fip * work[j]);
               if (temp < 0.0) temp = - temp;
               flag[j] = 0;
               fill--; /* since both v[i,j] and v[p,j] exist */
               if (temp == 0.0 || temp < eps_tol)
               {  /* new v[i,j] is closer to zero; replace it by exact
                     zero, i.e. remove it from the active submatrix */
                  /* remove v[i,j] from the i-th row */
                  sv_ind[i_ptr] = sv_ind[i_end];
                  sv_val[i_ptr] = sv_val[i_end];
                  vr_len[i]--;
                  i_ptr--;
                  i_end--;
                  /* find v[i,j] in the j-th column */
                  j_beg = vc_ptr[j];
                  j_end = j_beg + vc_len[j] - 1;
                  for (j_ptr = j_beg; sv_ind[j_ptr] != i; j_ptr++);
                  xassert(j_ptr <= j_end);
                  /* remove v[i,j] from the j-th column */
                  sv_ind[j_ptr] = sv_ind[j_end];
                  vc_len[j]--;
               }
               else
               {  /* v_big := max(v_big, |v[i,j]|) */
                  if (luf->big_v < temp) luf->big_v = temp;
               }
            }
         }
         /* now flag is the pattern of the set v[p,*] \ v[i,*], and fill
            is number of non-zeros in this set; therefore up to fill new
            non-zeros may appear in the i-th row */
         if (vr_len[i] + fill > vr_cap[i])
         {  /* enlarge the i-th row */
            if (luf_enlarge_row(luf, i, vr_len[i] + fill))
            {  /* overflow of the sparse vector area */
               ret = 1;
               goto done;
            }
            /* defragmentation may change row and column pointers of the
               matrix V */
            p_beg = vr_ptr[p];
            p_end = p_beg + vr_len[p] - 1;
            q_beg = vc_ptr[q];
            q_end = q_beg + vc_len[q] - 1;
         }
         /* walk through the p-th (pivot) row and create new elements
            of the i-th row that appear due to fill-in; column indices
            of these new elements are accumulated in the array ndx */
         len = 0;
         for (p_ptr = p_beg; p_ptr <= p_end; p_ptr++)
         {  /* get column index of v[p,j], which may cause fill-in */
            j = sv_ind[p_ptr];
            if (flag[j])
            {  /* compute new non-zero v[i,j] = 0 - f[i,p] * v[p,j] */
               temp = (val = - fip * work[j]);
               if (temp < 0.0) temp = - temp;
               if (temp == 0.0 || temp < eps_tol)
                  /* if v[i,j] is closer to zero; just ignore it */;
               else
               {  /* add v[i,j] to the i-th row */
                  i_ptr = vr_ptr[i] + vr_len[i];
                  sv_ind[i_ptr] = j;
                  sv_val[i_ptr] = val;
                  vr_len[i]++;
                  /* remember column index of v[i,j] */
                  ndx[++len] = j;
                  /* big_v := max(big_v, |v[i,j]|) */
                  if (luf->big_v < temp) luf->big_v = temp;
               }
            }
            else
            {  /* there is no fill-in, because v[i,j] already exists in
                  the i-th row; restore the flag of the element v[p,j],
                  which was reset before */
               flag[j] = 1;
            }
         }
         /* add new non-zeros v[i,j] to the corresponding columns */
         for (k = 1; k <= len; k++)
         {  /* get column index of new non-zero v[i,j] */
            j = ndx[k];
            /* one free location is needed in the j-th column */
            if (vc_len[j] + 1 > vc_cap[j])
            {  /* enlarge the j-th column */
               if (luf_enlarge_col(luf, j, vc_len[j] + 10))
               {  /* overflow of the sparse vector area */
                  ret = 1;
                  goto done;
               }
               /* defragmentation may change row and column pointers of
                  the matrix V */
               p_beg = vr_ptr[p];
               p_end = p_beg + vr_len[p] - 1;
               q_beg = vc_ptr[q];
               q_end = q_beg + vc_len[q] - 1;
            }
            /* add new non-zero v[i,j] to the j-th column */
            j_ptr = vc_ptr[j] + vc_len[j];
            sv_ind[j_ptr] = i;
            vc_len[j]++;
         }
         /* now the i-th row has been completely transformed, therefore
            it can return to the active set with new length */
         rs_prev[i] = 0;
         rs_next[i] = rs_head[vr_len[i]];
         if (rs_next[i] != 0) rs_prev[rs_next[i]] = i;
         rs_head[vr_len[i]] = i;
         /* the largest of absolute values of elements in the i-th row
            is currently unknown */
         vr_max[i] = -1.0;
         /* at least one free location is needed to store the gaussian
            multiplier */
         if (luf->sv_end - luf->sv_beg < 1)
         {  /* there are no free locations at all; defragment SVA */
            luf_defrag_sva(luf);
            if (luf->sv_end - luf->sv_beg < 1)
            {  /* overflow of the sparse vector area */
               ret = 1;
               goto done;
            }
            /* defragmentation may change row and column pointers of the
               matrix V */
            p_beg = vr_ptr[p];
            p_end = p_beg + vr_len[p] - 1;
            q_beg = vc_ptr[q];
            q_end = q_beg + vc_len[q] - 1;
         }
         /* add the element f[i,p], which is the gaussian multiplier,
            to the matrix F */
         luf->sv_end--;
         sv_ind[luf->sv_end] = i;
         sv_val[luf->sv_end] = fip;
         fc_len[p]++;
         /* end of elimination loop */
      }
      /* at this point the q-th (pivot) column should be empty */
      xassert(vc_len[q] == 0);
      /* reset capacity of the q-th column */
      vc_cap[q] = 0;
      /* remove node of the q-th column from the addressing list */
      k = n + q;
      if (sv_prev[k] == 0)
         luf->sv_head = sv_next[k];
      else
         sv_next[sv_prev[k]] = sv_next[k];
      if (sv_next[k] == 0)
         luf->sv_tail = sv_prev[k];
      else
         sv_prev[sv_next[k]] = sv_prev[k];
      /* the p-th column of the matrix F has been completely built; set
         its pointer */
      fc_ptr[p] = luf->sv_end;
      /* walk through the p-th (pivot) row and do the following... */
      for (p_ptr = p_beg; p_ptr <= p_end; p_ptr++)
      {  /* get column index of v[p,j] */
         j = sv_ind[p_ptr];
         /* erase v[p,j] from the working array */
         flag[j] = 0;
         work[j] = 0.0;
         /* the j-th column has been completely transformed, therefore
            it can return to the active set with new length; however
            the special case c_prev[j] = c_next[j] = j means that the
            routine find_pivot excluded the j-th column from the active
            set due to Uwe Suhl's rule, and therefore in this case the
            column can return to the active set only if it is a column
            singleton */
         if (!(vc_len[j] != 1 && cs_prev[j] == j && cs_next[j] == j))
         {  cs_prev[j] = 0;
            cs_next[j] = cs_head[vc_len[j]];
            if (cs_next[j] != 0) cs_prev[cs_next[j]] = j;
            cs_head[vc_len[j]] = j;
         }
      }
done: /* return to the factorizing routine */
      return ret;
}

int build_v_cols(LUF *luf)
{     int n = luf->n;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *vc_cap = luf->vc_cap;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int *sv_prev = luf->sv_prev;
      int *sv_next = luf->sv_next;
      int ret = 0;
      int i, i_beg, i_end, i_ptr, j, j_ptr, k, nnz;
      /* it is assumed that on entry all columns of the matrix V are
         empty, i.e. vc_len[j] = vc_cap[j] = 0 for all j = 1, ..., n,
         and have been removed from the addressing list */
      /* count non-zeros in columns of the matrix V; count total number
         of non-zeros in this matrix */
      nnz = 0;
      for (i = 1; i <= n; i++)
      {  /* walk through elements of the i-th row and count non-zeros
            in the corresponding columns */
         i_beg = vr_ptr[i];
         i_end = i_beg + vr_len[i] - 1;
         for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
            vc_cap[sv_ind[i_ptr]]++;
         /* count total number of non-zeros */
         nnz += vr_len[i];
      }
      /* store total number of non-zeros */
      luf->nnz_v = nnz;
      /* check for free locations */
      if (luf->sv_end - luf->sv_beg < nnz)
      {  /* overflow of the sparse vector area */
         ret = 1;
         goto done;
      }
      /* allocate columns of the matrix V */
      for (j = 1; j <= n; j++)
      {  /* set pointer to the j-th column */
         vc_ptr[j] = luf->sv_beg;
         /* reserve locations for the j-th column */
         luf->sv_beg += vc_cap[j];
      }
      /* build the matrix V in column-wise format using this matrix in
         row-wise format */
      for (i = 1; i <= n; i++)
      {  /* walk through elements of the i-th row */
         i_beg = vr_ptr[i];
         i_end = i_beg + vr_len[i] - 1;
         for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
         {  /* get column index */
            j = sv_ind[i_ptr];
            /* store element in the j-th column */
            j_ptr = vc_ptr[j] + vc_len[j];
            sv_ind[j_ptr] = i;
            sv_val[j_ptr] = sv_val[i_ptr];
            /* increase length of the j-th column */
            vc_len[j]++;
         }
      }
      /* now columns are placed in the sparse vector area behind rows
         in the order n+1, n+2, ..., n+n; so insert column nodes in the
         addressing list using this order */
      for (k = n+1; k <= n+n; k++)
      {  sv_prev[k] = k-1;
         sv_next[k] = k+1;
      }
      sv_prev[n+1] = luf->sv_tail;
      sv_next[luf->sv_tail] = n+1;
      sv_next[n+n] = 0;
      luf->sv_tail = n+n;
done: /* return to the factorizing routine */
      return ret;
}

int build_f_rows(LUF *luf)
{     int n = luf->n;
      int *fr_ptr = luf->fr_ptr;
      int *fr_len = luf->fr_len;
      int *fc_ptr = luf->fc_ptr;
      int *fc_len = luf->fc_len;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int ret = 0;
      int i, j, j_beg, j_end, j_ptr, ptr, nnz;
      /* clear rows of the matrix F */
      for (i = 1; i <= n; i++) fr_len[i] = 0;
      /* count non-zeros in rows of the matrix F; count total number of
         non-zeros in this matrix */
      nnz = 0;
      for (j = 1; j <= n; j++)
      {  /* walk through elements of the j-th column and count non-zeros
            in the corresponding rows */
         j_beg = fc_ptr[j];
         j_end = j_beg + fc_len[j] - 1;
         for (j_ptr = j_beg; j_ptr <= j_end; j_ptr++)
            fr_len[sv_ind[j_ptr]]++;
         /* increase total number of non-zeros */
         nnz += fc_len[j];
      }
      /* store total number of non-zeros */
      luf->nnz_f = nnz;
      /* check for free locations */
      if (luf->sv_end - luf->sv_beg < nnz)
      {  /* overflow of the sparse vector area */
         ret = 1;
         goto done;
      }
      /* allocate rows of the matrix F */
      for (i = 1; i <= n; i++)
      {  /* set pointer to the end of the i-th row; later this pointer
            will be set to the beginning of the i-th row */
         fr_ptr[i] = luf->sv_end;
         /* reserve locations for the i-th row */
         luf->sv_end -= fr_len[i];
      }
      /* build the matrix F in row-wise format using this matrix in
         column-wise format */
      for (j = 1; j <= n; j++)
      {  /* walk through elements of the j-th column */
         j_beg = fc_ptr[j];
         j_end = j_beg + fc_len[j] - 1;
         for (j_ptr = j_beg; j_ptr <= j_end; j_ptr++)
         {  /* get row index */
            i = sv_ind[j_ptr];
            /* store element in the i-th row */
            ptr = --fr_ptr[i];
            sv_ind[ptr] = j;
            sv_val[ptr] = sv_val[j_ptr];
         }
      }
done: /* return to the factorizing routine */
      return ret;
}

int luf_factorize(LUF *luf, int n, int (*col)(void *info, int j,
      int ind[], double val[]), void *info)
{     int *pp_row, *pp_col, *qq_row, *qq_col;
      double max_gro = luf->max_gro;
      int i, j, k, p, q, t, ret;
      if (n < 1)
         xerror("luf_factorize: n = %d; invalid parameter", n);
      if (n > N_MAX)
         xerror("luf_factorize: n = %d; matrix too big", n);
      /* invalidate the factorization */
      luf->valid = 0;
      /* reallocate arrays, if necessary */
      reallocate(luf, n);
      pp_row = luf->pp_row;
      pp_col = luf->pp_col;
      qq_row = luf->qq_row;
      qq_col = luf->qq_col;
      /* estimate initial size of the SVA, if not specified */
      if (luf->sv_size == 0 && luf->new_sva == 0)
         luf->new_sva = 5 * (n + 10);
more: /* reallocate the sparse vector area, if required */
      if (luf->new_sva > 0)
      {  if (luf->sv_ind != NULL) xfree(luf->sv_ind);
         if (luf->sv_val != NULL) xfree(luf->sv_val);
         luf->sv_size = luf->new_sva;
         luf->sv_ind = (int*)xcalloc(1+luf->sv_size, sizeof(int));
         luf->sv_val = (double*)xcalloc(1+luf->sv_size, sizeof(double));
         luf->new_sva = 0;
      }
      /* initialize LU-factorization data structures */
      if (initialize(luf, col, info))
      {  /* overflow of the sparse vector area */
         luf->new_sva = luf->sv_size + luf->sv_size;
         xassert(luf->new_sva > luf->sv_size);
         goto more;
      }
      /* main elimination loop */
      for (k = 1; k <= n; k++)
      {  /* choose a pivot element v[p,q] */
         if (find_pivot(luf, &p, &q))
         {  /* no pivot can be chosen, because the active submatrix is
               exactly zero */
            luf->rank = k - 1;
            ret = LUF_ESING;
            goto done;
         }
         /* let v[p,q] correspond to u[i',j']; permute k-th and i'-th
            rows and k-th and j'-th columns of the matrix U = P*V*Q to
            move the element u[i',j'] to the position u[k,k] */
         i = pp_col[p], j = qq_row[q];
         xassert(k <= i && i <= n && k <= j && j <= n);
         /* permute k-th and i-th rows of the matrix U */
         t = pp_row[k];
         pp_row[i] = t, pp_col[t] = i;
         pp_row[k] = p, pp_col[p] = k;
         /* permute k-th and j-th columns of the matrix U */
         t = qq_col[k];
         qq_col[j] = t, qq_row[t] = j;
         qq_col[k] = q, qq_row[q] = k;
         /* eliminate subdiagonal elements of k-th column of the matrix
            U = P*V*Q using the pivot element u[k,k] = v[p,q] */
         if (eliminate(luf, p, q))
         {  /* overflow of the sparse vector area */
            luf->new_sva = luf->sv_size + luf->sv_size;
            xassert(luf->new_sva > luf->sv_size);
            goto more;
         }
         /* check relative growth of elements of the matrix V */
         if (luf->big_v > max_gro * luf->max_a)
         {  /* the growth is too intensive, therefore most probably the
               matrix A is ill-conditioned */
            luf->rank = k - 1;
            ret = LUF_ECOND;
            goto done;
         }
      }
      /* now the matrix U = P*V*Q is upper triangular, the matrix V has
         been built in row-wise format, and the matrix F has been built
         in column-wise format */
      /* defragment the sparse vector area in order to merge all free
         locations in one continuous extent */
      luf_defrag_sva(luf);
      /* build the matrix V in column-wise format */
      if (build_v_cols(luf))
      {  /* overflow of the sparse vector area */
         luf->new_sva = luf->sv_size + luf->sv_size;
         xassert(luf->new_sva > luf->sv_size);
         goto more;
      }
      /* build the matrix F in row-wise format */
      if (build_f_rows(luf))
      {  /* overflow of the sparse vector area */
         luf->new_sva = luf->sv_size + luf->sv_size;
         xassert(luf->new_sva > luf->sv_size);
         goto more;
      }
      /* the LU-factorization has been successfully computed */
      luf->valid = 1;
      luf->rank = n;
      ret = 0;
      /* if there are few free locations in the sparse vector area, try
         increasing its size in the future */
      t = 3 * (n + luf->nnz_v) + 2 * luf->nnz_f;
      if (luf->sv_size < t)
      {  luf->new_sva = luf->sv_size;
         while (luf->new_sva < t)
         {  k = luf->new_sva;
            luf->new_sva = k + k;
            xassert(luf->new_sva > k);
         }
      }
done: /* return to the calling program */
      return ret;
}

void luf_f_solve(LUF *luf, int tr, double x[])
{     int n = luf->n;
      int *fr_ptr = luf->fr_ptr;
      int *fr_len = luf->fr_len;
      int *fc_ptr = luf->fc_ptr;
      int *fc_len = luf->fc_len;
      int *pp_row = luf->pp_row;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      int i, j, k, beg, end, ptr;
      double xk;
      if (!luf->valid)
         xerror("luf_f_solve: LU-factorization is not valid");
      if (!tr)
      {  /* solve the system F*x = b */
         for (j = 1; j <= n; j++)
         {  k = pp_row[j];
            xk = x[k];
            if (xk != 0.0)
            {  beg = fc_ptr[k];
               end = beg + fc_len[k] - 1;
               for (ptr = beg; ptr <= end; ptr++)
                  x[sv_ind[ptr]] -= sv_val[ptr] * xk;
            }
         }
      }
      else
      {  /* solve the system F'*x = b */
         for (i = n; i >= 1; i--)
         {  k = pp_row[i];
            xk = x[k];
            if (xk != 0.0)
            {  beg = fr_ptr[k];
               end = beg + fr_len[k] - 1;
               for (ptr = beg; ptr <= end; ptr++)
                  x[sv_ind[ptr]] -= sv_val[ptr] * xk;
            }
         }
      }
      return;
}

void luf_v_solve(LUF *luf, int tr, double x[])
{     int n = luf->n;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      double *vr_piv = luf->vr_piv;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *pp_row = luf->pp_row;
      int *qq_col = luf->qq_col;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      double *b = luf->work;
      int i, j, k, beg, end, ptr;
      double temp;
      if (!luf->valid)
         xerror("luf_v_solve: LU-factorization is not valid");
      for (k = 1; k <= n; k++) b[k] = x[k], x[k] = 0.0;
      if (!tr)
      {  /* solve the system V*x = b */
         for (k = n; k >= 1; k--)
         {  i = pp_row[k], j = qq_col[k];
            temp = b[i];
            if (temp != 0.0)
            {  x[j] = (temp /= vr_piv[i]);
               beg = vc_ptr[j];
               end = beg + vc_len[j] - 1;
               for (ptr = beg; ptr <= end; ptr++)
                  b[sv_ind[ptr]] -= sv_val[ptr] * temp;
            }
         }
      }
      else
      {  /* solve the system V'*x = b */
         for (k = 1; k <= n; k++)
         {  i = pp_row[k], j = qq_col[k];
            temp = b[j];
            if (temp != 0.0)
            {  x[i] = (temp /= vr_piv[i]);
               beg = vr_ptr[i];
               end = beg + vr_len[i] - 1;
               for (ptr = beg; ptr <= end; ptr++)
                  b[sv_ind[ptr]] -= sv_val[ptr] * temp;
            }
         }
      }
      return;
}

void luf_a_solve(LUF *luf, int tr, double x[])
{     if (!luf->valid)
         xerror("luf_a_solve: LU-factorization is not valid");
      if (!tr)
      {  /* A = F*V, therefore inv(A) = inv(V)*inv(F) */
         luf_f_solve(luf, 0, x);
         luf_v_solve(luf, 0, x);
      }
      else
      {  /* A' = V'*F', therefore inv(A') = inv(F')*inv(V') */
         luf_v_solve(luf, 1, x);
         luf_f_solve(luf, 1, x);
      }
      return;
}

void luf_delete_it(LUF *luf)
{     if (luf->fr_ptr != NULL) xfree(luf->fr_ptr);
      if (luf->fr_len != NULL) xfree(luf->fr_len);
      if (luf->fc_ptr != NULL) xfree(luf->fc_ptr);
      if (luf->fc_len != NULL) xfree(luf->fc_len);
      if (luf->vr_ptr != NULL) xfree(luf->vr_ptr);
      if (luf->vr_len != NULL) xfree(luf->vr_len);
      if (luf->vr_cap != NULL) xfree(luf->vr_cap);
      if (luf->vr_piv != NULL) xfree(luf->vr_piv);
      if (luf->vc_ptr != NULL) xfree(luf->vc_ptr);
      if (luf->vc_len != NULL) xfree(luf->vc_len);
      if (luf->vc_cap != NULL) xfree(luf->vc_cap);
      if (luf->pp_row != NULL) xfree(luf->pp_row);
      if (luf->pp_col != NULL) xfree(luf->pp_col);
      if (luf->qq_row != NULL) xfree(luf->qq_row);
      if (luf->qq_col != NULL) xfree(luf->qq_col);
      if (luf->sv_ind != NULL) xfree(luf->sv_ind);
      if (luf->sv_val != NULL) xfree(luf->sv_val);
      if (luf->sv_prev != NULL) xfree(luf->sv_prev);
      if (luf->sv_next != NULL) xfree(luf->sv_next);
      if (luf->vr_max != NULL) xfree(luf->vr_max);
      if (luf->rs_head != NULL) xfree(luf->rs_head);
      if (luf->rs_prev != NULL) xfree(luf->rs_prev);
      if (luf->rs_next != NULL) xfree(luf->rs_next);
      if (luf->cs_head != NULL) xfree(luf->cs_head);
      if (luf->cs_prev != NULL) xfree(luf->cs_prev);
      if (luf->cs_next != NULL) xfree(luf->cs_next);
      if (luf->flag != NULL) xfree(luf->flag);
      if (luf->work != NULL) xfree(luf->work);
      xfree(luf);
      return;
}

////////// END LUF


////////// START SCF HEADER

struct SCF
{     /* Schur complement factorization */
      int n_max;
      /* maximal order of matrices C, F, U, P; n_max >= 1 */
      int n;
      /* current order of matrices C, F, U, P; n >= 0 */
      double *f; /* double f[1+n_max*n_max]; */
      /* matrix F stored by rows */
      double *u; /* double u[1+n_max*(n_max+1)/2]; */
      /* upper triangle of matrix U stored by rows */
      int *p; /* int p[1+n_max]; */
      /* matrix P; p[i] = j means that P[i,j] = 1 */
      int t_opt;
      /* type of transformation used to restore triangular structure of
         matrix U: */
#define SCF_TBG      1  /* Bartels-Golub elimination */
#define SCF_TGR      2  /* Givens plane rotation */
      int rank;
      /* estimated rank of matrices C and U */
      double *c; /* double c[1+n_max*n_max]; */
      /* matrix C stored in the same format as matrix F and used only
         for debugging; normally this array is not allocated */
      double *w; /* double w[1+n_max]; */
      /* working array */
};

/* return codes: */
#define SCF_ESING    1  /* singular matrix */
#define SCF_ELIMIT   2  /* update limit reached */

#define scf_create_it _glp_scf_create_it
SCF *scf_create_it(int n_max);
/* create Schur complement factorization */

#define scf_update_exp _glp_scf_update_exp
int scf_update_exp(SCF *scf, const double x[], const double y[],
      double z);
/* update factorization on expanding C */

#define scf_solve_it _glp_scf_solve_it
void scf_solve_it(SCF *scf, int tr, double x[]);
/* solve either system C * x = b or C' * x = b */

#define scf_reset_it _glp_scf_reset_it
void scf_reset_it(SCF *scf);
/* reset factorization for empty matrix C */

#define scf_delete_it _glp_scf_delete_it
void scf_delete_it(SCF *scf);
/* delete Schur complement factorization */

////////// END SCF HEADER

////////// START SCF

#define eps 1e-10

SCF *scf_create_it(int n_max)
{     SCF *scf;
      if (!(1 <= n_max && n_max <= 32767))
         xerror("scf_create_it: n_max = %d; invalid parameter", n_max);
      scf = (SCF*)xmalloc(sizeof(SCF));
      scf->n_max = n_max;
      scf->n = 0;
      scf->f = (double*)xcalloc(1 + n_max * n_max, sizeof(double));
      scf->u = (double*)xcalloc(1 + n_max * (n_max + 1) / 2, sizeof(double));
      scf->p = (int*)xcalloc(1 + n_max, sizeof(int));
      scf->t_opt = SCF_TBG;
      scf->rank = 0;
      scf->c = NULL;
      scf->w = (double*)xcalloc(1 + n_max, sizeof(double));
      return scf;
}

int f_loc(SCF *scf, int i, int j)
{     int n_max = scf->n_max;
      xassert(1 <= i && i <= n_max);
      xassert(1 <= j && j <= n_max);
      return (i - 1) * n_max + j;
}

int u_loc(SCF *scf, int i, int j)
{     int n = scf->n_max;
      xassert(1 <= i && i <= n);
      xassert(i <= j && j <= n);
      return (i - 1) * n + j - i * (i - 1) / 2;
}

void bg_transform(SCF *scf, int k, double un[])
{     int n = scf->n;
      double *f = scf->f;
      double *u = scf->u;
      int j, k1, kj, kk, n1, nj;
      double t;
      xassert(1 <= k && k <= n);
      /* main elimination loop */
      for (k = k; k < n; k++)
      {  /* determine location of U[k,k] */
         kk = u_loc(scf, k, k);
         /* determine location of F[k,1] */
         k1 = f_loc(scf, k, 1);
         /* determine location of F[n,1] */
         n1 = f_loc(scf, n, 1);
         /* if |U[k,k]| < |U[n,k]|, interchange k-th and n-th rows to
            provide |U[k,k]| >= |U[n,k]| */
         if (fabs(u[kk]) < fabs(un[k]))
         {  /* interchange k-th and n-th rows of matrix U */
            for (j = k, kj = kk; j <= n; j++, kj++)
               t = u[kj], u[kj] = un[j], un[j] = t;
            /* interchange k-th and n-th rows of matrix F to keep the
               main equality F * C = U * P */
            for (j = 1, kj = k1, nj = n1; j <= n; j++, kj++, nj++)
               t = f[kj], f[kj] = f[nj], f[nj] = t;
         }
         /* now |U[k,k]| >= |U[n,k]| */
         /* if U[k,k] is too small in the magnitude, replace U[k,k] and
            U[n,k] by exact zero */
         if (fabs(u[kk]) < eps) u[kk] = un[k] = 0.0;
         /* if U[n,k] is already zero, elimination is not needed */
         if (un[k] == 0.0) continue;
         /* compute gaussian multiplier t = U[n,k] / U[k,k] */
         t = un[k] / u[kk];
         /* apply gaussian elimination to nullify U[n,k] */
         /* (n-th row of U) := (n-th row of U) - t * (k-th row of U) */
         for (j = k+1, kj = kk+1; j <= n; j++, kj++)
            un[j] -= t * u[kj];
         /* (n-th row of F) := (n-th row of F) - t * (k-th row of F)
            to keep the main equality F * C = U * P */
         for (j = 1, kj = k1, nj = n1; j <= n; j++, kj++, nj++)
            f[nj] -= t * f[kj];
      }
      /* if U[n,n] is too small in the magnitude, replace it by exact
         zero */
      if (fabs(un[n]) < eps) un[n] = 0.0;
      /* store U[n,n] in a proper location */
      u[u_loc(scf, n, n)] = un[n];
      return;
}

void givens(double a, double b, double *c, double *s)
{     double t;
      if (b == 0.0)
         (*c) = 1.0, (*s) = 0.0;
      else if (fabs(a) <= fabs(b))
         t = - a / b, (*s) = 1.0 / sqrt(1.0 + t * t), (*c) = (*s) * t;
      else
         t = - b / a, (*c) = 1.0 / sqrt(1.0 + t * t), (*s) = (*c) * t;
      return;
}

void gr_transform(SCF *scf, int k, double un[])
{     int n = scf->n;
      double *f = scf->f;
      double *u = scf->u;
      int j, k1, kj, kk, n1, nj;
      double c, s;
      xassert(1 <= k && k <= n);
      /* main elimination loop */
      for (k = k; k < n; k++)
      {  /* determine location of U[k,k] */
         kk = u_loc(scf, k, k);
         /* determine location of F[k,1] */
         k1 = f_loc(scf, k, 1);
         /* determine location of F[n,1] */
         n1 = f_loc(scf, n, 1);
         /* if both U[k,k] and U[n,k] are too small in the magnitude,
            replace them by exact zero */
         if (fabs(u[kk]) < eps && fabs(un[k]) < eps)
            u[kk] = un[k] = 0.0;
         /* if U[n,k] is already zero, elimination is not needed */
         if (un[k] == 0.0) continue;
         /* compute the parameters of Givens plane rotation */
         givens(u[kk], un[k], &c, &s);
         /* apply Givens rotation to k-th and n-th rows of matrix U */
         for (j = k, kj = kk; j <= n; j++, kj++)
         {  double ukj = u[kj], unj = un[j];
            u[kj] = c * ukj - s * unj;
            un[j] = s * ukj + c * unj;
         }
         /* apply Givens rotation to k-th and n-th rows of matrix F
            to keep the main equality F * C = U * P */
         for (j = 1, kj = k1, nj = n1; j <= n; j++, kj++, nj++)
         {  double fkj = f[kj], fnj = f[nj];
            f[kj] = c * fkj - s * fnj;
            f[nj] = s * fkj + c * fnj;
         }
      }
      /* if U[n,n] is too small in the magnitude, replace it by exact
         zero */
      if (fabs(un[n]) < eps) un[n] = 0.0;
      /* store U[n,n] in a proper location */
      u[u_loc(scf, n, n)] = un[n];
      return;
}

void transform(SCF *scf, int k, double un[])
{     switch (scf->t_opt)
      {  case SCF_TBG:
            bg_transform(scf, k, un);
            break;
         case SCF_TGR:
            gr_transform(scf, k, un);
            break;
         default:
            xassert(scf != scf);
      }
      return;
}

int estimate_rank(SCF *scf)
{     int n_max = scf->n_max;
      int n = scf->n;
      double *u = scf->u;
      int i, ii, inc, rank = 0;
      for (i = 1, ii = u_loc(scf, i, i), inc = n_max; i <= n;
         i++, ii += inc, inc--)
         if (u[ii] != 0.0) rank++;
      return rank;
}

int scf_update_exp(SCF *scf, const double x[], const double y[],
      double z)
{     int n_max = scf->n_max;
      int n = scf->n;
      double *f = scf->f;
      double *u = scf->u;
      int *p = scf->p;
      double *un = scf->w;
      int i, ij, in, j, k, nj, ret = 0;
      double t;
      /* check if the factorization can be expanded */
      if (n == n_max)
      {  /* there is not enough room */
         ret = SCF_ELIMIT;
         goto done;
      }
      /* increase the order of the factorization */
      scf->n = ++n;
      /* fill new zero column of matrix F */
      for (i = 1, in = f_loc(scf, i, n); i < n; i++, in += n_max)
         f[in] = 0.0;
      /* fill new zero row of matrix F */
      for (j = 1, nj = f_loc(scf, n, j); j < n; j++, nj++)
         f[nj] = 0.0;
      /* fill new unity diagonal element of matrix F */
      f[f_loc(scf, n, n)] = 1.0;
      /* compute new column of matrix U, which is (old F) * x */
      for (i = 1; i < n; i++)
      {  /* u[i,n] := (i-th row of old F) * x */
         t = 0.0;
         for (j = 1, ij = f_loc(scf, i, 1); j < n; j++, ij++)
            t += f[ij] * x[j];
         u[u_loc(scf, i, n)] = t;
      }
      /* compute new (spiked) row of matrix U, which is (old P) * y */
      for (j = 1; j < n; j++) un[j] = y[p[j]];
      /* store new diagonal element of matrix U, which is z */
      un[n] = z;
      /* expand matrix P */
      p[n] = n;
      /* restore upper triangular structure of matrix U */
      for (k = 1; k < n; k++)
         if (un[k] != 0.0) break;
      transform(scf, k, un);
      /* estimate the rank of matrices C and U */
      scf->rank = estimate_rank(scf);
      if (scf->rank != n) ret = SCF_ESING;
done: return ret;
}

void solve(SCF *scf, double x[])
{     int n = scf->n;
      double *f = scf->f;
      double *u = scf->u;
      int *p = scf->p;
      double *y = scf->w;
      int i, j, ij;
      double t;
      /* y := F * b */
      for (i = 1; i <= n; i++)
      {  /* y[i] = (i-th row of F) * b */
         t = 0.0;
         for (j = 1, ij = f_loc(scf, i, 1); j <= n; j++, ij++)
            t += f[ij] * x[j];
         y[i] = t;
      }
      /* y := inv(U) * y */
      for (i = n; i >= 1; i--)
      {  t = y[i];
         for (j = n, ij = u_loc(scf, i, n); j > i; j--, ij--)
            t -= u[ij] * y[j];
         y[i] = t / u[ij];
      }
      /* x := P' * y */
      for (i = 1; i <= n; i++) x[p[i]] = y[i];
      return;
}

void tsolve(SCF *scf, double x[])
{     int n = scf->n;
      double *f = scf->f;
      double *u = scf->u;
      int *p = scf->p;
      double *y = scf->w;
      int i, j, ij;
      double t;
      /* y := P * b */
      for (i = 1; i <= n; i++) y[i] = x[p[i]];
      /* y := inv(U') * y */
      for (i = 1; i <= n; i++)
      {  /* compute y[i] */
         ij = u_loc(scf, i, i);
         t = (y[i] /= u[ij]);
         /* substitute y[i] in other equations */
         for (j = i+1, ij++; j <= n; j++, ij++)
            y[j] -= u[ij] * t;
      }
      /* x := F' * y (computed as linear combination of rows of F) */
      for (j = 1; j <= n; j++) x[j] = 0.0;
      for (i = 1; i <= n; i++)
      {  t = y[i]; /* coefficient of linear combination */
         for (j = 1, ij = f_loc(scf, i, 1); j <= n; j++, ij++)
            x[j] += f[ij] * t;
      }
      return;
}

void scf_solve_it(SCF *scf, int tr, double x[])
{     if (scf->rank < scf->n)
         xerror("scf_solve_it: singular matrix");
      if (!tr)
         solve(scf, x);
      else
         tsolve(scf, x);
      return;
}

void scf_reset_it(SCF *scf)
{     /* reset factorization for empty matrix C */
      scf->n = scf->rank = 0;
      return;
}

void scf_delete_it(SCF *scf)
{     xfree(scf->f);
      xfree(scf->u);
      xfree(scf->p);
      xfree(scf->w);
      xfree(scf);
      return;
}

#undef eps

////////// END SCF


////////// START LPF HEADER

struct LPF
{     /* LP basis factorization */
      int valid;
      /* the factorization is valid only if this flag is set */
      /*--------------------------------------------------------------*/
      /* initial basis matrix B0 */
      int m0_max;
      /* maximal value of m0 (increased automatically, if necessary) */
      int m0;
      /* the order of B0 */
      LUF *luf;
      /* LU-factorization of B0 */
      /*--------------------------------------------------------------*/
      /* current basis matrix B */
      int m;
      /* the order of B */
      double *B; /* double B[1+m*m]; */
      /* B in dense format stored by rows and used only for debugging;
         normally this array is not allocated */
      /*--------------------------------------------------------------*/
      /* augmented matrix (B0 F G H) of the order m0+n */
      int n_max;
      /* maximal number of additional rows and columns */
      int n;
      /* current number of additional rows and columns */
      /*--------------------------------------------------------------*/
      /* m0xn matrix R in column-wise format */
      int *R_ptr; /* int R_ptr[1+n_max]; */
      /* R_ptr[j], 1 <= j <= n, is a pointer to j-th column */
      int *R_len; /* int R_len[1+n_max]; */
      /* R_len[j], 1 <= j <= n, is the length of j-th column */
      /*--------------------------------------------------------------*/
      /* nxm0 matrix S in row-wise format */
      int *S_ptr; /* int S_ptr[1+n_max]; */
      /* S_ptr[i], 1 <= i <= n, is a pointer to i-th row */
      int *S_len; /* int S_len[1+n_max]; */
      /* S_len[i], 1 <= i <= n, is the length of i-th row */
      /*--------------------------------------------------------------*/
      /* Schur complement C of the order n */
      SCF *scf; /* SCF scf[1:n_max]; */
      /* factorization of the Schur complement */
      /*--------------------------------------------------------------*/
      /* matrix P of the order m0+n */
      int *P_row; /* int P_row[1+m0_max+n_max]; */
      /* P_row[i] = j means that P[i,j] = 1 */
      int *P_col; /* int P_col[1+m0_max+n_max]; */
      /* P_col[j] = i means that P[i,j] = 1 */
      /*--------------------------------------------------------------*/
      /* matrix Q of the order m0+n */
      int *Q_row; /* int Q_row[1+m0_max+n_max]; */
      /* Q_row[i] = j means that Q[i,j] = 1 */
      int *Q_col; /* int Q_col[1+m0_max+n_max]; */
      /* Q_col[j] = i means that Q[i,j] = 1 */
      /*--------------------------------------------------------------*/
      /* Sparse Vector Area (SVA) is a set of locations intended to
         store sparse vectors which represent columns of matrix R and
         rows of matrix S; each location is a doublet (ind, val), where
         ind is an index, val is a numerical value of a sparse vector
         element; in the whole each sparse vector is a set of adjacent
         locations defined by a pointer to its first element and its
         length, i.e. the number of its elements */
      int v_size;
      /* the SVA size, in locations; locations are numbered by integers
         1, 2, ..., v_size, and location 0 is not used */
      int v_ptr;
      /* pointer to the first available location */
      int *v_ind; /* int v_ind[1+v_size]; */
      /* v_ind[k], 1 <= k <= v_size, is the index field of location k */
      double *v_val; /* double v_val[1+v_size]; */
      /* v_val[k], 1 <= k <= v_size, is the value field of location k */
      /*--------------------------------------------------------------*/
      double *work1; /* double work1[1+m0+n_max]; */
      /* working array */
      double *work2; /* double work2[1+m0+n_max]; */
      /* working array */
};

/* return codes: */
#define LPF_ESING    1  /* singular matrix */
#define LPF_ECOND    2  /* ill-conditioned matrix */
#define LPF_ELIMIT   3  /* update limit reached */

#define lpf_create_it _glp_lpf_create_it
LPF *lpf_create_it(void);
/* create LP basis factorization */

#define lpf_factorize _glp_lpf_factorize
int lpf_factorize(LPF *lpf, int m, const int bh[], int (*col)
      (void *info, int j, int ind[], double val[]), void *info);
/* compute LP basis factorization */

#define lpf_ftran _glp_lpf_ftran
void lpf_ftran(LPF *lpf, double x[]);
/* perform forward transformation (solve system B*x = b) */

#define lpf_btran _glp_lpf_btran
void lpf_btran(LPF *lpf, double x[]);
/* perform backward transformation (solve system B'*x = b) */

#define lpf_update_it _glp_lpf_update_it
int lpf_update_it(LPF *lpf, int j, int bh, int len, const int ind[],
      const double val[]);
/* update LP basis factorization */

#define lpf_delete_it _glp_lpf_delete_it
void lpf_delete_it(LPF *lpf);
/* delete LP basis factorization */

////////// END LPF HEADER

////////// START LPF

/* CAUTION: DO NOT CHANGE THE LIMIT BELOW */

#define M_MAX 100000000 /* = 100*10^6 */
/* maximal order of the basis matrix */

LPF *lpf_create_it(void)
{     LPF *lpf;
      lpf = (LPF*)xmalloc(sizeof(LPF));
      lpf->valid = 0;
      lpf->m0_max = lpf->m0 = 0;
      lpf->luf = luf_create_it();
      lpf->m = 0;
      lpf->B = NULL;
      lpf->n_max = 50;
      lpf->n = 0;
      lpf->R_ptr = lpf->R_len = NULL;
      lpf->S_ptr = lpf->S_len = NULL;
      lpf->scf = NULL;
      lpf->P_row = lpf->P_col = NULL;
      lpf->Q_row = lpf->Q_col = NULL;
      lpf->v_size = 1000;
      lpf->v_ptr = 0;
      lpf->v_ind = NULL;
      lpf->v_val = NULL;
      lpf->work1 = lpf->work2 = NULL;
      return lpf;
}

int lpf_factorize(LPF *lpf, int m, const int bh[], int (*col)
      (void *info, int j, int ind[], double val[]), void *info)
{     int k, ret;
      xassert(bh == bh);
      if (m < 1)
         xerror("lpf_factorize: m = %d; invalid parameter", m);
      if (m > M_MAX)
         xerror("lpf_factorize: m = %d; matrix too big", m);
      lpf->m0 = lpf->m = m;
      /* invalidate the factorization */
      lpf->valid = 0;
      /* allocate/reallocate arrays, if necessary */
      if (lpf->R_ptr == NULL)
         lpf->R_ptr = (int*)xcalloc(1+lpf->n_max, sizeof(int));
      if (lpf->R_len == NULL)
         lpf->R_len = (int*)xcalloc(1+lpf->n_max, sizeof(int));
      if (lpf->S_ptr == NULL)
         lpf->S_ptr = (int*)xcalloc(1+lpf->n_max, sizeof(int));
      if (lpf->S_len == NULL)
         lpf->S_len = (int*)xcalloc(1+lpf->n_max, sizeof(int));
      if (lpf->scf == NULL)
         lpf->scf = scf_create_it(lpf->n_max);
      if (lpf->v_ind == NULL)
         lpf->v_ind = (int*)xcalloc(1+lpf->v_size, sizeof(int));
      if (lpf->v_val == NULL)
         lpf->v_val = (double*)xcalloc(1+lpf->v_size, sizeof(double));
      if (lpf->m0_max < m)
      {  if (lpf->P_row != NULL) xfree(lpf->P_row);
         if (lpf->P_col != NULL) xfree(lpf->P_col);
         if (lpf->Q_row != NULL) xfree(lpf->Q_row);
         if (lpf->Q_col != NULL) xfree(lpf->Q_col);
         if (lpf->work1 != NULL) xfree(lpf->work1);
         if (lpf->work2 != NULL) xfree(lpf->work2);
         lpf->m0_max = m + 100;
         lpf->P_row = (int*)xcalloc(1+lpf->m0_max+lpf->n_max, sizeof(int));
         lpf->P_col = (int*)xcalloc(1+lpf->m0_max+lpf->n_max, sizeof(int));
         lpf->Q_row = (int*)xcalloc(1+lpf->m0_max+lpf->n_max, sizeof(int));
         lpf->Q_col = (int*)xcalloc(1+lpf->m0_max+lpf->n_max, sizeof(int));
         lpf->work1 = (double*)xcalloc(1+lpf->m0_max+lpf->n_max, sizeof(double));
         lpf->work2 = (double*)xcalloc(1+lpf->m0_max+lpf->n_max, sizeof(double));
      }
      /* try to factorize the basis matrix */
      switch (luf_factorize(lpf->luf, m, col, info))
      {  case 0:
            break;
         case LUF_ESING:
            ret = LPF_ESING;
            goto done;
         case LUF_ECOND:
            ret = LPF_ECOND;
            goto done;
         default:
            xassert(lpf != lpf);
      }
      /* the basis matrix has been successfully factorized */
      lpf->valid = 1;
      /* B = B0, so there are no additional rows/columns */
      lpf->n = 0;
      /* reset the Schur complement factorization */
      scf_reset_it(lpf->scf);
      /* P := Q := I */
      for (k = 1; k <= m; k++)
      {  lpf->P_row[k] = lpf->P_col[k] = k;
         lpf->Q_row[k] = lpf->Q_col[k] = k;
      }
      /* make all SVA locations free */
      lpf->v_ptr = 1;
      ret = 0;
done: /* return to the calling program */
      return ret;
}

void r_prod(LPF *lpf, double y[], double a, const double x[])
{     int n = lpf->n;
      int *R_ptr = lpf->R_ptr;
      int *R_len = lpf->R_len;
      int *v_ind = lpf->v_ind;
      double *v_val = lpf->v_val;
      int j, beg, end, ptr;
      double t;
      for (j = 1; j <= n; j++)
      {  if (x[j] == 0.0) continue;
         /* y := y + alpha * R[j] * x[j] */
         t = a * x[j];
         beg = R_ptr[j];
         end = beg + R_len[j];
         for (ptr = beg; ptr < end; ptr++)
            y[v_ind[ptr]] += t * v_val[ptr];
      }
      return;
}

void rt_prod(LPF *lpf, double y[], double a, const double x[])
{     int n = lpf->n;
      int *R_ptr = lpf->R_ptr;
      int *R_len = lpf->R_len;
      int *v_ind = lpf->v_ind;
      double *v_val = lpf->v_val;
      int j, beg, end, ptr;
      double t;
      for (j = 1; j <= n; j++)
      {  /* t := (j-th column of R) * x */
         t = 0.0;
         beg = R_ptr[j];
         end = beg + R_len[j];
         for (ptr = beg; ptr < end; ptr++)
            t += v_val[ptr] * x[v_ind[ptr]];
         /* y[j] := y[j] + alpha * t */
         y[j] += a * t;
      }
      return;
}

void s_prod(LPF *lpf, double y[], double a, const double x[])
{     int n = lpf->n;
      int *S_ptr = lpf->S_ptr;
      int *S_len = lpf->S_len;
      int *v_ind = lpf->v_ind;
      double *v_val = lpf->v_val;
      int i, beg, end, ptr;
      double t;
      for (i = 1; i <= n; i++)
      {  /* t := (i-th row of S) * x */
         t = 0.0;
         beg = S_ptr[i];
         end = beg + S_len[i];
         for (ptr = beg; ptr < end; ptr++)
            t += v_val[ptr] * x[v_ind[ptr]];
         /* y[i] := y[i] + alpha * t */
         y[i] += a * t;
      }
      return;
}

void st_prod(LPF *lpf, double y[], double a, const double x[])
{     int n = lpf->n;
      int *S_ptr = lpf->S_ptr;
      int *S_len = lpf->S_len;
      int *v_ind = lpf->v_ind;
      double *v_val = lpf->v_val;
      int i, beg, end, ptr;
      double t;
      for (i = 1; i <= n; i++)
      {  if (x[i] == 0.0) continue;
         /* y := y + alpha * S'[i] * x[i] */
         t = a * x[i];
         beg = S_ptr[i];
         end = beg + S_len[i];
         for (ptr = beg; ptr < end; ptr++)
            y[v_ind[ptr]] += t * v_val[ptr];
      }
      return;
}

void lpf_ftran(LPF *lpf, double x[])
{     int m0 = lpf->m0;
      int m = lpf->m;
      int n  = lpf->n;
      int *P_col = lpf->P_col;
      int *Q_col = lpf->Q_col;
      double *fg = lpf->work1;
      double *f = fg;
      double *g = fg + m0;
      int i, ii;
      if (!lpf->valid)
         xerror("lpf_ftran: the factorization is not valid");
      xassert(0 <= m && m <= m0 + n);
      /* (f g) := inv(P) * (b 0) */
      for (i = 1; i <= m0 + n; i++)
         fg[i] = ((ii = P_col[i]) <= m ? x[ii] : 0.0);
      /* f1 := inv(L0) * f */
      luf_f_solve(lpf->luf, 0, f);
      /* g1 := g - S * f1 */
      s_prod(lpf, g, -1.0, f);
      /* g2 := inv(C) * g1 */
      scf_solve_it(lpf->scf, 0, g);
      /* f2 := inv(U0) * (f1 - R * g2) */
      r_prod(lpf, f, -1.0, g);
      luf_v_solve(lpf->luf, 0, f);
      /* (x y) := inv(Q) * (f2 g2) */
      for (i = 1; i <= m; i++)
         x[i] = fg[Q_col[i]];
      return;
}

void lpf_btran(LPF *lpf, double x[])
{     int m0 = lpf->m0;
      int m = lpf->m;
      int n = lpf->n;
      int *P_row = lpf->P_row;
      int *Q_row = lpf->Q_row;
      double *fg = lpf->work1;
      double *f = fg;
      double *g = fg + m0;
      int i, ii;
      if (!lpf->valid)
         xerror("lpf_btran: the factorization is not valid");
      xassert(0 <= m && m <= m0 + n);
      /* (f g) := Q * (b 0) */
      for (i = 1; i <= m0 + n; i++)
         fg[i] = ((ii = Q_row[i]) <= m ? x[ii] : 0.0);
      /* f1 := inv(U'0) * f */
      luf_v_solve(lpf->luf, 1, f);
      /* g1 := inv(C') * (g - R' * f1) */
      rt_prod(lpf, g, -1.0, f);
      scf_solve_it(lpf->scf, 1, g);
      /* g2 := g1 */
      g = g;
      /* f2 := inv(L'0) * (f1 - S' * g2) */
      st_prod(lpf, f, -1.0, g);
      luf_f_solve(lpf->luf, 1, f);
      /* (x y) := P * (f2 g2) */
      for (i = 1; i <= m; i++)
         x[i] = fg[P_row[i]];
      return;
}

void enlarge_sva(LPF *lpf, int new_size)
{     int v_size = lpf->v_size;
      int used = lpf->v_ptr - 1;
      int *v_ind = lpf->v_ind;
      double *v_val = lpf->v_val;
      xassert(v_size < new_size);
      while (v_size < new_size) v_size += v_size;
      lpf->v_size = v_size;
      lpf->v_ind = (int*)xcalloc(1+v_size, sizeof(int));
      lpf->v_val = (double*)xcalloc(1+v_size, sizeof(double));
      xassert(used >= 0);
      memcpy(&lpf->v_ind[1], &v_ind[1], used * sizeof(int));
      memcpy(&lpf->v_val[1], &v_val[1], used * sizeof(double));
      xfree(v_ind);
      xfree(v_val);
      return;
}

int lpf_update_it(LPF *lpf, int j, int bh, int len, const int ind[],
      const double val[])
{     int m0 = lpf->m0;
      int m = lpf->m;
      int n = lpf->n;
      int *R_ptr = lpf->R_ptr;
      int *R_len = lpf->R_len;
      int *S_ptr = lpf->S_ptr;
      int *S_len = lpf->S_len;
      int *P_row = lpf->P_row;
      int *P_col = lpf->P_col;
      int *Q_row = lpf->Q_row;
      int *Q_col = lpf->Q_col;
      int v_ptr = lpf->v_ptr;
      int *v_ind = lpf->v_ind;
      double *v_val = lpf->v_val;
      double *a = lpf->work2; /* new column */
      double *fg = lpf->work1, *f = fg, *g = fg + m0;
      double *vw = lpf->work2, *v = vw, *w = vw + m0;
      double *x = g, *y = w, z;
      int i, ii, k, ret;
      xassert(bh == bh);
      if (!lpf->valid)
         xerror("lpf_update_it: the factorization is not valid");
      if (!(1 <= j && j <= m))
         xerror("lpf_update_it: j = %d; column number out of range",
            j);
      xassert(0 <= m && m <= m0 + n);
      /* check if the basis factorization can be expanded */
      if (n == lpf->n_max)
      {  lpf->valid = 0;
         ret = LPF_ELIMIT;
         goto done;
      }
      /* convert new j-th column of B to dense format */
      for (i = 1; i <= m; i++)
         a[i] = 0.0;
      for (k = 1; k <= len; k++)
      {  i = ind[k];
         if (!(1 <= i && i <= m))
            xerror("lpf_update_it: ind[%d] = %d; row number out of range", k, i);
         if (a[i] != 0.0)
            xerror("lpf_update_it: ind[%d] = %d; duplicate row index not allowed", k, i);
         if (val[k] == 0.0)
            xerror("lpf_update_it: val[%d] = %f; zero element not allowed", k, val[k]);
         a[i] = val[k];
      }
      /* (f g) := inv(P) * (a 0) */
      for (i = 1; i <= m0+n; i++)
         fg[i] = ((ii = P_col[i]) <= m ? a[ii] : 0.0);
      /* (v w) := Q * (ej 0) */
      for (i = 1; i <= m0+n; i++) vw[i] = 0.0;
      vw[Q_col[j]] = 1.0;
      /* f1 := inv(L0) * f (new column of R) */
      luf_f_solve(lpf->luf, 0, f);
      /* v1 := inv(U'0) * v (new row of S) */
      luf_v_solve(lpf->luf, 1, v);
      /* we need at most 2 * m0 available locations in the SVA to store
         new column of matrix R and new row of matrix S */
      if (lpf->v_size < v_ptr + m0 + m0)
      {  enlarge_sva(lpf, v_ptr + m0 + m0);
         v_ind = lpf->v_ind;
         v_val = lpf->v_val;
      }
      /* store new column of R */
      R_ptr[n+1] = v_ptr;
      for (i = 1; i <= m0; i++)
      {  if (f[i] != 0.0)
            v_ind[v_ptr] = i, v_val[v_ptr] = f[i], v_ptr++;
      }
      R_len[n+1] = v_ptr - lpf->v_ptr;
      lpf->v_ptr = v_ptr;
      /* store new row of S */
      S_ptr[n+1] = v_ptr;
      for (i = 1; i <= m0; i++)
      {  if (v[i] != 0.0)
            v_ind[v_ptr] = i, v_val[v_ptr] = v[i], v_ptr++;
      }
      S_len[n+1] = v_ptr - lpf->v_ptr;
      lpf->v_ptr = v_ptr;
      /* x := g - S * f1 (new column of C) */
      s_prod(lpf, x, -1.0, f);
      /* y := w - R' * v1 (new row of C) */
      rt_prod(lpf, y, -1.0, v);
      /* z := - v1 * f1 (new diagonal element of C) */
      z = 0.0;
      for (i = 1; i <= m0; i++) z -= v[i] * f[i];
      /* update factorization of new matrix C */
      switch (scf_update_exp(lpf->scf, x, y, z))
      {  case 0:
            break;
         case SCF_ESING:
            lpf->valid = 0;
            ret = LPF_ESING;
            goto done;
         case SCF_ELIMIT:
            xassert(lpf != lpf);
         default:
            xassert(lpf != lpf);
      }
      /* expand matrix P */
      P_row[m0+n+1] = P_col[m0+n+1] = m0+n+1;
      /* expand matrix Q */
      Q_row[m0+n+1] = Q_col[m0+n+1] = m0+n+1;
      /* permute j-th and last (just added) column of matrix Q */
      i = Q_col[j], ii = Q_col[m0+n+1];
      Q_row[i] = m0+n+1, Q_col[m0+n+1] = i;
      Q_row[ii] = j, Q_col[j] = ii;
      /* increase the number of additional rows and columns */
      lpf->n++;
      xassert(lpf->n <= lpf->n_max);
      /* the factorization has been successfully updated */
      ret = 0;
done: /* return to the calling program */
      return ret;
}

void lpf_delete_it(LPF *lpf)
{     luf_delete_it(lpf->luf);
      xassert(lpf->B == NULL);
      if (lpf->R_ptr != NULL) xfree(lpf->R_ptr);
      if (lpf->R_len != NULL) xfree(lpf->R_len);
      if (lpf->S_ptr != NULL) xfree(lpf->S_ptr);
      if (lpf->S_len != NULL) xfree(lpf->S_len);
      if (lpf->scf != NULL) scf_delete_it(lpf->scf);
      if (lpf->P_row != NULL) xfree(lpf->P_row);
      if (lpf->P_col != NULL) xfree(lpf->P_col);
      if (lpf->Q_row != NULL) xfree(lpf->Q_row);
      if (lpf->Q_col != NULL) xfree(lpf->Q_col);
      if (lpf->v_ind != NULL) xfree(lpf->v_ind);
      if (lpf->v_val != NULL) xfree(lpf->v_val);
      if (lpf->work1 != NULL) xfree(lpf->work1);
      if (lpf->work2 != NULL) xfree(lpf->work2);
      xfree(lpf);
      return;
}

////////// END LPF


////////// START FHV HEADER

struct FHV
{     /* LP basis factorization */
      int m_max;
      /* maximal value of m (increased automatically, if necessary) */
      int m;
      /* the order of matrices B, F, H, V, P0, P, Q */
      int valid;
      /* the factorization is valid only if this flag is set */
      LUF *luf;
      /* LU-factorization (contains the matrices F, V, P, Q) */
      /*--------------------------------------------------------------*/
      /* matrix H in the form of eta file */
      int hh_max;
      /* maximal number of row-like factors (which limits the number of
         updates of the factorization) */
      int hh_nfs;
      /* current number of row-like factors (0 <= hh_nfs <= hh_max) */
      int *hh_ind; /* int hh_ind[1+hh_max]; */
      /* hh_ind[k], k = 1, ..., nfs, is the number of a non-trivial row
         of factor H[k] */
      int *hh_ptr; /* int hh_ptr[1+hh_max]; */
      /* hh_ptr[k], k = 1, ..., nfs, is a pointer to the first element
         of the non-trivial row of factor H[k] in the SVA */
      int *hh_len; /* int hh_len[1+hh_max]; */
      /* hh_len[k], k = 1, ..., nfs, is the number of non-zero elements
         in the non-trivial row of factor H[k] */
      /*--------------------------------------------------------------*/
      /* matrix P0 */
      int *p0_row; /* int p0_row[1+m_max]; */
      /* p0_row[i] = j means that p0[i,j] = 1 */
      int *p0_col; /* int p0_col[1+m_max]; */
      /* p0_col[j] = i means that p0[i,j] = 1 */
      /* if i-th row or column of the matrix F corresponds to i'-th row
         or column of the matrix L = P0*F*inv(P0), then p0_row[i'] = i
         and p0_col[i] = i' */
      /*--------------------------------------------------------------*/
      /* working arrays */
      int *cc_ind; /* int cc_ind[1+m_max]; */
      /* integer working array */
      double *cc_val; /* double cc_val[1+m_max]; */
      /* floating-point working array */
      /*--------------------------------------------------------------*/
      /* control parameters */
      double upd_tol;
      /* update tolerance; if after updating the factorization absolute
         value of some diagonal element u[k,k] of matrix U = P*V*Q is
         less than upd_tol * max(|u[k,*]|, |u[*,k]|), the factorization
         is considered as inaccurate */
      /*--------------------------------------------------------------*/
      /* some statistics */
      int nnz_h;
      /* current number of non-zeros in all factors of matrix H */
};

/* return codes: */
#define FHV_ESING    1  /* singular matrix */
#define FHV_ECOND    2  /* ill-conditioned matrix */
#define FHV_ECHECK   3  /* insufficient accuracy */
#define FHV_ELIMIT   4  /* update limit reached */
#define FHV_EROOM    5  /* SVA overflow */

#define fhv_create_it _glp_fhv_create_it
FHV *fhv_create_it(void);
/* create LP basis factorization */

#define fhv_factorize _glp_fhv_factorize
int fhv_factorize(FHV *fhv, int m, int (*col)(void *info, int j,
      int ind[], double val[]), void *info);
/* compute LP basis factorization */

#define fhv_h_solve _glp_fhv_h_solve
void fhv_h_solve(FHV *fhv, int tr, double x[]);
/* solve system H*x = b or H'*x = b */

#define fhv_ftran _glp_fhv_ftran
void fhv_ftran(FHV *fhv, double x[]);
/* perform forward transformation (solve system B*x = b) */

#define fhv_btran _glp_fhv_btran
void fhv_btran(FHV *fhv, double x[]);
/* perform backward transformation (solve system B'*x = b) */

#define fhv_update_it _glp_fhv_update_it
int fhv_update_it(FHV *fhv, int j, int len, const int ind[],
      const double val[]);
/* update LP basis factorization */

#define fhv_delete_it _glp_fhv_delete_it
void fhv_delete_it(FHV *fhv);
/* delete LP basis factorization */

////////// END FHV HEADER


////////// START FHV

/* CAUTION: DO NOT CHANGE THE LIMIT BELOW */

#define M_MAX 100000000 /* = 100*10^6 */
/* maximal order of the basis matrix */

FHV *fhv_create_it(void)
{     FHV *fhv;
      fhv = (FHV*)xmalloc(sizeof(FHV));
      fhv->m_max = fhv->m = 0;
      fhv->valid = 0;
      fhv->luf = luf_create_it();
      fhv->hh_max = 50;
      fhv->hh_nfs = 0;
      fhv->hh_ind = fhv->hh_ptr = fhv->hh_len = NULL;
      fhv->p0_row = fhv->p0_col = NULL;
      fhv->cc_ind = NULL;
      fhv->cc_val = NULL;
      fhv->upd_tol = 1e-6;
      fhv->nnz_h = 0;
      return fhv;
}

int fhv_factorize(FHV *fhv, int m, int (*col)(void *info, int j,
      int ind[], double val[]), void *info)
{     int ret;
      if (m < 1)
         xerror("fhv_factorize: m = %d; invalid parameter", m);
      if (m > M_MAX)
         xerror("fhv_factorize: m = %d; matrix too big", m);
      fhv->m = m;
      /* invalidate the factorization */
      fhv->valid = 0;
      /* allocate/reallocate arrays, if necessary */
      if (fhv->hh_ind == NULL)
         fhv->hh_ind = (int*)xcalloc(1+fhv->hh_max, sizeof(int));
      if (fhv->hh_ptr == NULL)
         fhv->hh_ptr = (int*)xcalloc(1+fhv->hh_max, sizeof(int));
      if (fhv->hh_len == NULL)
         fhv->hh_len = (int*)xcalloc(1+fhv->hh_max, sizeof(int));
      if (fhv->m_max < m)
      {  if (fhv->p0_row != NULL) xfree(fhv->p0_row);
         if (fhv->p0_col != NULL) xfree(fhv->p0_col);
         if (fhv->cc_ind != NULL) xfree(fhv->cc_ind);
         if (fhv->cc_val != NULL) xfree(fhv->cc_val);
         fhv->m_max = m + 100;
         fhv->p0_row = (int*)xcalloc(1+fhv->m_max, sizeof(int));
         fhv->p0_col = (int*)xcalloc(1+fhv->m_max, sizeof(int));
         fhv->cc_ind = (int*)xcalloc(1+fhv->m_max, sizeof(int));
         fhv->cc_val = (double*)xcalloc(1+fhv->m_max, sizeof(double));
      }
      /* try to factorize the basis matrix */
      switch (luf_factorize(fhv->luf, m, col, info))
      {  case 0:
            break;
         case LUF_ESING:
            ret = FHV_ESING;
            goto done;
         case LUF_ECOND:
            ret = FHV_ECOND;
            goto done;
         default:
            xassert(fhv != fhv);
      }
      /* the basis matrix has been successfully factorized */
      fhv->valid = 1;
      /* H := I */
      fhv->hh_nfs = 0;
      /* P0 := P */
      memcpy(&fhv->p0_row[1], &fhv->luf->pp_row[1], sizeof(int) * m);
      memcpy(&fhv->p0_col[1], &fhv->luf->pp_col[1], sizeof(int) * m);
      /* currently H has no factors */
      fhv->nnz_h = 0;
      ret = 0;
done: /* return to the calling program */
      return ret;
}

void fhv_h_solve(FHV *fhv, int tr, double x[])
{     int nfs = fhv->hh_nfs;
      int *hh_ind = fhv->hh_ind;
      int *hh_ptr = fhv->hh_ptr;
      int *hh_len = fhv->hh_len;
      int *sv_ind = fhv->luf->sv_ind;
      double *sv_val = fhv->luf->sv_val;
      int i, k, beg, end, ptr;
      double temp;
      if (!fhv->valid)
         xerror("fhv_h_solve: the factorization is not valid");
      if (!tr)
      {  /* solve the system H*x = b */
         for (k = 1; k <= nfs; k++)
         {  i = hh_ind[k];
            temp = x[i];
            beg = hh_ptr[k];
            end = beg + hh_len[k] - 1;
            for (ptr = beg; ptr <= end; ptr++)
               temp -= sv_val[ptr] * x[sv_ind[ptr]];
            x[i] = temp;
         }
      }
      else
      {  /* solve the system H'*x = b */
         for (k = nfs; k >= 1; k--)
         {  i = hh_ind[k];
            temp = x[i];
            if (temp == 0.0) continue;
            beg = hh_ptr[k];
            end = beg + hh_len[k] - 1;
            for (ptr = beg; ptr <= end; ptr++)
               x[sv_ind[ptr]] -= sv_val[ptr] * temp;
         }
      }
      return;
}

void fhv_ftran(FHV *fhv, double x[])
{     int *pp_row = fhv->luf->pp_row;
      int *pp_col = fhv->luf->pp_col;
      int *p0_row = fhv->p0_row;
      int *p0_col = fhv->p0_col;
      if (!fhv->valid)
         xerror("fhv_ftran: the factorization is not valid");
      /* B = F*H*V, therefore inv(B) = inv(V)*inv(H)*inv(F) */
      fhv->luf->pp_row = p0_row;
      fhv->luf->pp_col = p0_col;
      luf_f_solve(fhv->luf, 0, x);
      fhv->luf->pp_row = pp_row;
      fhv->luf->pp_col = pp_col;
      fhv_h_solve(fhv, 0, x);
      luf_v_solve(fhv->luf, 0, x);
      return;
}

void fhv_btran(FHV *fhv, double x[])
{     int *pp_row = fhv->luf->pp_row;
      int *pp_col = fhv->luf->pp_col;
      int *p0_row = fhv->p0_row;
      int *p0_col = fhv->p0_col;
      if (!fhv->valid)
         xerror("fhv_btran: the factorization is not valid");
      /* B = F*H*V, therefore inv(B') = inv(F')*inv(H')*inv(V') */
      luf_v_solve(fhv->luf, 1, x);
      fhv_h_solve(fhv, 1, x);
      fhv->luf->pp_row = p0_row;
      fhv->luf->pp_col = p0_col;
      luf_f_solve(fhv->luf, 1, x);
      fhv->luf->pp_row = pp_row;
      fhv->luf->pp_col = pp_col;
      return;
}

int fhv_update_it(FHV *fhv, int j, int len, const int ind[],
      const double val[])
{     int m = fhv->m;
      LUF *luf = fhv->luf;
      int *vr_ptr = luf->vr_ptr;
      int *vr_len = luf->vr_len;
      int *vr_cap = luf->vr_cap;
      double *vr_piv = luf->vr_piv;
      int *vc_ptr = luf->vc_ptr;
      int *vc_len = luf->vc_len;
      int *vc_cap = luf->vc_cap;
      int *pp_row = luf->pp_row;
      int *pp_col = luf->pp_col;
      int *qq_row = luf->qq_row;
      int *qq_col = luf->qq_col;
      int *sv_ind = luf->sv_ind;
      double *sv_val = luf->sv_val;
      double *work = luf->work;
      double eps_tol = luf->eps_tol;
      int *hh_ind = fhv->hh_ind;
      int *hh_ptr = fhv->hh_ptr;
      int *hh_len = fhv->hh_len;
      int *p0_row = fhv->p0_row;
      int *p0_col = fhv->p0_col;
      int *cc_ind = fhv->cc_ind;
      double *cc_val = fhv->cc_val;
      double upd_tol = fhv->upd_tol;
      int i, i_beg, i_end, i_ptr, j_beg, j_end, j_ptr, k, k1, k2, p, q,
         p_beg, p_end, p_ptr, ptr, ret;
      double f, temp;
      if (!fhv->valid)
         xerror("fhv_update_it: the factorization is not valid");
      if (!(1 <= j && j <= m))
         xerror("fhv_update_it: j = %d; column number out of range",
            j);
      /* check if the new factor of matrix H can be created */
      if (fhv->hh_nfs == fhv->hh_max)
      {  /* maximal number of updates has been reached */
         fhv->valid = 0;
         ret = FHV_ELIMIT;
         goto done;
      }
      /* convert new j-th column of B to dense format */
      for (i = 1; i <= m; i++)
         cc_val[i] = 0.0;
      for (k = 1; k <= len; k++)
      {  i = ind[k];
         if (!(1 <= i && i <= m))
            xerror("fhv_update_it: ind[%d] = %d; row number out of range", k, i);
         if (cc_val[i] != 0.0)
            xerror("fhv_update_it: ind[%d] = %d; duplicate row index not allowed", k, i);
         if (val[k] == 0.0)
            xerror("fhv_update_it: val[%d] = %f; zero element not allowed", k, val[k]);
         cc_val[i] = val[k];
      }
      /* new j-th column of V := inv(F * H) * (new B[j]) */
      fhv->luf->pp_row = p0_row;
      fhv->luf->pp_col = p0_col;
      luf_f_solve(fhv->luf, 0, cc_val);
      fhv->luf->pp_row = pp_row;
      fhv->luf->pp_col = pp_col;
      fhv_h_solve(fhv, 0, cc_val);
      /* convert new j-th column of V to sparse format */
      len = 0;
      for (i = 1; i <= m; i++)
      {  temp = cc_val[i];
         if (temp == 0.0 || fabs(temp) < eps_tol) continue;
         len++, cc_ind[len] = i, cc_val[len] = temp;
      }
      /* clear old content of j-th column of matrix V */
      j_beg = vc_ptr[j];
      j_end = j_beg + vc_len[j] - 1;
      for (j_ptr = j_beg; j_ptr <= j_end; j_ptr++)
      {  /* get row index of v[i,j] */
         i = sv_ind[j_ptr];
         /* find v[i,j] in the i-th row */
         i_beg = vr_ptr[i];
         i_end = i_beg + vr_len[i] - 1;
         for (i_ptr = i_beg; sv_ind[i_ptr] != j; i_ptr++) /* nop */;
         xassert(i_ptr <= i_end);
         /* remove v[i,j] from the i-th row */
         sv_ind[i_ptr] = sv_ind[i_end];
         sv_val[i_ptr] = sv_val[i_end];
         vr_len[i]--;
      }
      /* now j-th column of matrix V is empty */
      luf->nnz_v -= vc_len[j];
      vc_len[j] = 0;
      /* add new elements of j-th column of matrix V to corresponding
         row lists; determine indices k1 and k2 */
      k1 = qq_row[j], k2 = 0;
      for (ptr = 1; ptr <= len; ptr++)
      {  /* get row index of v[i,j] */
         i = cc_ind[ptr];
         /* at least one unused location is needed in i-th row */
         if (vr_len[i] + 1 > vr_cap[i])
         {  if (luf_enlarge_row(luf, i, vr_len[i] + 10))
            {  /* overflow of the sparse vector area */
               fhv->valid = 0;
               luf->new_sva = luf->sv_size + luf->sv_size;
               xassert(luf->new_sva > luf->sv_size);
               ret = FHV_EROOM;
               goto done;
            }
         }
         /* add v[i,j] to i-th row */
         i_ptr = vr_ptr[i] + vr_len[i];
         sv_ind[i_ptr] = j;
         sv_val[i_ptr] = cc_val[ptr];
         vr_len[i]++;
         /* adjust index k2 */
         if (k2 < pp_col[i]) k2 = pp_col[i];
      }
      /* capacity of j-th column (which is currently empty) should be
         not less than len locations */
      if (vc_cap[j] < len)
      {  if (luf_enlarge_col(luf, j, len))
         {  /* overflow of the sparse vector area */
            fhv->valid = 0;
            luf->new_sva = luf->sv_size + luf->sv_size;
            xassert(luf->new_sva > luf->sv_size);
            ret = FHV_EROOM;
            goto done;
         }
      }
      /* add new elements of matrix V to j-th column list */
      j_ptr = vc_ptr[j];
      memmove(&sv_ind[j_ptr], &cc_ind[1], len * sizeof(int));
      memmove(&sv_val[j_ptr], &cc_val[1], len * sizeof(double));
      vc_len[j] = len;
      luf->nnz_v += len;
      /* if k1 > k2, diagonal element u[k2,k2] of matrix U is zero and
         therefore the adjacent basis matrix is structurally singular */
      if (k1 > k2)
      {  fhv->valid = 0;
         ret = FHV_ESING;
         goto done;
      }
      /* perform implicit symmetric permutations of rows and columns of
         matrix U */
      i = pp_row[k1], j = qq_col[k1];
      for (k = k1; k < k2; k++)
      {  pp_row[k] = pp_row[k+1], pp_col[pp_row[k]] = k;
         qq_col[k] = qq_col[k+1], qq_row[qq_col[k]] = k;
      }
      pp_row[k2] = i, pp_col[i] = k2;
      qq_col[k2] = j, qq_row[j] = k2;
      /* now i-th row of the matrix V is k2-th row of matrix U; since
         no pivoting is used, only this row will be transformed */
      /* copy elements of i-th row of matrix V to the working array and
         remove these elements from matrix V */
      for (j = 1; j <= m; j++) work[j] = 0.0;
      i_beg = vr_ptr[i];
      i_end = i_beg + vr_len[i] - 1;
      for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
      {  /* get column index of v[i,j] */
         j = sv_ind[i_ptr];
         /* store v[i,j] to the working array */
         work[j] = sv_val[i_ptr];
         /* find v[i,j] in the j-th column */
         j_beg = vc_ptr[j];
         j_end = j_beg + vc_len[j] - 1;
         for (j_ptr = j_beg; sv_ind[j_ptr] != i; j_ptr++) /* nop */;
         xassert(j_ptr <= j_end);
         /* remove v[i,j] from the j-th column */
         sv_ind[j_ptr] = sv_ind[j_end];
         sv_val[j_ptr] = sv_val[j_end];
         vc_len[j]--;
      }
      /* now i-th row of matrix V is empty */
      luf->nnz_v -= vr_len[i];
      vr_len[i] = 0;
      /* create the next row-like factor of the matrix H; this factor
         corresponds to i-th (transformed) row */
      fhv->hh_nfs++;
      hh_ind[fhv->hh_nfs] = i;
      /* hh_ptr[] will be set later */
      hh_len[fhv->hh_nfs] = 0;
      /* up to (k2 - k1) free locations are needed to add new elements
         to the non-trivial row of the row-like factor */
      if (luf->sv_end - luf->sv_beg < k2 - k1)
      {  luf_defrag_sva(luf);
         if (luf->sv_end - luf->sv_beg < k2 - k1)
         {  /* overflow of the sparse vector area */
            fhv->valid = luf->valid = 0;
            luf->new_sva = luf->sv_size + luf->sv_size;
            xassert(luf->new_sva > luf->sv_size);
            ret = FHV_EROOM;
            goto done;
         }
      }
      /* eliminate subdiagonal elements of matrix U */
      for (k = k1; k < k2; k++)
      {  /* v[p,q] = u[k,k] */
         p = pp_row[k], q = qq_col[k];
         /* this is the crucial point, where even tiny non-zeros should
            not be dropped */
         if (work[q] == 0.0) continue;
         /* compute gaussian multiplier f = v[i,q] / v[p,q] */
         f = work[q] / vr_piv[p];
         /* perform gaussian transformation:
            (i-th row) := (i-th row) - f * (p-th row)
            in order to eliminate v[i,q] = u[k2,k] */
         p_beg = vr_ptr[p];
         p_end = p_beg + vr_len[p] - 1;
         for (p_ptr = p_beg; p_ptr <= p_end; p_ptr++)
            work[sv_ind[p_ptr]] -= f * sv_val[p_ptr];
         /* store new element (gaussian multiplier that corresponds to
            p-th row) in the current row-like factor */
         luf->sv_end--;
         sv_ind[luf->sv_end] = p;
         sv_val[luf->sv_end] = f;
         hh_len[fhv->hh_nfs]++;
      }
      /* set pointer to the current row-like factor of the matrix H
         (if no elements were added to this factor, it is unity matrix
         and therefore can be discarded) */
      if (hh_len[fhv->hh_nfs] == 0)
         fhv->hh_nfs--;
      else
      {  hh_ptr[fhv->hh_nfs] = luf->sv_end;
         fhv->nnz_h += hh_len[fhv->hh_nfs];
      }
      /* store new pivot which corresponds to u[k2,k2] */
      vr_piv[i] = work[qq_col[k2]];
      /* new elements of i-th row of matrix V (which are non-diagonal
         elements u[k2,k2+1], ..., u[k2,m] of matrix U = P*V*Q) now are
         contained in the working array; add them to matrix V */
      len = 0;
      for (k = k2+1; k <= m; k++)
      {  /* get column index and value of v[i,j] = u[k2,k] */
         j = qq_col[k];
         temp = work[j];
         /* if v[i,j] is close to zero, skip it */
         if (fabs(temp) < eps_tol) continue;
         /* at least one unused location is needed in j-th column */
         if (vc_len[j] + 1 > vc_cap[j])
         {  if (luf_enlarge_col(luf, j, vc_len[j] + 10))
            {  /* overflow of the sparse vector area */
               fhv->valid = 0;
               luf->new_sva = luf->sv_size + luf->sv_size;
               xassert(luf->new_sva > luf->sv_size);
               ret = FHV_EROOM;
               goto done;
            }
         }
         /* add v[i,j] to j-th column */
         j_ptr = vc_ptr[j] + vc_len[j];
         sv_ind[j_ptr] = i;
         sv_val[j_ptr] = temp;
         vc_len[j]++;
         /* also store v[i,j] to the auxiliary array */
         len++, cc_ind[len] = j, cc_val[len] = temp;
      }
      /* capacity of i-th row (which is currently empty) should be not
         less than len locations */
      if (vr_cap[i] < len)
      {  if (luf_enlarge_row(luf, i, len))
         {  /* overflow of the sparse vector area */
            fhv->valid = 0;
            luf->new_sva = luf->sv_size + luf->sv_size;
            xassert(luf->new_sva > luf->sv_size);
            ret = FHV_EROOM;
            goto done;
         }
      }
      /* add new elements to i-th row list */
      i_ptr = vr_ptr[i];
      memmove(&sv_ind[i_ptr], &cc_ind[1], len * sizeof(int));
      memmove(&sv_val[i_ptr], &cc_val[1], len * sizeof(double));
      vr_len[i] = len;
      luf->nnz_v += len;
      /* updating is finished; check that diagonal element u[k2,k2] is
         not very small in absolute value among other elements in k2-th
         row and k2-th column of matrix U = P*V*Q */
      /* temp = max(|u[k2,*]|, |u[*,k2]|) */
      temp = 0.0;
      /* walk through k2-th row of U which is i-th row of V */
      i = pp_row[k2];
      i_beg = vr_ptr[i];
      i_end = i_beg + vr_len[i] - 1;
      for (i_ptr = i_beg; i_ptr <= i_end; i_ptr++)
         if (temp < fabs(sv_val[i_ptr])) temp = fabs(sv_val[i_ptr]);
      /* walk through k2-th column of U which is j-th column of V */
      j = qq_col[k2];
      j_beg = vc_ptr[j];
      j_end = j_beg + vc_len[j] - 1;
      for (j_ptr = j_beg; j_ptr <= j_end; j_ptr++)
         if (temp < fabs(sv_val[j_ptr])) temp = fabs(sv_val[j_ptr]);
      /* check that u[k2,k2] is not very small */
      if (fabs(vr_piv[i]) < upd_tol * temp)
      {  /* the factorization seems to be inaccurate and therefore must
            be recomputed */
         fhv->valid = 0;
         ret = FHV_ECHECK;
         goto done;
      }
      /* the factorization has been successfully updated */
      ret = 0;
done: /* return to the calling program */
      return ret;
}

void fhv_delete_it(FHV *fhv)
{     luf_delete_it(fhv->luf);
      if (fhv->hh_ind != NULL) xfree(fhv->hh_ind);
      if (fhv->hh_ptr != NULL) xfree(fhv->hh_ptr);
      if (fhv->hh_len != NULL) xfree(fhv->hh_len);
      if (fhv->p0_row != NULL) xfree(fhv->p0_row);
      if (fhv->p0_col != NULL) xfree(fhv->p0_col);
      if (fhv->cc_ind != NULL) xfree(fhv->cc_ind);
      if (fhv->cc_val != NULL) xfree(fhv->cc_val);
      xfree(fhv);
      return;
}

////////// END FHV


////////// START BFD HEADER

/* return codes: */
#define BFD_ESING    1  /* singular matrix */
#define BFD_ECOND    2  /* ill-conditioned matrix */
#define BFD_ECHECK   3  /* insufficient accuracy */
#define BFD_ELIMIT   4  /* update limit reached */
#define BFD_EROOM    5  /* SVA overflow */

struct BFD
{     /* LP basis factorization */
      int valid;
      /* factorization is valid only if this flag is set */
      int type;
      /* factorization type:
         GLP_BF_FT - LUF + Forrest-Tomlin
         GLP_BF_BG - LUF + Schur compl. + Bartels-Golub
         GLP_BF_GR - LUF + Schur compl. + Givens rotation */
      FHV *fhv;
      /* LP basis factorization (GLP_BF_FT) */
      LPF *lpf;
      /* LP basis factorization (GLP_BF_BG, GLP_BF_GR) */
      int lu_size;      /* luf.sv_size */
      double piv_tol;   /* luf.piv_tol */
      int piv_lim;      /* luf.piv_lim */
      int suhl;         /* luf.suhl */
      double eps_tol;   /* luf.eps_tol */
      double max_gro;   /* luf.max_gro */
      int nfs_max;      /* fhv.hh_max */
      double upd_tol;   /* fhv.upd_tol */
      int nrs_max;      /* lpf.n_max */
      int rs_size;      /* lpf.v_size */
      /* internal control parameters */
      int upd_lim;
      /* the factorization update limit */
      int upd_cnt;
      /* the factorization update count */
};

#define bfd_create_it _glp_bfd_create_it
BFD *bfd_create_it(void);
/* create LP basis factorization */

#define bfd_set_parm _glp_bfd_set_parm
void bfd_set_parm(BFD *bfd, const void *parm);
/* change LP basis factorization control parameters */

#define bfd_factorize _glp_bfd_factorize
int bfd_factorize(BFD *bfd, int m, const int bh[], int (*col)
      (void *info, int j, int ind[], double val[]), void *info);
/* compute LP basis factorization */

#define bfd_ftran _glp_bfd_ftran
void bfd_ftran(BFD *bfd, double x[]);
/* perform forward transformation (solve system B*x = b) */

#define bfd_btran _glp_bfd_btran
void bfd_btran(BFD *bfd, double x[]);
/* perform backward transformation (solve system B'*x = b) */

#define bfd_update_it _glp_bfd_update_it
int bfd_update_it(BFD *bfd, int j, int bh, int len, const int ind[],
      const double val[]);
/* update LP basis factorization */

#define bfd_get_count _glp_bfd_get_count
int bfd_get_count(BFD *bfd);
/* determine factorization update count */

#define bfd_delete_it _glp_bfd_delete_it
void bfd_delete_it(BFD *bfd);
/* delete LP basis factorization */


////////// END BFD HEADER


////////// START API HEADER

struct GLPROW;
struct GLPCOL;
struct GLPAIJ;

#define GLP_PROB_MAGIC 0xD7D9D6C2

struct GLPROW
{     /* LP/MIP row (auxiliary variable) */
      int i;
      /* ordinal number (1 to m) assigned to this row */
      char *name;
      /* row name (1 to 255 chars); NULL means no name is assigned to
         this row */
//       AVLNODE *node;
      /* pointer to corresponding node in the row index; NULL means
         that either the row index does not exist or this row has no
         name assigned */
      int level;
      unsigned char origin;
      unsigned char klass;
      int type;
      /* type of the auxiliary variable:
         GLP_FR - free variable
         GLP_LO - variable with lower bound
         GLP_UP - variable with upper bound
         GLP_DB - double-bounded variable
         GLP_FX - fixed variable */
      double lb; /* non-scaled */
      /* lower bound; if the row has no lower bound, lb is zero */
      double ub; /* non-scaled */
      /* upper bound; if the row has no upper bound, ub is zero */
      /* if the row type is GLP_FX, ub is equal to lb */
      GLPAIJ *ptr; /* non-scaled */
      /* pointer to doubly linked list of constraint coefficients which
         are placed in this row */
      double rii;
      /* diagonal element r[i,i] of scaling matrix R for this row;
         if the scaling is not used, r[i,i] is 1 */
      int stat;
      /* status of the auxiliary variable:
         GLP_BS - basic variable
         GLP_NL - non-basic variable on lower bound
         GLP_NU - non-basic variable on upper bound
         GLP_NF - non-basic free variable
         GLP_NS - non-basic fixed variable */
      int bind;
      /* if the auxiliary variable is basic, head[bind] refers to this
         row, otherwise, bind is 0; this attribute is valid only if the
         basis factorization is valid */
      double prim; /* non-scaled */
      /* primal value of the auxiliary variable in basic solution */
      double dual; /* non-scaled */
      /* dual value of the auxiliary variable in basic solution */
      double pval; /* non-scaled */
      /* primal value of the auxiliary variable in interior solution */
      double dval; /* non-scaled */
      /* dual value of the auxiliary variable in interior solution */
      double mipx; /* non-scaled */
      /* primal value of the auxiliary variable in integer solution */
};

struct GLPCOL
{     /* LP/MIP column (structural variable) */
      int j;
      /* ordinal number (1 to n) assigned to this column */
      char *name;
      /* column name (1 to 255 chars); NULL means no name is assigned
         to this column */
//       AVLNODE *node;
      /* pointer to corresponding node in the column index; NULL means
         that either the column index does not exist or the column has
         no name assigned */
      int kind;
      /* kind of the structural variable:
         GLP_CV - continuous variable
         GLP_IV - integer or binary variable */
      int type;
      /* type of the structural variable:
         GLP_FR - free variable
         GLP_LO - variable with lower bound
         GLP_UP - variable with upper bound
         GLP_DB - double-bounded variable
         GLP_FX - fixed variable */
      double lb; /* non-scaled */
      /* lower bound; if the column has no lower bound, lb is zero */
      double ub; /* non-scaled */
      /* upper bound; if the column has no upper bound, ub is zero */
      /* if the column type is GLP_FX, ub is equal to lb */
      double coef; /* non-scaled */
      /* objective coefficient at the structural variable */
      GLPAIJ *ptr; /* non-scaled */
      /* pointer to doubly linked list of constraint coefficients which
         are placed in this column */
      double sjj;
      /* diagonal element s[j,j] of scaling matrix S for this column;
         if the scaling is not used, s[j,j] is 1 */
      int stat;
      /* status of the structural variable:
         GLP_BS - basic variable
         GLP_NL - non-basic variable on lower bound
         GLP_NU - non-basic variable on upper bound
         GLP_NF - non-basic free variable
         GLP_NS - non-basic fixed variable */
      int bind;
      /* if the structural variable is basic, head[bind] refers to
         this column; otherwise, bind is 0; this attribute is valid only
         if the basis factorization is valid */
      double prim; /* non-scaled */
      /* primal value of the structural variable in basic solution */
      double dual; /* non-scaled */
      /* dual value of the structural variable in basic solution */
      double pval; /* non-scaled */
      /* primal value of the structural variable in interior solution */
      double dval; /* non-scaled */
      /* dual value of the structural variable in interior solution */
      double mipx; /* non-scaled */
      /* primal value of the structural variable in integer solution */
};

struct GLPAIJ
{     /* constraint coefficient a[i,j] */
      GLPROW *row;
      /* pointer to row, where this coefficient is placed */
      GLPCOL *col;
      /* pointer to column, where this coefficient is placed */
      double val;
      /* numeric (non-zero) value of this coefficient */
      GLPAIJ *r_prev;
      /* pointer to previous coefficient in the same row */
      GLPAIJ *r_next;
      /* pointer to next coefficient in the same row */
      GLPAIJ *c_prev;
      /* pointer to previous coefficient in the same column */
      GLPAIJ *c_next;
      /* pointer to next coefficient in the same column */
};

struct glp_prob
{     /* LP/MIP problem object */
      int magic;
      /* magic value used for debugging */
      DMP *pool;
      /* memory pool to store problem object components */
//       glp_tree *tree;
      /* pointer to the search tree; set by the MIP solver when this
         object is used in the tree as a core MIP object */
      void *parms;
      /* reserved for backward compatibility */
      /*--------------------------------------------------------------*/
      /* LP/MIP data */
      char *name;
      /* problem name (1 to 255 chars); NULL means no name is assigned
         to the problem */
      char *obj;
      /* objective function name (1 to 255 chars); NULL means no name
         is assigned to the objective function */
      int dir;
      /* optimization direction flag (objective "sense"):
         GLP_MIN - minimization
         GLP_MAX - maximization */
      double c0;
      /* constant term of the objective function ("shift") */
      int m_max;
      /* length of the array of rows (enlarged automatically) */
      int n_max;
      /* length of the array of columns (enlarged automatically) */
      int m;
      /* number of rows, 0 <= m <= m_max */
      int n;
      /* number of columns, 0 <= n <= n_max */
      int nnz;
      /* number of non-zero constraint coefficients, nnz >= 0 */
      GLPROW **row; /* GLPROW *row[1+m_max]; */
      /* row[i], 1 <= i <= m, is a pointer to i-th row */
      GLPCOL **col; /* GLPCOL *col[1+n_max]; */
      /* col[j], 1 <= j <= n, is a pointer to j-th column */
//       AVL *r_tree;
      /* row index to find rows by their names; NULL means this index
         does not exist */
//       AVL *c_tree;
      /* column index to find columns by their names; NULL means this
         index does not exist */
      /*--------------------------------------------------------------*/
      /* basis factorization (LP) */
      int valid;
      /* the factorization is valid only if this flag is set */
      int *head; /* int head[1+m_max]; */
      /* basis header (valid only if the factorization is valid);
         head[i] = k is the ordinal number of auxiliary (1 <= k <= m)
         or structural (m+1 <= k <= m+n) variable which corresponds to
         i-th basic variable xB[i], 1 <= i <= m */
      glp_bfcp *bfcp;
      /* basis factorization control parameters; may be NULL */
      BFD *bfd; /* BFD bfd[1:m,1:m]; */
      /* basis factorization driver; may be NULL */
      /*--------------------------------------------------------------*/
      /* basic solution (LP) */
      int pbs_stat;
      /* primal basic solution status:
         GLP_UNDEF  - primal solution is undefined
         GLP_FEAS   - primal solution is feasible
         GLP_INFEAS - primal solution is infeasible
         GLP_NOFEAS - no primal feasible solution exists */
      int dbs_stat;
      /* dual basic solution status:
         GLP_UNDEF  - dual solution is undefined
         GLP_FEAS   - dual solution is feasible
         GLP_INFEAS - dual solution is infeasible
         GLP_NOFEAS - no dual feasible solution exists */
      double obj_val;
      /* objective function value */
      int it_cnt;
      /* simplex method iteration count; increased by one on performing
         one simplex iteration */
      int some;
      /* ordinal number of some auxiliary or structural variable having
         certain property, 0 <= some <= m+n */
      /*--------------------------------------------------------------*/
      /* interior-point solution (LP) */
      int ipt_stat;
      /* interior-point solution status:
         GLP_UNDEF  - interior solution is undefined
         GLP_OPT    - interior solution is optimal
         GLP_INFEAS - interior solution is infeasible
         GLP_NOFEAS - no feasible solution exists */
      double ipt_obj;
      /* objective function value */
      /*--------------------------------------------------------------*/
      /* integer solution (MIP) */
      int mip_stat;
      /* integer solution status:
         GLP_UNDEF  - integer solution is undefined
         GLP_OPT    - integer solution is optimal
         GLP_FEAS   - integer solution is feasible
         GLP_NOFEAS - no integer solution exists */
      double mip_obj;
      /* objective function value */
};

////////// END API HEADER


////////// START BFD

BFD *bfd_create_it(void)
{     BFD *bfd;
      bfd = (BFD*)xmalloc(sizeof(BFD));
      bfd->valid = 0;
      bfd->type = GLP_BF_FT;
      bfd->fhv = NULL;
      bfd->lpf = NULL;
      bfd->lu_size = 0;
      bfd->piv_tol = 0.10;
      bfd->piv_lim = 4;
      bfd->suhl = 1;
      bfd->eps_tol = 1e-15;
      bfd->max_gro = 1e+10;
      bfd->nfs_max = 100;
      bfd->upd_tol = 1e-6;
      bfd->nrs_max = 100;
      bfd->rs_size = 1000;
      bfd->upd_lim = -1;
      bfd->upd_cnt = 0;
      return bfd;
}

void bfd_set_parm(BFD *bfd, const void *_parm)
{     /* change LP basis factorization control parameters */
      const glp_bfcp *parm = (glp_bfcp*)_parm;
      xassert(bfd != NULL);
      bfd->type = parm->type;
      bfd->lu_size = parm->lu_size;
      bfd->piv_tol = parm->piv_tol;
      bfd->piv_lim = parm->piv_lim;
      bfd->suhl = parm->suhl;
      bfd->eps_tol = parm->eps_tol;
      bfd->max_gro = parm->max_gro;
      bfd->nfs_max = parm->nfs_max;
      bfd->upd_tol = parm->upd_tol;
      bfd->nrs_max = parm->nrs_max;
      bfd->rs_size = parm->rs_size;
      return;
}

int bfd_factorize(BFD *bfd, int m, const int bh[], int (*col)
      (void *info, int j, int ind[], double val[]), void *info)
{     LUF *luf = NULL;
      int nov, ret;
      xassert(bfd != NULL);
      xassert(1 <= m && m <= M_MAX);
      /* invalidate the factorization */
      bfd->valid = 0;
      /* create the factorization, if necessary */
      nov = 0;
      switch (bfd->type)
      {  case GLP_BF_FT:
            if (bfd->lpf != NULL)
               lpf_delete_it(bfd->lpf), bfd->lpf = NULL;
            if (bfd->fhv == NULL)
               bfd->fhv = fhv_create_it(), nov = 1;
            break;
         case GLP_BF_BG:
         case GLP_BF_GR:
            if (bfd->fhv != NULL)
               fhv_delete_it(bfd->fhv), bfd->fhv = NULL;
            if (bfd->lpf == NULL)
               bfd->lpf = lpf_create_it(), nov = 1;
            break;
         default:
            xassert(bfd != bfd);
      }
      /* set control parameters specific to LUF */
      if (bfd->fhv != NULL)
         luf = bfd->fhv->luf;
      else if (bfd->lpf != NULL)
         luf = bfd->lpf->luf;
      else
         xassert(bfd != bfd);
      if (nov) luf->new_sva = bfd->lu_size;
      luf->piv_tol = bfd->piv_tol;
      luf->piv_lim = bfd->piv_lim;
      luf->suhl = bfd->suhl;
      luf->eps_tol = bfd->eps_tol;
      luf->max_gro = bfd->max_gro;
      /* set control parameters specific to FHV */
      if (bfd->fhv != NULL)
      {  if (nov) bfd->fhv->hh_max = bfd->nfs_max;
         bfd->fhv->upd_tol = bfd->upd_tol;
      }
      /* set control parameters specific to LPF */
      if (bfd->lpf != NULL)
      {  if (nov) bfd->lpf->n_max = bfd->nrs_max;
         if (nov) bfd->lpf->v_size = bfd->rs_size;
      }
      /* try to factorize the basis matrix */
      if (bfd->fhv != NULL)
      {  switch (fhv_factorize(bfd->fhv, m, col, info))
         {  case 0:
               break;
            case FHV_ESING:
               ret = BFD_ESING;
               goto done;
            case FHV_ECOND:
               ret = BFD_ECOND;
               goto done;
            default:
               xassert(bfd != bfd);
         }
      }
      else if (bfd->lpf != NULL)
      {  switch (lpf_factorize(bfd->lpf, m, bh, col, info))
         {  case 0:
               /* set the Schur complement update type */
               switch (bfd->type)
               {  case GLP_BF_BG:
                     /* Bartels-Golub update */
                     bfd->lpf->scf->t_opt = SCF_TBG;
                     break;
                  case GLP_BF_GR:
                     /* Givens rotation update */
                     bfd->lpf->scf->t_opt = SCF_TGR;
                     break;
                  default:
                     xassert(bfd != bfd);
               }
               break;
            case LPF_ESING:
               ret = BFD_ESING;
               goto done;
            case LPF_ECOND:
               ret = BFD_ECOND;
               goto done;
            default:
               xassert(bfd != bfd);
         }
      }
      else
         xassert(bfd != bfd);
      /* the basis matrix has been successfully factorized */
      bfd->valid = 1;
      bfd->upd_cnt = 0;
      ret = 0;
done: /* return to the calling program */
      return ret;
}

void bfd_ftran(BFD *bfd, double x[])
{     xassert(bfd != NULL);
      xassert(bfd->valid);
      if (bfd->fhv != NULL)
         fhv_ftran(bfd->fhv, x);
      else if (bfd->lpf != NULL)
         lpf_ftran(bfd->lpf, x);
      else
         xassert(bfd != bfd);
      return;
}

void bfd_btran(BFD *bfd, double x[])
{     xassert(bfd != NULL);
      xassert(bfd->valid);
      if (bfd->fhv != NULL)
         fhv_btran(bfd->fhv, x);
      else if (bfd->lpf != NULL)
         lpf_btran(bfd->lpf, x);
      else
         xassert(bfd != bfd);
      return;
}

int bfd_update_it(BFD *bfd, int j, int bh, int len, const int ind[],
      const double val[])
{     int ret;
      xassert(bfd != NULL);
      xassert(bfd->valid);
      /* try to update the factorization */
      if (bfd->fhv != NULL)
      {  switch (fhv_update_it(bfd->fhv, j, len, ind, val))
         {  case 0:
               break;
            case FHV_ESING:
               bfd->valid = 0;
               ret = BFD_ESING;
               goto done;
            case FHV_ECHECK:
               bfd->valid = 0;
               ret = BFD_ECHECK;
               goto done;
            case FHV_ELIMIT:
               bfd->valid = 0;
               ret = BFD_ELIMIT;
               goto done;
            case FHV_EROOM:
               bfd->valid = 0;
               ret = BFD_EROOM;
               goto done;
            default:
               xassert(bfd != bfd);
         }
      }
      else if (bfd->lpf != NULL)
      {  switch (lpf_update_it(bfd->lpf, j, bh, len, ind, val))
         {  case 0:
               break;
            case LPF_ESING:
               bfd->valid = 0;
               ret = BFD_ESING;
               goto done;
            case LPF_ELIMIT:
               bfd->valid = 0;
               ret = BFD_ELIMIT;
               goto done;
            default:
               xassert(bfd != bfd);
         }
      }
      else
         xassert(bfd != bfd);
      /* the factorization has been successfully updated */
      /* increase the update count */
      bfd->upd_cnt++;
      ret = 0;
done: /* return to the calling program */
      return ret;
}

int bfd_get_count(BFD *bfd)
{     /* determine factorization update count */
      xassert(bfd != NULL);
      xassert(bfd->valid);
      return bfd->upd_cnt;
}

void bfd_delete_it(BFD *bfd)
{     xassert(bfd != NULL);
      if (bfd->fhv != NULL)
         fhv_delete_it(bfd->fhv);
      if (bfd->lpf != NULL)
         lpf_delete_it(bfd->lpf);
      xfree(bfd);
      return;
}

////////// END BFD


////////// START SPX

#define spx_primal _glp_spx_primal

struct csa
{     /* common storage area */
      /*--------------------------------------------------------------*/
      /* LP data */
      int m;
      /* number of rows (auxiliary variables), m > 0 */
      int n;
      /* number of columns (structural variables), n > 0 */
      char *type; /* char type[1+m+n]; */
      /* type[0] is not used;
         type[k], 1 <= k <= m+n, is the type of variable x[k]:
         GLP_FR - free variable
         GLP_LO - variable with lower bound
         GLP_UP - variable with upper bound
         GLP_DB - double-bounded variable
         GLP_FX - fixed variable */
      double *lb; /* double lb[1+m+n]; */
      /* lb[0] is not used;
         lb[k], 1 <= k <= m+n, is an lower bound of variable x[k];
         if x[k] has no lower bound, lb[k] is zero */
      double *ub; /* double ub[1+m+n]; */
      /* ub[0] is not used;
         ub[k], 1 <= k <= m+n, is an upper bound of variable x[k];
         if x[k] has no upper bound, ub[k] is zero;
         if x[k] is of fixed type, ub[k] is the same as lb[k] */
      double *coef; /* double coef[1+m+n]; */
      /* coef[0] is not used;
         coef[k], 1 <= k <= m+n, is an objective coefficient at
         variable x[k] (note that on phase I auxiliary variables also
         may have non-zero objective coefficients) */
      /*--------------------------------------------------------------*/
      /* original objective function */
      double *obj; /* double obj[1+n]; */
      /* obj[0] is a constant term of the original objective function;
         obj[j], 1 <= j <= n, is an original objective coefficient at
         structural variable x[m+j] */
      double zeta;
      /* factor used to scale original objective coefficients; its
         sign defines original optimization direction: zeta > 0 means
         minimization, zeta < 0 means maximization */
      /*--------------------------------------------------------------*/
      /* constraint matrix A; it has m rows and n columns and is stored
         by columns */
      int *A_ptr; /* int A_ptr[1+n+1]; */
      /* A_ptr[0] is not used;
         A_ptr[j], 1 <= j <= n, is starting position of j-th column in
         arrays A_ind and A_val; note that A_ptr[1] is always 1;
         A_ptr[n+1] indicates the position after the last element in
         arrays A_ind and A_val */
      int *A_ind; /* int A_ind[A_ptr[n+1]]; */
      /* row indices */
      double *A_val; /* double A_val[A_ptr[n+1]]; */
      /* non-zero element values */
      /*--------------------------------------------------------------*/
      /* basis header */
      int *head; /* int head[1+m+n]; */
      /* head[0] is not used;
         head[i], 1 <= i <= m, is the ordinal number of basic variable
         xB[i]; head[i] = k means that xB[i] = x[k] and i-th column of
         matrix B is k-th column of matrix (I|-A);
         head[m+j], 1 <= j <= n, is the ordinal number of non-basic
         variable xN[j]; head[m+j] = k means that xN[j] = x[k] and j-th
         column of matrix N is k-th column of matrix (I|-A) */
      char *stat; /* char stat[1+n]; */
      /* stat[0] is not used;
         stat[j], 1 <= j <= n, is the status of non-basic variable
         xN[j], which defines its active bound:
         GLP_NL - lower bound is active
         GLP_NU - upper bound is active
         GLP_NF - free variable
         GLP_NS - fixed variable */
      /*--------------------------------------------------------------*/
      /* matrix B is the basis matrix; it is composed from columns of
         the augmented constraint matrix (I|-A) corresponding to basic
         variables and stored in a factorized (invertable) form */
      int valid;
      /* factorization is valid only if this flag is set */
      BFD *bfd; /* BFD bfd[1:m,1:m]; */
      /* factorized (invertable) form of the basis matrix */
      /*--------------------------------------------------------------*/
      /* matrix N is a matrix composed from columns of the augmented
         constraint matrix (I|-A) corresponding to non-basic variables
         except fixed ones; it is stored by rows and changes every time
         the basis changes */
      int *N_ptr; /* int N_ptr[1+m+1]; */
      /* N_ptr[0] is not used;
         N_ptr[i], 1 <= i <= m, is starting position of i-th row in
         arrays N_ind and N_val; note that N_ptr[1] is always 1;
         N_ptr[m+1] indicates the position after the last element in
         arrays N_ind and N_val */
      int *N_len; /* int N_len[1+m]; */
      /* N_len[0] is not used;
         N_len[i], 1 <= i <= m, is length of i-th row (0 to n) */
      int *N_ind; /* int N_ind[N_ptr[m+1]]; */
      /* column indices */
      double *N_val; /* double N_val[N_ptr[m+1]]; */
      /* non-zero element values */
      /*--------------------------------------------------------------*/
      /* working parameters */
      int phase;
      /* search phase:
         0 - not determined yet
         1 - search for primal feasible solution
         2 - search for optimal solution */
      glp_long tm_beg;
      /* time value at the beginning of the search */
      int it_beg;
      /* simplex iteration count at the beginning of the search */
      int it_cnt;
      /* simplex iteration count; it increases by one every time the
         basis changes (including the case when a non-basic variable
         jumps to its opposite bound) */
      int it_dpy;
      /* simplex iteration count at the most recent display output */
      /*--------------------------------------------------------------*/
      /* basic solution components */
      double *bbar; /* double bbar[1+m]; */
      /* bbar[0] is not used;
         bbar[i], 1 <= i <= m, is primal value of basic variable xB[i]
         (if xB[i] is free, its primal value is not updated) */
      double *cbar; /* double cbar[1+n]; */
      /* cbar[0] is not used;
         cbar[j], 1 <= j <= n, is reduced cost of non-basic variable
         xN[j] (if xN[j] is fixed, its reduced cost is not updated) */
      /*--------------------------------------------------------------*/
      /* the following pricing technique options may be used:
         GLP_PT_STD - standard ("textbook") pricing;
         GLP_PT_PSE - projected steepest edge;
         GLP_PT_DVX - Devex pricing (not implemented yet);
         in case of GLP_PT_STD the reference space is not used, and all
         steepest edge coefficients are set to 1 */
      int refct;
      /* this count is set to an initial value when the reference space
         is defined and decreases by one every time the basis changes;
         once this count reaches zero, the reference space is redefined
         again */
      char *refsp; /* char refsp[1+m+n]; */
      /* refsp[0] is not used;
         refsp[k], 1 <= k <= m+n, is the flag which means that variable
         x[k] belongs to the current reference space */
      double *gamma; /* double gamma[1+n]; */
      /* gamma[0] is not used;
         gamma[j], 1 <= j <= n, is the steepest edge coefficient for
         non-basic variable xN[j]; if xN[j] is fixed, gamma[j] is not
         used and just set to 1 */
      /*--------------------------------------------------------------*/
      /* non-basic variable xN[q] chosen to enter the basis */
      int q;
      /* index of the non-basic variable xN[q] chosen, 1 <= q <= n;
         if the set of eligible non-basic variables is empty and thus
         no variable has been chosen, q is set to 0 */
      /*--------------------------------------------------------------*/
      /* pivot column of the simplex table corresponding to non-basic
         variable xN[q] chosen is the following vector:
            T * e[q] = - inv(B) * N * e[q] = - inv(B) * N[q],
         where B is the current basis matrix, N[q] is a column of the
         matrix (I|-A) corresponding to xN[q] */
      int tcol_nnz;
      /* number of non-zero components, 0 <= nnz <= m */
      int *tcol_ind; /* int tcol_ind[1+m]; */
      /* tcol_ind[0] is not used;
         tcol_ind[t], 1 <= t <= nnz, is an index of non-zero component,
         i.e. tcol_ind[t] = i means that tcol_vec[i] != 0 */
      double *tcol_vec; /* double tcol_vec[1+m]; */
      /* tcol_vec[0] is not used;
         tcol_vec[i], 1 <= i <= m, is a numeric value of i-th component
         of the column */
      double tcol_max;
      /* infinity (maximum) norm of the column (max |tcol_vec[i]|) */
      int tcol_num;
      /* number of significant non-zero components, which means that:
         |tcol_vec[i]| >= eps for i in tcol_ind[1,...,num],
         |tcol_vec[i]| <  eps for i in tcol_ind[num+1,...,nnz],
         where eps is a pivot tolerance */
      /*--------------------------------------------------------------*/
      /* basic variable xB[p] chosen to leave the basis */
      int p;
      /* index of the basic variable xB[p] chosen, 1 <= p <= m;
         p = 0 means that no basic variable reaches its bound;
         p < 0 means that non-basic variable xN[q] reaches its opposite
         bound before any basic variable */
      int p_stat;
      /* new status (GLP_NL, GLP_NU, or GLP_NS) to be assigned to xB[p]
         once it has left the basis */
      double teta;
      /* change of non-basic variable xN[q] (see above), on which xB[p]
         (or, if p < 0, xN[q] itself) reaches its bound */
      /*--------------------------------------------------------------*/
      /* pivot row of the simplex table corresponding to basic variable
         xB[p] chosen is the following vector:
            T' * e[p] = - N' * inv(B') * e[p] = - N' * rho,
         where B' is a matrix transposed to the current basis matrix,
         N' is a matrix, whose rows are columns of the matrix (I|-A)
         corresponding to non-basic non-fixed variables */
      int trow_nnz;
      /* number of non-zero components, 0 <= nnz <= n */
      int *trow_ind; /* int trow_ind[1+n]; */
      /* trow_ind[0] is not used;
         trow_ind[t], 1 <= t <= nnz, is an index of non-zero component,
         i.e. trow_ind[t] = j means that trow_vec[j] != 0 */
      double *trow_vec; /* int trow_vec[1+n]; */
      /* trow_vec[0] is not used;
         trow_vec[j], 1 <= j <= n, is a numeric value of j-th component
         of the row */
      /*--------------------------------------------------------------*/
      /* working arrays */
      double *work1; /* double work1[1+m]; */
      double *work2; /* double work2[1+m]; */
      double *work3; /* double work3[1+m]; */
      double *work4; /* double work4[1+m]; */
};

const double kappa = 0.10;

csa *alloc_csa(struct glp_prob *lp)
{     csa *thecsa;
      int m = lp->m;
      int n = lp->n;
      int nnz = lp->nnz;
      thecsa = (csa*)xmalloc(sizeof(csa));
      xassert(m > 0 && n > 0);
      thecsa->m = m;
      thecsa->n = n;
      thecsa->type = (char*)xcalloc(1+m+n, sizeof(char));
      thecsa->lb = (double*)xcalloc(1+m+n, sizeof(double));
      thecsa->ub = (double*)xcalloc(1+m+n, sizeof(double));
      thecsa->coef = (double*)xcalloc(1+m+n, sizeof(double));
      thecsa->obj = (double*)xcalloc(1+n, sizeof(double));
      thecsa->A_ptr = (int*)xcalloc(1+n+1, sizeof(int));
      thecsa->A_ind = (int*)xcalloc(1+nnz, sizeof(int));
      thecsa->A_val = (double*)xcalloc(1+nnz, sizeof(double));
      thecsa->head = (int*)xcalloc(1+m+n, sizeof(int));
      thecsa->stat = (char*)xcalloc(1+n, sizeof(char));
      thecsa->N_ptr = (int*)xcalloc(1+m+1, sizeof(int));
      thecsa->N_len = (int*)xcalloc(1+m, sizeof(int));
      thecsa->N_ind = NULL; /* will be allocated later */
      thecsa->N_val = NULL; /* will be allocated later */
      thecsa->bbar = (double*)xcalloc(1+m, sizeof(double));
      thecsa->cbar = (double*)xcalloc(1+n, sizeof(double));
      thecsa->refsp = (char*)xcalloc(1+m+n, sizeof(char));
      thecsa->gamma = (double*)xcalloc(1+n, sizeof(double));
      thecsa->tcol_ind = (int*)xcalloc(1+m, sizeof(int));
      thecsa->tcol_vec = (double*)xcalloc(1+m, sizeof(double));
      thecsa->trow_ind = (int*)xcalloc(1+n, sizeof(int));
      thecsa->trow_vec = (double*)xcalloc(1+n, sizeof(double));
      thecsa->work1 = (double*)xcalloc(1+m, sizeof(double));
      thecsa->work2 = (double*)xcalloc(1+m, sizeof(double));
      thecsa->work3 = (double*)xcalloc(1+m, sizeof(double));
      thecsa->work4 = (double*)xcalloc(1+m, sizeof(double));
      return thecsa;
}

void alloc_N(csa *csa);
void build_N(csa *csa);

void init_csa(csa *thecsa, glp_prob *lp)
{     int m = thecsa->m;
      int n = thecsa->n;
      char *type = thecsa->type;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      double *coef = thecsa->coef;
      double *obj = thecsa->obj;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      double *A_val = thecsa->A_val;
      int *head = thecsa->head;
      char *stat = thecsa->stat;
      char *refsp = thecsa->refsp;
      double *gamma = thecsa->gamma;
      int i, j, k, loc;
      double cmax;
      /* auxiliary variables */
      for (i = 1; i <= m; i++)
      {  GLPROW *row = lp->row[i];
         type[i] = (char)row->type;
         lb[i] = row->lb * row->rii;
         ub[i] = row->ub * row->rii;
         coef[i] = 0.0;
      }
      /* structural variables */
      for (j = 1; j <= n; j++)
      {  GLPCOL *col = lp->col[j];
         type[m+j] = (char)col->type;
         lb[m+j] = col->lb / col->sjj;
         ub[m+j] = col->ub / col->sjj;
         coef[m+j] = col->coef * col->sjj;
      }
      /* original objective function */
      obj[0] = lp->c0;
      memcpy(&obj[1], &coef[m+1], n * sizeof(double));
      /* factor used to scale original objective coefficients */
      cmax = 0.0;
      for (j = 1; j <= n; j++)
         if (cmax < fabs(obj[j])) cmax = fabs(obj[j]);
      if (cmax == 0.0) cmax = 1.0;
      switch (lp->dir)
      {  case GLP_MIN:
            thecsa->zeta = + 1.0 / cmax;
            break;
         case GLP_MAX:
            thecsa->zeta = - 1.0 / cmax;
            break;
         default:
            xassert(lp != lp);
      }
      if (fabs(thecsa->zeta) < 1.0) thecsa->zeta *= 1000.0;
      /* matrix A (by columns) */
      loc = 1;
      for (j = 1; j <= n; j++)
      {  GLPAIJ *aij;
         A_ptr[j] = loc;
         for (aij = lp->col[j]->ptr; aij != NULL; aij = aij->c_next)
         {  A_ind[loc] = aij->row->i;
            A_val[loc] = aij->row->rii * aij->val * aij->col->sjj;
            loc++;
         }
      }
      A_ptr[n+1] = loc;
      xassert(loc == lp->nnz+1);
      /* basis header */
      xassert(lp->valid);
      memcpy(&head[1], &lp->head[1], m * sizeof(int));
      k = 0;
      for (i = 1; i <= m; i++)
      {  GLPROW *row = lp->row[i];
         if (row->stat != GLP_BS)
         {  k++;
            xassert(k <= n);
            head[m+k] = i;
            stat[k] = (char)row->stat;
         }
      }
      for (j = 1; j <= n; j++)
      {  GLPCOL *col = lp->col[j];
         if (col->stat != GLP_BS)
         {  k++;
            xassert(k <= n);
            head[m+k] = m + j;
            stat[k] = (char)col->stat;
         }
      }
      xassert(k == n);
      /* factorization of matrix B */
      thecsa->valid = 1, lp->valid = 0;
      thecsa->bfd = lp->bfd, lp->bfd = NULL;
      /* matrix N (by rows) */
      alloc_N(thecsa);
      build_N(thecsa);
      /* working parameters */
      thecsa->phase = 0;
      thecsa->tm_beg = xtime();
      thecsa->it_beg = thecsa->it_cnt = lp->it_cnt;
      thecsa->it_dpy = -1;
      /* reference space and steepest edge coefficients */
      thecsa->refct = 0;
      memset(&refsp[1], 0, (m+n) * sizeof(char));
      for (j = 1; j <= n; j++) gamma[j] = 1.0;
      return;
}

int inv_col(void *info, int i, int ind[], double val[])
{     /* this auxiliary routine returns row indices and numeric values
         of non-zero elements of i-th column of the basis matrix */
      csa *thecsa = (csa*)info;
      int m = thecsa->m;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      double *A_val = thecsa->A_val;
      int *head = thecsa->head;
      int k, len, ptr, t;
      k = head[i]; /* B[i] is k-th column of (I|-A) */
      if (k <= m)
      {  /* B[i] is k-th column of submatrix I */
         len = 1;
         ind[1] = k;
         val[1] = 1.0;
      }
      else
      {  /* B[i] is (k-m)-th column of submatrix (-A) */
         ptr = A_ptr[k-m];
         len = A_ptr[k-m+1] - ptr;
         memcpy(&ind[1], &A_ind[ptr], len * sizeof(int));
         memcpy(&val[1], &A_val[ptr], len * sizeof(double));
         for (t = 1; t <= len; t++) val[t] = - val[t];
      }
      return len;
}

int invert_B(csa *thecsa)
{     int ret;
      ret = bfd_factorize(thecsa->bfd, thecsa->m, NULL, inv_col, thecsa);
      thecsa->valid = (ret == 0);
      return ret;
}

int update_B(csa *thecsa, int i, int k)
{     int m = thecsa->m;
      int ret;
      if (k <= m)
      {  /* new i-th column of B is k-th column of I */
         int ind[1+1];
         double val[1+1];
         ind[1] = k;
         val[1] = 1.0;
         xassert(thecsa->valid);
         ret = bfd_update_it(thecsa->bfd, i, 0, 1, ind, val);
      }
      else
      {  /* new i-th column of B is (k-m)-th column of (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         double *A_val = thecsa->A_val;
         double *val = thecsa->work1;
         int beg, end, ptr, len;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         len = 0;
         for (ptr = beg; ptr < end; ptr++)
            val[++len] = - A_val[ptr];
         xassert(thecsa->valid);
         ret = bfd_update_it(thecsa->bfd, i, 0, len, &A_ind[beg-1], val);
      }
      thecsa->valid = (ret == 0);
      return ret;
}

void error_ftran(csa *thecsa, double h[], double x[],
      double r[])
{     int m = thecsa->m;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      double *A_val = thecsa->A_val;
      int *head = thecsa->head;
      int i, k, beg, end, ptr;
      double temp;
      /* compute the residual vector:
         r = h - B * x = h - B[1] * x[1] - ... - B[m] * x[m],
         where B[1], ..., B[m] are columns of matrix B */
      memcpy(&r[1], &h[1], m * sizeof(double));
      for (i = 1; i <= m; i++)
      {  temp = x[i];
         if (temp == 0.0) continue;
         k = head[i]; /* B[i] is k-th column of (I|-A) */
         if (k <= m)
         {  /* B[i] is k-th column of submatrix I */
            r[k] -= temp;
         }
         else
         {  /* B[i] is (k-m)-th column of submatrix (-A) */
            beg = A_ptr[k-m];
            end = A_ptr[k-m+1];
            for (ptr = beg; ptr < end; ptr++)
               r[A_ind[ptr]] += A_val[ptr] * temp;
         }
      }
      return;
}

void refine_ftran(csa *thecsa, double h[], double x[])
{     int m = thecsa->m;
      double *r = thecsa->work1;
      double *d = thecsa->work1;
      int i;
      /* compute the residual vector r = h - B * x */
      error_ftran(thecsa, h, x, r);
      /* compute the correction vector d = inv(B) * r */
      xassert(thecsa->valid);
      bfd_ftran(thecsa->bfd, d);
      /* refine the solution vector (new x) = (old x) + d */
      for (i = 1; i <= m; i++) x[i] += d[i];
      return;
}

void error_btran(csa *thecsa, double h[], double x[],
      double r[])
{     int m = thecsa->m;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      double *A_val = thecsa->A_val;
      int *head = thecsa->head;
      int i, k, beg, end, ptr;
      double temp;
      /* compute the residual vector r = b - B'* x */
      for (i = 1; i <= m; i++)
      {  /* r[i] := b[i] - (i-th column of B)'* x */
         k = head[i]; /* B[i] is k-th column of (I|-A) */
         temp = h[i];
         if (k <= m)
         {  /* B[i] is k-th column of submatrix I */
            temp -= x[k];
         }
         else
         {  /* B[i] is (k-m)-th column of submatrix (-A) */
            beg = A_ptr[k-m];
            end = A_ptr[k-m+1];
            for (ptr = beg; ptr < end; ptr++)
               temp += A_val[ptr] * x[A_ind[ptr]];
         }
         r[i] = temp;
      }
      return;
}

void refine_btran(csa *thecsa, double h[], double x[])
{     int m = thecsa->m;
      double *r = thecsa->work1;
      double *d = thecsa->work1;
      int i;
      /* compute the residual vector r = h - B'* x */
      error_btran(thecsa, h, x, r);
      /* compute the correction vector d = inv(B') * r */
      xassert(thecsa->valid);
      bfd_btran(thecsa->bfd, d);
      /* refine the solution vector (new x) = (old x) + d */
      for (i = 1; i <= m; i++) x[i] += d[i];
      return;
}

void alloc_N(csa *thecsa)
{     int m = thecsa->m;
      int n = thecsa->n;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      int *N_ptr = thecsa->N_ptr;
      int *N_len = thecsa->N_len;
      int i, j, beg, end, ptr;
      /* determine number of non-zeros in each row of the augmented
         constraint matrix (I|-A) */
      for (i = 1; i <= m; i++)
         N_len[i] = 1;
      for (j = 1; j <= n; j++)
      {  beg = A_ptr[j];
         end = A_ptr[j+1];
         for (ptr = beg; ptr < end; ptr++)
            N_len[A_ind[ptr]]++;
      }
      /* determine maximal row lengths of matrix N and set its row
         pointers */
      N_ptr[1] = 1;
      for (i = 1; i <= m; i++)
      {  /* row of matrix N cannot have more than n non-zeros */
         if (N_len[i] > n) N_len[i] = n;
         N_ptr[i+1] = N_ptr[i] + N_len[i];
      }
      /* now maximal number of non-zeros in matrix N is known */
      thecsa->N_ind = (int*)xcalloc(N_ptr[m+1], sizeof(int));
      thecsa->N_val = (double*)xcalloc(N_ptr[m+1], sizeof(double));
      return;
}

void add_N_col(csa *thecsa, int j, int k)
{     int m = thecsa->m;
      int *N_ptr = thecsa->N_ptr;
      int *N_len = thecsa->N_len;
      int *N_ind = thecsa->N_ind;
      double *N_val = thecsa->N_val;
      int pos;
      if (k <= m)
      {  /* N[j] is k-th column of submatrix I */
         pos = N_ptr[k] + (N_len[k]++);
         N_ind[pos] = j;
         N_val[pos] = 1.0;
      }
      else
      {  /* N[j] is (k-m)-th column of submatrix (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         double *A_val = thecsa->A_val;
         int i, beg, end, ptr;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         for (ptr = beg; ptr < end; ptr++)
         {  i = A_ind[ptr]; /* row number */
            pos = N_ptr[i] + (N_len[i]++);
            N_ind[pos] = j;
            N_val[pos] = - A_val[ptr];
         }
      }
      return;
}

void del_N_col(csa *thecsa, int j, int k)
{     int m = thecsa->m;
      int *N_ptr = thecsa->N_ptr;
      int *N_len = thecsa->N_len;
      int *N_ind = thecsa->N_ind;
      double *N_val = thecsa->N_val;
      int pos, head, tail;
      if (k <= m)
      {  /* N[j] is k-th column of submatrix I */
         /* find element in k-th row of N */
         head = N_ptr[k];
         for (pos = head; N_ind[pos] != j; pos++) /* nop */;
         /* and remove it from the row list */
         tail = head + (--N_len[k]);
         N_ind[pos] = N_ind[tail];
         N_val[pos] = N_val[tail];
      }
      else
      {  /* N[j] is (k-m)-th column of submatrix (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         int i, beg, end, ptr;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         for (ptr = beg; ptr < end; ptr++)
         {  i = A_ind[ptr]; /* row number */
            /* find element in i-th row of N */
            head = N_ptr[i];
            for (pos = head; N_ind[pos] != j; pos++) /* nop */;
            /* and remove it from the row list */
            tail = head + (--N_len[i]);
            N_ind[pos] = N_ind[tail];
            N_val[pos] = N_val[tail];
         }
      }
      return;
}

void build_N(csa *thecsa)
{     int m = thecsa->m;
      int n = thecsa->n;
      int *head = thecsa->head;
      char *stat = thecsa->stat;
      int *N_len = thecsa->N_len;
      int j, k;
      /* N := empty matrix */
      memset(&N_len[1], 0, m * sizeof(int));
      /* go through non-basic columns of matrix (I|-A) */
      for (j = 1; j <= n; j++)
      {  if (stat[j] != GLP_NS)
         {  /* xN[j] is non-fixed; add j-th column to matrix N which is
               k-th column of matrix (I|-A) */
            k = head[m+j]; /* x[k] = xN[j] */
            add_N_col(thecsa, j, k);
         }
      }
      return;
}

double get_xN(csa *thecsa, int j)
{     int m = thecsa->m;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      int *head = thecsa->head;
      char *stat = thecsa->stat;
      int k;
      double xN = 0.0;
      k = head[m+j]; /* x[k] = xN[j] */
      switch (stat[j])
      {  case GLP_NL:
            /* x[k] is on its lower bound */
            xN = lb[k]; break;
         case GLP_NU:
            /* x[k] is on its upper bound */
            xN = ub[k]; break;
         case GLP_NF:
            /* x[k] is free non-basic variable */
            xN = 0.0; break;
         case GLP_NS:
            /* x[k] is fixed non-basic variable */
            xN = lb[k]; break;
         default:
            xassert(stat != stat);
      }
      return xN;
}

void eval_beta(csa *thecsa, double beta[])
{     int m = thecsa->m;
      int n = thecsa->n;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      double *A_val = thecsa->A_val;
      int *head = thecsa->head;
      double *h = thecsa->work2;
      int i, j, k, beg, end, ptr;
      double xN;
      /* compute the right-hand side vector:
         h := - N * xN = - N[1] * xN[1] - ... - N[n] * xN[n],
         where N[1], ..., N[n] are columns of matrix N */
      for (i = 1; i <= m; i++)
         h[i] = 0.0;
      for (j = 1; j <= n; j++)
      {  k = head[m+j]; /* x[k] = xN[j] */
         /* determine current value of xN[j] */
         xN = get_xN(thecsa, j);
         if (xN == 0.0) continue;
         if (k <= m)
         {  /* N[j] is k-th column of submatrix I */
            h[k] -= xN;
         }
         else
         {  /* N[j] is (k-m)-th column of submatrix (-A) */
            beg = A_ptr[k-m];
            end = A_ptr[k-m+1];
            for (ptr = beg; ptr < end; ptr++)
               h[A_ind[ptr]] += xN * A_val[ptr];
         }
      }
      /* solve system B * beta = h */
      memcpy(&beta[1], &h[1], m * sizeof(double));
      xassert(thecsa->valid);
      bfd_ftran(thecsa->bfd, beta);
      /* and refine the solution */
      refine_ftran(thecsa, h, beta);
      return;
}

void eval_pi(csa *thecsa, double pi[])
{     int m = thecsa->m;
      double *c = thecsa->coef;
      int *head = thecsa->head;
      double *cB = thecsa->work2;
      int i;
      /* construct the right-hand side vector cB */
      for (i = 1; i <= m; i++)
         cB[i] = c[head[i]];
      /* solve system B'* pi = cB */
      memcpy(&pi[1], &cB[1], m * sizeof(double));
      xassert(thecsa->valid);
      bfd_btran(thecsa->bfd, pi);
      /* and refine the solution */
      refine_btran(thecsa, cB, pi);
      return;
}

double eval_cost(csa *thecsa, double pi[], int j)
{     int m = thecsa->m;
      double *coef = thecsa->coef;
      int *head = thecsa->head;
      int k;
      double dj;
      k = head[m+j]; /* x[k] = xN[j] */
      dj = coef[k];
      if (k <= m)
      {  /* N[j] is k-th column of submatrix I */
         dj -= pi[k];
      }
      else
      {  /* N[j] is (k-m)-th column of submatrix (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         double *A_val = thecsa->A_val;
         int beg, end, ptr;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         for (ptr = beg; ptr < end; ptr++)
            dj += A_val[ptr] * pi[A_ind[ptr]];
      }
      return dj;
}

void eval_bbar(csa *thecsa)
{     eval_beta(thecsa, thecsa->bbar);
      return;
}

void eval_cbar(csa *thecsa)
{
      int n = thecsa->n;
      double *cbar = thecsa->cbar;
      double *pi = thecsa->work3;
      int j;
      /* compute simplex multipliers */
      eval_pi(thecsa, pi);
      /* compute and store reduced costs */
      for (j = 1; j <= n; j++)
      {
         cbar[j] = eval_cost(thecsa, pi, j);
      }
      return;
}

void reset_refsp(csa *thecsa)
{     int m = thecsa->m;
      int n = thecsa->n;
      int *head = thecsa->head;
      char *refsp = thecsa->refsp;
      double *gamma = thecsa->gamma;
      int j, k;
      xassert(thecsa->refct == 0);
      thecsa->refct = 1000;
      memset(&refsp[1], 0, (m+n) * sizeof(char));
      for (j = 1; j <= n; j++)
      {  k = head[m+j]; /* x[k] = xN[j] */
         refsp[k] = 1;
         gamma[j] = 1.0;
      }
      return;
}

double eval_gamma(csa *thecsa, int j)
{     int m = thecsa->m;
      int *head = thecsa->head;
      char *refsp = thecsa->refsp;
      double *alfa = thecsa->work3;
      double *h = thecsa->work3;
      int i, k;
      double gamma;
      k = head[m+j]; /* x[k] = xN[j] */
      /* construct the right-hand side vector h = - N[j] */
      for (i = 1; i <= m; i++)
         h[i] = 0.0;
      if (k <= m)
      {  /* N[j] is k-th column of submatrix I */
         h[k] = -1.0;
      }
      else
      {  /* N[j] is (k-m)-th column of submatrix (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         double *A_val = thecsa->A_val;
         int beg, end, ptr;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         for (ptr = beg; ptr < end; ptr++)
            h[A_ind[ptr]] = A_val[ptr];
      }
      /* solve system B * alfa = h */
      xassert(thecsa->valid);
      bfd_ftran(thecsa->bfd, alfa);
      /* compute gamma */
      gamma = (refsp[k] ? 1.0 : 0.0);
      for (i = 1; i <= m; i++)
      {  k = head[i];
         if (refsp[k]) gamma += alfa[i] * alfa[i];
      }
      return gamma;
}

void chuzc(csa *thecsa, double tol_dj)
{     int n = thecsa->n;
      char *stat = thecsa->stat;
      double *cbar = thecsa->cbar;
      double *gamma = thecsa->gamma;
      int j, q;
      double dj, best, temp;
      /* nothing is chosen so far */
      q = 0, best = 0.0;
      /* look through the list of non-basic variables */
      for (j = 1; j <= n; j++)
      {  dj = cbar[j];
         switch (stat[j])
         {  case GLP_NL:
               /* xN[j] can increase */
               if (dj >= - tol_dj) continue;
               break;
            case GLP_NU:
               /* xN[j] can decrease */
               if (dj <= + tol_dj) continue;
               break;
            case GLP_NF:
               /* xN[j] can change in any direction */
               if (- tol_dj <= dj && dj <= + tol_dj) continue;
               break;
            case GLP_NS:
               /* xN[j] cannot change at all */
               continue;
            default:
               xassert(stat != stat);
         }
         /* xN[j] is eligible non-basic variable; choose one which has
            largest weighted reduced cost */
         temp = (dj * dj) / gamma[j];
         if (best < temp)
            q = j, best = temp;
      }
      /* store the index of non-basic variable xN[q] chosen */
      thecsa->q = q;
      return;
}

void eval_tcol(csa *thecsa)
{     int m = thecsa->m;
      int *head = thecsa->head;
      int q = thecsa->q;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      double *h = thecsa->tcol_vec;
      int i, k, nnz;
      k = head[m+q]; /* x[k] = xN[q] */
      /* construct the right-hand side vector h = - N[q] */
      for (i = 1; i <= m; i++)
         h[i] = 0.0;
      if (k <= m)
      {  /* N[q] is k-th column of submatrix I */
         h[k] = -1.0;
      }
      else
      {  /* N[q] is (k-m)-th column of submatrix (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         double *A_val = thecsa->A_val;
         int beg, end, ptr;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         for (ptr = beg; ptr < end; ptr++)
            h[A_ind[ptr]] = A_val[ptr];
      }
      /* solve system B * tcol = h */
      xassert(thecsa->valid);
      bfd_ftran(thecsa->bfd, tcol_vec);
      /* construct sparse pattern of the pivot column */
      nnz = 0;
      for (i = 1; i <= m; i++)
      {  if (tcol_vec[i] != 0.0)
            tcol_ind[++nnz] = i;
      }
      thecsa->tcol_nnz = nnz;
      return;
}

void refine_tcol(csa *thecsa)
{     int m = thecsa->m;
      int *head = thecsa->head;
      int q = thecsa->q;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      double *h = thecsa->work3;
      int i, k, nnz;
      k = head[m+q]; /* x[k] = xN[q] */
      /* construct the right-hand side vector h = - N[q] */
      for (i = 1; i <= m; i++)
         h[i] = 0.0;
      if (k <= m)
      {  /* N[q] is k-th column of submatrix I */
         h[k] = -1.0;
      }
      else
      {  /* N[q] is (k-m)-th column of submatrix (-A) */
         int *A_ptr = thecsa->A_ptr;
         int *A_ind = thecsa->A_ind;
         double *A_val = thecsa->A_val;
         int beg, end, ptr;
         beg = A_ptr[k-m];
         end = A_ptr[k-m+1];
         for (ptr = beg; ptr < end; ptr++)
            h[A_ind[ptr]] = A_val[ptr];
      }
      /* refine solution of B * tcol = h */
      refine_ftran(thecsa, h, tcol_vec);
      /* construct sparse pattern of the pivot column */
      nnz = 0;
      for (i = 1; i <= m; i++)
      {  if (tcol_vec[i] != 0.0)
            tcol_ind[++nnz] = i;
      }
      thecsa->tcol_nnz = nnz;
      return;
}

void sort_tcol(csa *thecsa, double tol_piv)
{
      int nnz = thecsa->tcol_nnz;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      int i, num, pos;
      double big, eps, temp;
      /* compute infinity (maximum) norm of the column */
      big = 0.0;
      for (pos = 1; pos <= nnz; pos++)
      {
         temp = fabs(tcol_vec[tcol_ind[pos]]);
         if (big < temp) big = temp;
      }
      thecsa->tcol_max = big;
      /* determine absolute pivot tolerance */
      eps = tol_piv * (1.0 + 0.01 * big);
      /* move significant column components to front of the list */
      for (num = 0; num < nnz; )
      {  i = tcol_ind[nnz];
         if (fabs(tcol_vec[i]) < eps)
            nnz--;
         else
         {  num++;
            tcol_ind[nnz] = tcol_ind[num];
            tcol_ind[num] = i;
         }
      }
      thecsa->tcol_num = num;
      return;
}

void chuzr(csa *thecsa, double rtol)
{     int m = thecsa->m;
      char *type = thecsa->type;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      double *coef = thecsa->coef;
      int *head = thecsa->head;
      int phase = thecsa->phase;
      double *bbar = thecsa->bbar;
      double *cbar = thecsa->cbar;
      int q = thecsa->q;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      int tcol_num = thecsa->tcol_num;
      int i, i_stat, k, p, p_stat, pos;
      double alfa, big, delta, s, t, teta, tmax;
      /* s := - sign(d[q]), where d[q] is reduced cost of xN[q] */
      s = (cbar[q] > 0.0 ? -1.0 : +1.0);
      /*** FIRST PASS ***/
      k = head[m+q]; /* x[k] = xN[q] */
      if (type[k] == GLP_DB)
      {  /* xN[q] has both lower and upper bounds */
         p = -1, p_stat = 0, teta = ub[k] - lb[k], big = 1.0;
      }
      else
      {  /* xN[q] has no opposite bound */
         p = 0, p_stat = 0, teta = DBL_MAX, big = 0.0;
      }
      /* walk through significant elements of the pivot column */
      for (pos = 1; pos <= tcol_num; pos++)
      {  i = tcol_ind[pos];
         k = head[i]; /* x[k] = xB[i] */
         alfa = s * tcol_vec[i];
         /* xB[i] = ... + alfa * xN[q] + ..., and due to s we need to
            consider the only case when xN[q] is increasing */
         if (alfa > 0.0)
         {  /* xB[i] is increasing */
            if (phase == 1 && coef[k] < 0.0)
            {  /* xB[i] violates its lower bound, which plays the role
                  of an upper bound on phase I */
               delta = rtol * (1.0 + kappa * fabs(lb[k]));
               t = ((lb[k] + delta) - bbar[i]) / alfa;
               i_stat = GLP_NL;
            }
            else if (phase == 1 && coef[k] > 0.0)
            {  /* xB[i] violates its upper bound, which plays the role
                  of an lower bound on phase I */
               continue;
            }
            else if (type[k] == GLP_UP || type[k] == GLP_DB ||
                     type[k] == GLP_FX)
            {  /* xB[i] is within its bounds and has an upper bound */
               delta = rtol * (1.0 + kappa * fabs(ub[k]));
               t = ((ub[k] + delta) - bbar[i]) / alfa;
               i_stat = GLP_NU;
            }
            else
            {  /* xB[i] is within its bounds and has no upper bound */
               continue;
            }
         }
         else
         {  /* xB[i] is decreasing */
            if (phase == 1 && coef[k] > 0.0)
            {  /* xB[i] violates its upper bound, which plays the role
                  of an lower bound on phase I */
               delta = rtol * (1.0 + kappa * fabs(ub[k]));
               t = ((ub[k] - delta) - bbar[i]) / alfa;
               i_stat = GLP_NU;
            }
            else if (phase == 1 && coef[k] < 0.0)
            {  /* xB[i] violates its lower bound, which plays the role
                  of an upper bound on phase I */
               continue;
            }
            else if (type[k] == GLP_LO || type[k] == GLP_DB ||
                     type[k] == GLP_FX)
            {  /* xB[i] is within its bounds and has an lower bound */
               delta = rtol * (1.0 + kappa * fabs(lb[k]));
               t = ((lb[k] - delta) - bbar[i]) / alfa;
               i_stat = GLP_NL;
            }
            else
            {  /* xB[i] is within its bounds and has no lower bound */
               continue;
            }
         }
         /* t is a change of xN[q], on which xB[i] reaches its bound
            (possibly relaxed); since the basic solution is assumed to
            be primal feasible (or pseudo feasible on phase I), t has
            to be non-negative by definition; however, it may happen
            that xB[i] slightly (i.e. within a tolerance) violates its
            bound, that leads to negative t; in the latter case, if
            xB[i] is chosen, negative t means that xN[q] changes in
            wrong direction; if pivot alfa[i,q] is close to zero, even
            small bound violation of xB[i] may lead to a large change
            of xN[q] in wrong direction; let, for example, xB[i] >= 0
            and in the current basis its value be -5e-9; let also xN[q]
            be on its zero bound and should increase; from the ratio
            test rule it follows that the pivot alfa[i,q] < 0; however,
            if alfa[i,q] is, say, -1e-9, the change of xN[q] in wrong
            direction is 5e-9 / (-1e-9) = -5, and using it for updating
            values of other basic variables will give absolutely wrong
            results; therefore, if t is negative, we should replace it
            by exact zero assuming that xB[i] is exactly on its bound,
            and the violation appears due to round-off errors */
         if (t < 0.0) t = 0.0;
         /* apply minimal ratio test */
         if (teta > t || (teta == t && big < fabs(alfa)))
            p = i, p_stat = i_stat, teta = t, big = fabs(alfa);
      }
      /* the second pass is skipped in the following cases: */
      /* if the standard ratio test is used */
      if (rtol == 0.0) goto done;
      /* if xN[q] reaches its opposite bound or if no basic variable
         has been chosen on the first pass */
      if (p <= 0) goto done;
      /* if xB[p] is a blocking variable, i.e. if it prevents xN[q]
         from any change */
      if (teta == 0.0) goto done;
      /*** SECOND PASS ***/
      /* here tmax is a maximal change of xN[q], on which the solution
         remains primal feasible (or pseudo feasible on phase I) within
         a tolerance */
      tmax = teta;
      /* nothing is chosen so far */
      p = 0, p_stat = 0, teta = DBL_MAX, big = 0.0;
      /* walk through significant elements of the pivot column */
      for (pos = 1; pos <= tcol_num; pos++)
      {  i = tcol_ind[pos];
         k = head[i]; /* x[k] = xB[i] */
         alfa = s * tcol_vec[i];
         /* xB[i] = ... + alfa * xN[q] + ..., and due to s we need to
            consider the only case when xN[q] is increasing */
         if (alfa > 0.0)
         {  /* xB[i] is increasing */
            if (phase == 1 && coef[k] < 0.0)
            {  /* xB[i] violates its lower bound, which plays the role
                  of an upper bound on phase I */
               t = (lb[k] - bbar[i]) / alfa;
               i_stat = GLP_NL;
            }
            else if (phase == 1 && coef[k] > 0.0)
            {  /* xB[i] violates its upper bound, which plays the role
                  of an lower bound on phase I */
               continue;
            }
            else if (type[k] == GLP_UP || type[k] == GLP_DB ||
                     type[k] == GLP_FX)
            {  /* xB[i] is within its bounds and has an upper bound */
               t = (ub[k] - bbar[i]) / alfa;
               i_stat = GLP_NU;
            }
            else
            {  /* xB[i] is within its bounds and has no upper bound */
               continue;
            }
         }
         else
         {  /* xB[i] is decreasing */
            if (phase == 1 && coef[k] > 0.0)
            {  /* xB[i] violates its upper bound, which plays the role
                  of an lower bound on phase I */
               t = (ub[k] - bbar[i]) / alfa;
               i_stat = GLP_NU;
            }
            else if (phase == 1 && coef[k] < 0.0)
            {  /* xB[i] violates its lower bound, which plays the role
                  of an upper bound on phase I */
               continue;
            }
            else if (type[k] == GLP_LO || type[k] == GLP_DB ||
                     type[k] == GLP_FX)
            {  /* xB[i] is within its bounds and has an lower bound */
               t = (lb[k] - bbar[i]) / alfa;
               i_stat = GLP_NL;
            }
            else
            {  /* xB[i] is within its bounds and has no lower bound */
               continue;
            }
         }
         /* (see comments for the first pass) */
         if (t < 0.0) t = 0.0;
         /* t is a change of xN[q], on which xB[i] reaches its bound;
            if t <= tmax, all basic variables can violate their bounds
            only within relaxation tolerance delta; we can use this
            freedom and choose basic variable having largest influence
            coefficient to avoid possible numeric instability */
         if (t <= tmax && big < fabs(alfa))
            p = i, p_stat = i_stat, teta = t, big = fabs(alfa);
      }
      /* something must be chosen on the second pass */
      xassert(p != 0);
done: /* store the index and status of basic variable xB[p] chosen */
      thecsa->p = p;
      if (p > 0 && type[head[p]] == GLP_FX)
         thecsa->p_stat = GLP_NS;
      else
         thecsa->p_stat = p_stat;
      /* store corresponding change of non-basic variable xN[q] */
      thecsa->teta = s * teta;
      return;
}

void eval_rho(csa *thecsa, double rho[])
{     int m = thecsa->m;
      int p = thecsa->p;
      double *e = rho;
      int i;
      /* construct the right-hand side vector e[p] */
      for (i = 1; i <= m; i++)
         e[i] = 0.0;
      e[p] = 1.0;
      /* solve system B'* rho = e[p] */
      xassert(thecsa->valid);
      bfd_btran(thecsa->bfd, rho);
      return;
}

void refine_rho(csa *thecsa, double rho[])
{     int m = thecsa->m;
      int p = thecsa->p;
      double *e = thecsa->work3;
      int i;
      /* construct the right-hand side vector e[p] */
      for (i = 1; i <= m; i++)
         e[i] = 0.0;
      e[p] = 1.0;
      /* refine solution of B'* rho = e[p] */
      refine_btran(thecsa, e, rho);
      return;
}

void eval_trow(csa *thecsa, double rho[])
{     int m = thecsa->m;
      int n = thecsa->n;
      int *N_ptr = thecsa->N_ptr;
      int *N_len = thecsa->N_len;
      int *N_ind = thecsa->N_ind;
      double *N_val = thecsa->N_val;
      int *trow_ind = thecsa->trow_ind;
      double *trow_vec = thecsa->trow_vec;
      int i, j, beg, end, ptr, nnz;
      double temp;
      /* clear the pivot row */
      for (j = 1; j <= n; j++)
         trow_vec[j] = 0.0;
      /* compute the pivot row as a linear combination of rows of the
         matrix N: trow = - rho[1] * N'[1] - ... - rho[m] * N'[m] */
      for (i = 1; i <= m; i++)
      {  temp = rho[i];
         if (temp == 0.0) continue;
         /* trow := trow - rho[i] * N'[i] */
         beg = N_ptr[i];
         end = beg + N_len[i];
         for (ptr = beg; ptr < end; ptr++)
         {
            trow_vec[N_ind[ptr]] -= temp * N_val[ptr];
         }
      }
      /* construct sparse pattern of the pivot row */
      nnz = 0;
      for (j = 1; j <= n; j++)
      {  if (trow_vec[j] != 0.0)
            trow_ind[++nnz] = j;
      }
      thecsa->trow_nnz = nnz;
      return;
}

void update_bbar(csa *thecsa)
{
      double *bbar = thecsa->bbar;
      int q = thecsa->q;
      int tcol_nnz = thecsa->tcol_nnz;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      int p = thecsa->p;
      double teta = thecsa->teta;
      int i, pos;
      /* if xN[q] leaves the basis, compute its value in the adjacent
         basis, where it will replace xB[p] */
      if (p > 0)
         bbar[p] = get_xN(thecsa, q) + teta;
      /* update values of other basic variables (except xB[p], because
         it will be replaced by xN[q]) */
      if (teta == 0.0) goto done;
      for (pos = 1; pos <= tcol_nnz; pos++)
      {  i = tcol_ind[pos];
         /* skip xB[p] */
         if (i == p) continue;
         /* (change of xB[i]) = alfa[i,q] * (change of xN[q]) */
         bbar[i] += tcol_vec[i] * teta;
      }
done: return;
}

double reeval_cost(csa *thecsa)
{     int m = thecsa->m;
      double *coef = thecsa->coef;
      int *head = thecsa->head;
      int q = thecsa->q;
      int tcol_nnz = thecsa->tcol_nnz;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      int i, pos;
      double dq;
      dq = coef[head[m+q]];
      for (pos = 1; pos <= tcol_nnz; pos++)
      {  i = tcol_ind[pos];
         dq += coef[head[i]] * tcol_vec[i];
      }
      return dq;
}

void update_cbar(csa *thecsa)
{
      double *cbar = thecsa->cbar;
      int q = thecsa->q;
      int trow_nnz = thecsa->trow_nnz;
      int *trow_ind = thecsa->trow_ind;
      double *trow_vec = thecsa->trow_vec;
      int j, pos;
      double new_dq;
      /* compute reduced cost of xB[p] in the adjacent basis, where it
         will replace xN[q] */
      new_dq = (cbar[q] /= trow_vec[q]);
      /* update reduced costs of other non-basic variables (except
         xN[q], because it will be replaced by xB[p]) */
      for (pos = 1; pos <= trow_nnz; pos++)
      {  j = trow_ind[pos];
         /* skip xN[q] */
         if (j == q) continue;
         cbar[j] -= trow_vec[j] * new_dq;
      }
      return;
}

void update_gamma(csa *thecsa)
{     int m = thecsa->m;
      char *type = thecsa->type;
      int *A_ptr = thecsa->A_ptr;
      int *A_ind = thecsa->A_ind;
      double *A_val = thecsa->A_val;
      int *head = thecsa->head;
      char *refsp = thecsa->refsp;
      double *gamma = thecsa->gamma;
      int q = thecsa->q;
      int tcol_nnz = thecsa->tcol_nnz;
      int *tcol_ind = thecsa->tcol_ind;
      double *tcol_vec = thecsa->tcol_vec;
      int p = thecsa->p;
      int trow_nnz = thecsa->trow_nnz;
      int *trow_ind = thecsa->trow_ind;
      double *trow_vec = thecsa->trow_vec;
      double *u = thecsa->work3;
      int i, j, k, pos, beg, end, ptr;
      double gamma_q, delta_q, pivot, s, t, t1, t2;

      /* the basis changes, so decrease the count */
      xassert(thecsa->refct > 0);
      thecsa->refct--;
      /* recompute gamma[q] for the current basis more accurately and
         compute auxiliary vector u */
      gamma_q = delta_q = (refsp[head[m+q]] ? 1.0 : 0.0);
      for (i = 1; i <= m; i++) u[i] = 0.0;
      for (pos = 1; pos <= tcol_nnz; pos++)
      {  i = tcol_ind[pos];
         if (refsp[head[i]])
         {  u[i] = t = tcol_vec[i];
            gamma_q += t * t;
         }
         else
            u[i] = 0.0;
      }
      xassert(thecsa->valid);
      bfd_btran(thecsa->bfd, u);
      /* update gamma[k] for other non-basic variables (except fixed
         variables and xN[q], because it will be replaced by xB[p]) */
      pivot = trow_vec[q];
      for (pos = 1; pos <= trow_nnz; pos++)
      {  j = trow_ind[pos];
         /* skip xN[q] */
         if (j == q) continue;
         /* compute t */
         t = trow_vec[j] / pivot;
         /* compute inner product s = N'[j] * u */
         k = head[m+j]; /* x[k] = xN[j] */
         if (k <= m)
            s = u[k];
         else
         {  s = 0.0;
            beg = A_ptr[k-m];
            end = A_ptr[k-m+1];
            for (ptr = beg; ptr < end; ptr++)
               s -= A_val[ptr] * u[A_ind[ptr]];
         }
         /* compute gamma[k] for the adjacent basis */
         t1 = gamma[j] + t * t * gamma_q + 2.0 * t * s;
         t2 = (refsp[k] ? 1.0 : 0.0) + delta_q * t * t;
         gamma[j] = (t1 >= t2 ? t1 : t2);
         if (gamma[j] < DBL_EPSILON) gamma[j] = DBL_EPSILON;
      }
      /* compute gamma[q] for the adjacent basis */
      if (type[head[p]] == GLP_FX)
         gamma[q] = 1.0;
      else
      {  gamma[q] = gamma_q / (pivot * pivot);
         if (gamma[q] < DBL_EPSILON) gamma[q] = DBL_EPSILON;
      }
      return;
}

double err_in_bbar(csa *thecsa)
{     int m = thecsa->m;
      double *bbar = thecsa->bbar;
      int i;
      double e, emax, *beta;
      beta = (double*)xcalloc(1+m, sizeof(double));
      eval_beta(thecsa, beta);
      emax = 0.0;
      for (i = 1; i <= m; i++)
      {  e = fabs(beta[i] - bbar[i]) / (1.0 + fabs(beta[i]));
         if (emax < e) emax = e;
      }
      xfree(beta);
      return emax;
}

double err_in_cbar(csa *thecsa)
{     int m = thecsa->m;
      int n = thecsa->n;
      char *stat = thecsa->stat;
      double *cbar = thecsa->cbar;
      int j;
      double e, emax, cost, *pi;
      pi = (double*)xcalloc(1+m, sizeof(double));
      eval_pi(thecsa, pi);
      emax = 0.0;
      for (j = 1; j <= n; j++)
      {  if (stat[j] == GLP_NS) continue;
         cost = eval_cost(thecsa, pi, j);
         e = fabs(cost - cbar[j]) / (1.0 + fabs(cost));
         if (emax < e) emax = e;
      }
      xfree(pi);
      return emax;
}

double err_in_gamma(csa *thecsa)
{     int n = thecsa->n;
      char *stat = thecsa->stat;
      double *gamma = thecsa->gamma;
      int j;
      double e, emax, temp;
      emax = 0.0;
      for (j = 1; j <= n; j++)
      {  if (stat[j] == GLP_NS)
         {  xassert(gamma[j] == 1.0);
            continue;
         }
         temp = eval_gamma(thecsa, j);
         e = fabs(temp - gamma[j]) / (1.0 + fabs(temp));
         if (emax < e) emax = e;
      }
      return emax;
}

void change_basis(csa *thecsa)
{     int m = thecsa->m;
      int *head = thecsa->head;
      char *stat = thecsa->stat;
      int q = thecsa->q;
      int p = thecsa->p;
      int p_stat = thecsa->p_stat;
      int k;
      if (p < 0)
      {  /* xN[q] goes to its opposite bound */
         switch (stat[q])
         {  case GLP_NL:
               /* xN[q] increases */
               stat[q] = GLP_NU;
               break;
            case GLP_NU:
               /* xN[q] decreases */
               stat[q] = GLP_NL;
               break;
            default:
               xassert(stat != stat);
         }
      }
      else
      {  /* xB[p] leaves the basis, xN[q] enters the basis */
         /* xB[p] <-> xN[q] */
         k = head[p], head[p] = head[m+q], head[m+q] = k;
         stat[q] = (char)p_stat;
      }
      return;
}

int set_aux_obj(csa *thecsa, double tol_bnd)
{     int m = thecsa->m;
      int n = thecsa->n;
      char *type = thecsa->type;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      double *coef = thecsa->coef;
      int *head = thecsa->head;
      double *bbar = thecsa->bbar;
      int i, k, cnt = 0;
      double eps;
      /* use a bit more restrictive tolerance */
      tol_bnd *= 0.90;
      /* clear all objective coefficients */
      for (k = 1; k <= m+n; k++)
         coef[k] = 0.0;
      /* walk through the list of basic variables */
      for (i = 1; i <= m; i++)
      {  k = head[i]; /* x[k] = xB[i] */
         if (type[k] == GLP_LO || type[k] == GLP_DB ||
             type[k] == GLP_FX)
         {  /* x[k] has lower bound */
            eps = tol_bnd * (1.0 + kappa * fabs(lb[k]));
            if (bbar[i] < lb[k] - eps)
            {  /* and violates it */
               coef[k] = -1.0;
               cnt++;
            }
         }
         if (type[k] == GLP_UP || type[k] == GLP_DB ||
             type[k] == GLP_FX)
         {  /* x[k] has upper bound */
            eps = tol_bnd * (1.0 + kappa * fabs(ub[k]));
            if (bbar[i] > ub[k] + eps)
            {  /* and violates it */
               coef[k] = +1.0;
               cnt++;
            }
         }
      }
      return cnt;
}

void set_orig_obj(csa *thecsa)
{     int m = thecsa->m;
      int n = thecsa->n;
      double *coef = thecsa->coef;
      double *obj = thecsa->obj;
      double zeta = thecsa->zeta;
      int i, j;
      for (i = 1; i <= m; i++)
         coef[i] = 0.0;
      for (j = 1; j <= n; j++)
         coef[m+j] = zeta * obj[j];
      return;
}

int check_stab(csa *thecsa, double tol_bnd)
{     int m = thecsa->m;
      char *type = thecsa->type;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      double *coef = thecsa->coef;
      int *head = thecsa->head;
      int phase = thecsa->phase;
      double *bbar = thecsa->bbar;
      int i, k;
      double eps;
      /* walk through the list of basic variables */
      for (i = 1; i <= m; i++)
      {  k = head[i]; /* x[k] = xB[i] */
         if (phase == 1 && coef[k] < 0.0)
         {  /* x[k] must not be greater than its lower bound */
            eps = tol_bnd * (1.0 + kappa * fabs(lb[k]));
            if (bbar[i] > lb[k] + eps) return 1;
         }
         else if (phase == 1 && coef[k] > 0.0)
         {  /* x[k] must not be less than its upper bound */
            eps = tol_bnd * (1.0 + kappa * fabs(ub[k]));
            if (bbar[i] < ub[k] - eps) return 1;
         }
         else
         {  /* either phase = 1 and coef[k] = 0, or phase = 2 */
            if (type[k] == GLP_LO || type[k] == GLP_DB ||
                type[k] == GLP_FX)
            {  /* x[k] must not be less than its lower bound */
               eps = tol_bnd * (1.0 + kappa * fabs(lb[k]));
               if (bbar[i] < lb[k] - eps) return 1;
            }
            if (type[k] == GLP_UP || type[k] == GLP_DB ||
                type[k] == GLP_FX)
            {  /* x[k] must not be greater then its upper bound */
               eps = tol_bnd * (1.0 + kappa * fabs(ub[k]));
               if (bbar[i] > ub[k] + eps) return 1;
            }
         }
      }
      /* basic solution is primal feasible within a tolerance */
      return 0;
}

int check_feas(csa *thecsa, double tol_bnd)
{     int m = thecsa->m;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      double *coef = thecsa->coef;
      int *head = thecsa->head;
      double *bbar = thecsa->bbar;
      int i, k;
      double eps;
      xassert(thecsa->phase == 1);
      /* walk through the list of basic variables */
      for (i = 1; i <= m; i++)
      {  k = head[i]; /* x[k] = xB[i] */
         if (coef[k] < 0.0)
         {  /* check if x[k] still violates its lower bound */
            eps = tol_bnd * (1.0 + kappa * fabs(lb[k]));
            if (bbar[i] < lb[k] - eps) return 1;
         }
         else if (coef[k] > 0.0)
         {  /* check if x[k] still violates its upper bound */
            eps = tol_bnd * (1.0 + kappa * fabs(ub[k]));
            if (bbar[i] > ub[k] + eps) return 1;
         }
      }
      /* basic solution is primal feasible within a tolerance */
      return 0;
}

double eval_obj(csa *thecsa)
{     int m = thecsa->m;
      int n = thecsa->n;
      double *obj = thecsa->obj;
      int *head = thecsa->head;
      double *bbar = thecsa->bbar;
      int i, j, k;
      double sum;
      sum = obj[0];
      /* walk through the list of basic variables */
      for (i = 1; i <= m; i++)
      {  k = head[i]; /* x[k] = xB[i] */
         if (k > m)
            sum += obj[k-m] * bbar[i];
      }
      /* walk through the list of non-basic variables */
      for (j = 1; j <= n; j++)
      {  k = head[m+j]; /* x[k] = xN[j] */
         if (k > m)
            sum += obj[k-m] * get_xN(thecsa, j);
      }
      return sum;
}

void display(csa *thecsa, const glp_smcp *parm, int spec)
{     int m = thecsa->m;
      char *type = thecsa->type;
      double *lb = thecsa->lb;
      double *ub = thecsa->ub;
      //int phase = thecsa->phase; //mtcomment: compiler says this is unused
      int *head = thecsa->head;
      double *bbar = thecsa->bbar;
      int i, k, cnt;
      double sum;
      if (parm->msg_lev < GLP_MSG_ON) goto skip;
      if (parm->out_dly > 0 &&
         1000.0 * xdifftime(xtime(), thecsa->tm_beg) < parm->out_dly)
         goto skip;
      if (thecsa->it_cnt == thecsa->it_dpy) goto skip;
      if (!spec && thecsa->it_cnt % parm->out_frq != 0) goto skip;
      /* compute the sum of primal infeasibilities and determine the
         number of basic fixed variables */
      sum = 0.0, cnt = 0;
      for (i = 1; i <= m; i++)
      {  k = head[i]; /* x[k] = xB[i] */
         if (type[k] == GLP_LO || type[k] == GLP_DB ||
             type[k] == GLP_FX)
         {  /* x[k] has lower bound */
            if (bbar[i] < lb[k])
               sum += (lb[k] - bbar[i]);
         }
         if (type[k] == GLP_UP || type[k] == GLP_DB ||
             type[k] == GLP_FX)
         {  /* x[k] has upper bound */
            if (bbar[i] > ub[k])
               sum += (bbar[i] - ub[k]);
         }
         if (type[k] == GLP_FX) cnt++;
      }
      thecsa->it_dpy = thecsa->it_cnt;
skip: return;
}

void store_sol(csa *thecsa, glp_prob *lp, int p_stat,
      int d_stat, int ray)
{     int m = thecsa->m;
      int n = thecsa->n;
      double zeta = thecsa->zeta;
      int *head = thecsa->head;
      char *stat = thecsa->stat;
      double *bbar = thecsa->bbar;
      double *cbar = thecsa->cbar;
      int i, j, k;
      /* basis factorization */
      lp->valid = 1, thecsa->valid = 0;
      lp->bfd = thecsa->bfd, thecsa->bfd = NULL;
      memcpy(&lp->head[1], &head[1], m * sizeof(int));
      /* basic solution status */
      lp->pbs_stat = p_stat;
      lp->dbs_stat = d_stat;
      /* objective function value */
      lp->obj_val = eval_obj(thecsa);
      /* simplex iteration count */
      lp->it_cnt = thecsa->it_cnt;
      /* unbounded ray */
      lp->some = ray;
      /* basic variables */
      for (i = 1; i <= m; i++)
      {  k = head[i]; /* x[k] = xB[i] */
         if (k <= m)
         {  GLPROW *row = lp->row[k];
            row->stat = GLP_BS;
            row->bind = i;
            row->prim = bbar[i] / row->rii;
            row->dual = 0.0;
         }
         else
         {  GLPCOL *col = lp->col[k-m];
            col->stat = GLP_BS;
            col->bind = i;
            col->prim = bbar[i] * col->sjj;
            col->dual = 0.0;
         }
      }
      /* non-basic variables */
      for (j = 1; j <= n; j++)
      {  k = head[m+j]; /* x[k] = xN[j] */
         if (k <= m)
         {  GLPROW *row = lp->row[k];
            row->stat = stat[j];
            row->bind = 0;
            switch (stat[j])
            {  case GLP_NL:
                  row->prim = row->lb; break;
               case GLP_NU:
                  row->prim = row->ub; break;
               case GLP_NF:
                  row->prim = 0.0; break;
               case GLP_NS:
                  row->prim = row->lb; break;
               default:
                  xassert(stat != stat);
            }
            row->dual = (cbar[j] * row->rii) / zeta;
         }
         else
         {  GLPCOL *col = lp->col[k-m];
            col->stat = stat[j];
            col->bind = 0;
            switch (stat[j])
            {  case GLP_NL:
                  col->prim = col->lb; break;
               case GLP_NU:
                  col->prim = col->ub; break;
               case GLP_NF:
                  col->prim = 0.0; break;
               case GLP_NS:
                  col->prim = col->lb; break;
               default:
                  xassert(stat != stat);
            }
            col->dual = (cbar[j] / col->sjj) / zeta;
         }
      }
      return;
}

void free_thecsa(csa *thecsa)
{     xfree(thecsa->type);
      xfree(thecsa->lb);
      xfree(thecsa->ub);
      xfree(thecsa->coef);
      xfree(thecsa->obj);
      xfree(thecsa->A_ptr);
      xfree(thecsa->A_ind);
      xfree(thecsa->A_val);
      xfree(thecsa->head);
      xfree(thecsa->stat);
      xfree(thecsa->N_ptr);
      xfree(thecsa->N_len);
      xfree(thecsa->N_ind);
      xfree(thecsa->N_val);
      xfree(thecsa->bbar);
      xfree(thecsa->cbar);
      xfree(thecsa->refsp);
      xfree(thecsa->gamma);
      xfree(thecsa->tcol_ind);
      xfree(thecsa->tcol_vec);
      xfree(thecsa->trow_ind);
      xfree(thecsa->trow_vec);
      xfree(thecsa->work1);
      xfree(thecsa->work2);
      xfree(thecsa->work3);
      xfree(thecsa->work4);
      xfree(thecsa);
      return;
}

int spx_primal(glp_prob *lp, const glp_smcp *parm)
{     csa *thecsa;
      int binv_st = 2;
      /* status of basis matrix factorization:
         0 - invalid; 1 - just computed; 2 - updated */
      int bbar_st = 0;
      /* status of primal values of basic variables:
         0 - invalid; 1 - just computed; 2 - updated */
      int cbar_st = 0;
      /* status of reduced costs of non-basic variables:
         0 - invalid; 1 - just computed; 2 - updated */
      int rigorous = 0;
      /* rigorous mode flag; this flag is used to enable iterative
         refinement on computing pivot rows and columns of the simplex
         table */
      int check = 0;
      int p_stat = 0, d_stat = 0, ret = 0;
      /* allocate and initialize the common storage area */
      thecsa = alloc_csa(lp);
      init_csa(thecsa, lp);
loop: /* main loop starts here */
      /* compute factorization of the basis matrix */
      if (binv_st == 0)
      {  ret = invert_B(thecsa);
         if (ret != 0)
         {  xassert(!lp->valid && lp->bfd == NULL);
            lp->bfd = thecsa->bfd, thecsa->bfd = NULL;
            lp->pbs_stat = lp->dbs_stat = GLP_UNDEF;
            lp->obj_val = 0.0;
            lp->it_cnt = thecsa->it_cnt;
            lp->some = 0;
            ret = GLP_EFAIL;
            goto done;
         }
         thecsa->valid = 1;
         binv_st = 1; /* just computed */
         /* invalidate basic solution components */
         bbar_st = cbar_st = 0;
      }
      /* compute primal values of basic variables */
      if (bbar_st == 0)
      {  eval_bbar(thecsa);
         bbar_st = 1; /* just computed */
         /* determine the search phase, if not determined yet */
         if (thecsa->phase == 0)
         {  if (set_aux_obj(thecsa, parm->tol_bnd) > 0)
            {  /* current basic solution is primal infeasible */
               /* start to minimize the sum of infeasibilities */
               thecsa->phase = 1;
            }
            else
            {  /* current basic solution is primal feasible */
               /* start to minimize the original objective function */
               set_orig_obj(thecsa);
               thecsa->phase = 2;
            }
            xassert(check_stab(thecsa, parm->tol_bnd) == 0);
            /* working objective coefficients have been changed, so
               invalidate reduced costs */
            cbar_st = 0;
            display(thecsa, parm, 1);
         }
         /* make sure that the current basic solution remains primal
            feasible (or pseudo feasible on phase I) */
         if (check_stab(thecsa, parm->tol_bnd))
         {  /* there are excessive bound violations due to round-off
               errors */
            /* restart the search */
            thecsa->phase = 0;
            binv_st = 0;
            rigorous = 5;
            goto loop;
         }
      }
      xassert(thecsa->phase == 1 || thecsa->phase == 2);
      /* on phase I we do not need to wait until the current basic
         solution becomes dual feasible; it is sufficient to make sure
         that no basic variable violates its bounds */
      if (thecsa->phase == 1 && !check_feas(thecsa, parm->tol_bnd))
      {  /* the current basis is primal feasible; switch to phase II */
         thecsa->phase = 2;
         set_orig_obj(thecsa);
         cbar_st = 0;
         display(thecsa, parm, 1);
      }
      /* compute reduced costs of non-basic variables */
      if (cbar_st == 0)
      {  eval_cbar(thecsa);
         cbar_st = 1; /* just computed */
      }
      /* redefine the reference space, if required */
      switch (parm->pricing)
      {  case GLP_PT_STD:
            break;
         case GLP_PT_PSE:
            if (thecsa->refct == 0) reset_refsp(thecsa);
            break;
         default:
            xassert(parm != parm);
      }
      /* at this point the basis factorization and all basic solution
         components are valid */
      xassert(binv_st && bbar_st && cbar_st);
      /* check accuracy of current basic solution components (only for
         debugging) */
      if (check)
      {  double e_bbar = err_in_bbar(thecsa);
         double e_cbar = err_in_cbar(thecsa);
         double e_gamma =
            (parm->pricing == GLP_PT_PSE ? err_in_gamma(thecsa) : 0.0);
         xassert(e_bbar <= 1e-5 && e_cbar <= 1e-5 && e_gamma <= 1e-3);
      }
      /* check if the iteration limit has been exhausted */
      if (parm->it_lim < INT_MAX &&
          thecsa->it_cnt - thecsa->it_beg >= parm->it_lim)
      {  if (bbar_st != 1 || (thecsa->phase == 2 && cbar_st != 1))
         {  if (bbar_st != 1) bbar_st = 0;
            if (thecsa->phase == 2 && cbar_st != 1) cbar_st = 0;
            goto loop;
         }
         display(thecsa, parm, 1);
         switch (thecsa->phase)
         {  case 1:
               p_stat = GLP_INFEAS;
               set_orig_obj(thecsa);
               eval_cbar(thecsa);
               break;
            case 2:
               p_stat = GLP_FEAS;
               break;
            default:
               xassert(thecsa != thecsa);
         }
         chuzc(thecsa, parm->tol_dj);
         d_stat = (thecsa->q == 0 ? GLP_FEAS : GLP_INFEAS);
         store_sol(thecsa, lp, p_stat, d_stat, 0);
         ret = GLP_EITLIM;
         goto done;
      }
      /* check if the time limit has been exhausted */
      if (parm->tm_lim < INT_MAX &&
          1000.0 * xdifftime(xtime(), thecsa->tm_beg) >= parm->tm_lim)
      {  if (bbar_st != 1 || (thecsa->phase == 2 && cbar_st != 1))
         {  if (bbar_st != 1) bbar_st = 0;
            if (thecsa->phase == 2 && cbar_st != 1) cbar_st = 0;
            goto loop;
         }
         display(thecsa, parm, 1);
         switch (thecsa->phase)
         {  case 1:
               p_stat = GLP_INFEAS;
               set_orig_obj(thecsa);
               eval_cbar(thecsa);
               break;
            case 2:
               p_stat = GLP_FEAS;
               break;
            default:
               xassert(thecsa != thecsa);
         }
         chuzc(thecsa, parm->tol_dj);
         d_stat = (thecsa->q == 0 ? GLP_FEAS : GLP_INFEAS);
         store_sol(thecsa, lp, p_stat, d_stat, 0);
         ret = GLP_ETMLIM;
         goto done;
      }
      /* display the search progress */
      display(thecsa, parm, 0);
      /* choose non-basic variable xN[q] */
      chuzc(thecsa, parm->tol_dj);
      if (thecsa->q == 0)
      {  if (bbar_st != 1 || cbar_st != 1)
         {  if (bbar_st != 1) bbar_st = 0;
            if (cbar_st != 1) cbar_st = 0;
            goto loop;
         }
         display(thecsa, parm, 1);
         switch (thecsa->phase)
         {  case 1:
               p_stat = GLP_NOFEAS;
               set_orig_obj(thecsa);
               eval_cbar(thecsa);
               chuzc(thecsa, parm->tol_dj);
               d_stat = (thecsa->q == 0 ? GLP_FEAS : GLP_INFEAS);
               break;
            case 2:
               p_stat = d_stat = GLP_FEAS;
               break;
            default:
               xassert(thecsa != thecsa);
         }
         store_sol(thecsa, lp, p_stat, d_stat, 0);
         ret = 0;
         goto done;
      }
      /* compute pivot column of the simplex table */
      eval_tcol(thecsa);
      if (rigorous) refine_tcol(thecsa);
      sort_tcol(thecsa, parm->tol_piv);
      /* check accuracy of the reduced cost of xN[q] */
      {  double d1 = thecsa->cbar[thecsa->q]; /* less accurate */
         double d2 = reeval_cost(thecsa);  /* more accurate */
         xassert(d1 != 0.0);
         if (fabs(d1 - d2) > 1e-5 * (1.0 + fabs(d2)) ||
             !((d1 < 0.0 && d2 < 0.0) || (d1 > 0.0 && d2 > 0.0)))
         {  if (cbar_st != 1 || !rigorous)
            {  if (cbar_st != 1) cbar_st = 0;
               rigorous = 5;
               goto loop;
            }
         }
         /* replace cbar[q] by more accurate value keeping its sign */
         if (d1 > 0.0)
            thecsa->cbar[thecsa->q] = (d2 > 0.0 ? d2 : +DBL_EPSILON);
         else
            thecsa->cbar[thecsa->q] = (d2 < 0.0 ? d2 : -DBL_EPSILON);
      }
      /* choose basic variable xB[p] */
      switch (parm->r_test)
      {  case GLP_RT_STD:
            chuzr(thecsa, 0.0);
            break;
         case GLP_RT_HAR:
            chuzr(thecsa, 0.30 * parm->tol_bnd);
            break;
         default:
            xassert(parm != parm);
      }
      if (thecsa->p == 0)
      {  if (bbar_st != 1 || cbar_st != 1 || !rigorous)
         {  if (bbar_st != 1) bbar_st = 0;
            if (cbar_st != 1) cbar_st = 0;
            rigorous = 1;
            goto loop;
         }
         display(thecsa, parm, 1);
         switch (thecsa->phase)
         {  case 1:
               xassert(!lp->valid && lp->bfd == NULL);
               lp->bfd = thecsa->bfd, thecsa->bfd = NULL;
               lp->pbs_stat = lp->dbs_stat = GLP_UNDEF;
               lp->obj_val = 0.0;
               lp->it_cnt = thecsa->it_cnt;
               lp->some = 0;
               ret = GLP_EFAIL;
               break;
            case 2:
               store_sol(thecsa, lp, GLP_FEAS, GLP_NOFEAS,
                  thecsa->head[thecsa->m+thecsa->q]);
               ret = 0;
               break;
            default:
               xassert(thecsa != thecsa);
         }
         goto done;
      }
      /* check if the pivot element is acceptable */
      if (thecsa->p > 0)
      {  double piv = thecsa->tcol_vec[thecsa->p];
         double eps = 1e-5 * (1.0 + 0.01 * thecsa->tcol_max);
         if (fabs(piv) < eps)
         {  if (!rigorous)
            {  rigorous = 5;
               goto loop;
            }
         }
      }
      /* now xN[q] and xB[p] have been chosen anyhow */
      /* compute pivot row of the simplex table */
      if (thecsa->p > 0)
      {  double *rho = thecsa->work4;
         eval_rho(thecsa, rho);
         if (rigorous) refine_rho(thecsa, rho);
         eval_trow(thecsa, rho);
      }
      /* accuracy check based on the pivot element */
      if (thecsa->p > 0)
      {  double piv1 = thecsa->tcol_vec[thecsa->p]; /* more accurate */
         double piv2 = thecsa->trow_vec[thecsa->q]; /* less accurate */
         xassert(piv1 != 0.0);
         if (fabs(piv1 - piv2) > 1e-8 * (1.0 + fabs(piv1)) ||
             !((piv1 > 0.0 && piv2 > 0.0) || (piv1 < 0.0 && piv2 < 0.0)))
         {  if (binv_st != 1 || !rigorous)
            {  if (binv_st != 1) binv_st = 0;
               rigorous = 5;
               goto loop;
            }
            /* use more accurate version in the pivot row */
            if (thecsa->trow_vec[thecsa->q] == 0.0)
            {  thecsa->trow_nnz++;
               xassert(thecsa->trow_nnz <= thecsa->n);
               thecsa->trow_ind[thecsa->trow_nnz] = thecsa->q;
            }
            thecsa->trow_vec[thecsa->q] = piv1;
         }
      }
      /* update primal values of basic variables */
      update_bbar(thecsa);
      bbar_st = 2; /* updated */
      /* update reduced costs of non-basic variables */
      if (thecsa->p > 0)
      {  update_cbar(thecsa);
         cbar_st = 2; /* updated */
         /* on phase I objective coefficient of xB[p] in the adjacent
            basis becomes zero */
         if (thecsa->phase == 1)
         {  int k = thecsa->head[thecsa->p]; /* x[k] = xB[p] -> xN[q] */
            thecsa->cbar[thecsa->q] -= thecsa->coef[k];
            thecsa->coef[k] = 0.0;
         }
      }
      /* update steepest edge coefficients */
      if (thecsa->p > 0)
      {  switch (parm->pricing)
         {  case GLP_PT_STD:
               break;
            case GLP_PT_PSE:
               if (thecsa->refct > 0) update_gamma(thecsa);
               break;
            default:
               xassert(parm != parm);
         }
      }
      /* update factorization of the basis matrix */
      if (thecsa->p > 0)
      {  ret = update_B(thecsa, thecsa->p, thecsa->head[thecsa->m+thecsa->q]);
         if (ret == 0)
            binv_st = 2; /* updated */
         else
         {  thecsa->valid = 0;
            binv_st = 0; /* invalid */
         }
      }
      /* update matrix N */
      if (thecsa->p > 0)
      {  del_N_col(thecsa, thecsa->q, thecsa->head[thecsa->m+thecsa->q]);
         if (thecsa->type[thecsa->head[thecsa->p]] != GLP_FX)
            add_N_col(thecsa, thecsa->q, thecsa->head[thecsa->p]);
      }
      /* change the basis header */
      change_basis(thecsa);
      /* iteration complete */
      thecsa->it_cnt++;
      if (rigorous > 0) rigorous--;
      goto loop;
done: /* deallocate the common storage area */
      free_thecsa(thecsa);
      /* return to the calling program */
      return ret;
}

////////// END SPX


////////// START API-1

#define NNZ_MAX 500000000 /* = 500*10^6 */
/* maximal number of constraint coefficients in the problem object */

void create_prob(glp_prob *lp)
{     lp->magic = GLP_PROB_MAGIC;
      lp->pool = dmp_create_pool();
      lp->parms = NULL;

      /* LP/MIP data */
      lp->name = NULL;
      lp->obj = NULL;
      lp->dir = GLP_MIN;
      lp->c0 = 0.0;
      lp->m_max = 100;
      lp->n_max = 200;
      lp->m = lp->n = 0;
      lp->nnz = 0;
      lp->row = (GLPROW**)xcalloc(1+lp->m_max, sizeof(GLPROW *));
      lp->col = (GLPCOL**)xcalloc(1+lp->n_max, sizeof(GLPCOL *));
      /* basis factorization */
      lp->valid = 0;
      lp->head = (int*)xcalloc(1+lp->m_max, sizeof(int));
      lp->bfcp = NULL;
      lp->bfd = NULL;
      /* basic solution (LP) */
      lp->pbs_stat = lp->dbs_stat = GLP_UNDEF;
      lp->obj_val = 0.0;
      lp->it_cnt = 0;
      lp->some = 0;
      /* interior-point solution (LP) */
      lp->ipt_stat = GLP_UNDEF;
      lp->ipt_obj = 0.0;
      /* integer solution (MIP) */
      lp->mip_stat = GLP_UNDEF;
      lp->mip_obj = 0.0;
      return;
}

glp_prob *glp_create_prob(void)
{     glp_prob *lp;
      lp = (glp_prob*)xmalloc(sizeof(glp_prob));
      create_prob(lp);
      return lp;
}

void glp_set_obj_dir(glp_prob *lp, int dir)
{
     if (!(dir == GLP_MIN || dir == GLP_MAX))
         xerror("glp_set_obj_dir: dir = %d; invalid direction flag",
            dir);
      lp->dir = dir;
      return;
}

int glp_add_rows(glp_prob *lp, int nrs)
{
      GLPROW *row;
      int m_new, i;
      /* determine new number of rows */
      if (nrs < 1)
         xerror("glp_add_rows: nrs = %d; invalid number of rows",
            nrs);
      if (nrs > M_MAX - lp->m)
         xerror("glp_add_rows: nrs = %d; too many rows", nrs);
      m_new = lp->m + nrs;
      /* increase the room, if necessary */
      if (lp->m_max < m_new)
      {  GLPROW **save = lp->row;
         while (lp->m_max < m_new)
         {  lp->m_max += lp->m_max;
            xassert(lp->m_max > 0);
         }
         lp->row = (GLPROW**)xcalloc(1+lp->m_max, sizeof(GLPROW *));
         memcpy(&lp->row[1], &save[1], lp->m * sizeof(GLPROW *));
         xfree(save);
         /* do not forget about the basis header */
         xfree(lp->head);
         lp->head = (int*)xcalloc(1+lp->m_max, sizeof(int));
      }
      /* add new rows to the end of the row list */
      for (i = lp->m+1; i <= m_new; i++)
      {  /* create row descriptor */
         lp->row[i] = row = (GLPROW*)dmp_get_atom(lp->pool, sizeof(GLPROW));
         row->i = i;
         row->name = NULL;
         row->level = 0;
         row->origin = 0;
         row->klass = 0;
         row->type = GLP_FR;
         row->lb = row->ub = 0.0;
         row->ptr = NULL;
         row->rii = 1.0;
         row->stat = GLP_BS;
         row->bind = 0;
         row->prim = row->dual = 0.0;
         row->pval = row->dval = 0.0;
         row->mipx = 0.0;
      }
      /* set new number of rows */
      lp->m = m_new;
      /* invalidate the basis factorization */
      lp->valid = 0;
      /* return the ordinal number of the first row added */
      return m_new - nrs + 1;
}

int glp_add_cols(glp_prob *lp, int ncs)
{
      GLPCOL *col;
      int n_new, j;
      /* determine new number of columns */
      if (ncs < 1)
         xerror("glp_add_cols: ncs = %d; invalid number of columns",
            ncs);
      if (ncs > N_MAX - lp->n)
         xerror("glp_add_cols: ncs = %d; too many columns", ncs);
      n_new = lp->n + ncs;
      /* increase the room, if necessary */
      if (lp->n_max < n_new)
      {  GLPCOL **save = lp->col;
         while (lp->n_max < n_new)
         {  lp->n_max += lp->n_max;
            xassert(lp->n_max > 0);
         }
         lp->col = (GLPCOL**)xcalloc(1+lp->n_max, sizeof(GLPCOL *));
         memcpy(&lp->col[1], &save[1], lp->n * sizeof(GLPCOL *));
         xfree(save);
      }
      /* add new columns to the end of the column list */
      for (j = lp->n+1; j <= n_new; j++)
      {  /* create column descriptor */
         lp->col[j] = col = (GLPCOL*)dmp_get_atom(lp->pool, sizeof(GLPCOL));
         col->j = j;
         col->name = NULL;
         col->kind = GLP_CV;
         col->type = GLP_FX;
         col->lb = col->ub = 0.0;
         col->coef = 0.0;
         col->ptr = NULL;
         col->sjj = 1.0;
         col->stat = GLP_NS;
         col->bind = 0; /* the basis may remain valid */
         col->prim = col->dual = 0.0;
         col->pval = col->dval = 0.0;
         col->mipx = 0.0;
      }
      /* set new number of columns */
      lp->n = n_new;
      /* return the ordinal number of the first column added */
      return n_new - ncs + 1;
}

void glp_set_row_bnds(glp_prob *lp, int i, int type, double lb,
      double ub)
{     GLPROW *row;
      if (!(1 <= i && i <= lp->m))
         xerror("glp_set_row_bnds: i = %d; row number out of range", i);
      row = lp->row[i];
      row->type = type;
      switch (type)
      {  case GLP_FR:
            row->lb = row->ub = 0.0;
            if (row->stat != GLP_BS) row->stat = GLP_NF;
            break;
         case GLP_LO:
            row->lb = lb, row->ub = 0.0;
            if (row->stat != GLP_BS) row->stat = GLP_NL;
            break;
         case GLP_UP:
            row->lb = 0.0, row->ub = ub;
            if (row->stat != GLP_BS) row->stat = GLP_NU;
            break;
         case GLP_DB:
            row->lb = lb, row->ub = ub;
            if (!(row->stat == GLP_BS ||
                  row->stat == GLP_NL || row->stat == GLP_NU))
               row->stat = (fabs(lb) <= fabs(ub) ? GLP_NL : GLP_NU);
            break;
         case GLP_FX:
            row->lb = row->ub = lb;
            if (row->stat != GLP_BS) row->stat = GLP_NS;
            break;
         default:
            xerror("glp_set_row_bnds: i = %d; type = %d; invalid row type", i, type);
      }
      return;
}

void glp_set_col_bnds(glp_prob *lp, int j, int type, double lb,
      double ub)
{     GLPCOL *col;
      if (!(1 <= j && j <= lp->n))
         xerror("glp_set_col_bnds: j = %d; column number out of range", j);
      col = lp->col[j];
      col->type = type;
      switch (type)
      {  case GLP_FR:
            col->lb = col->ub = 0.0;
            if (col->stat != GLP_BS) col->stat = GLP_NF;
            break;
         case GLP_LO:
            col->lb = lb, col->ub = 0.0;
            if (col->stat != GLP_BS) col->stat = GLP_NL;
            break;
         case GLP_UP:
            col->lb = 0.0, col->ub = ub;
            if (col->stat != GLP_BS) col->stat = GLP_NU;
            break;
         case GLP_DB:
            col->lb = lb, col->ub = ub;
            if (!(col->stat == GLP_BS ||
                  col->stat == GLP_NL || col->stat == GLP_NU))
               col->stat = (fabs(lb) <= fabs(ub) ? GLP_NL : GLP_NU);
            break;
         case GLP_FX:
            col->lb = col->ub = lb;
            if (col->stat != GLP_BS) col->stat = GLP_NS;
            break;
         default:
            xerror("glp_set_col_bnds: j = %d; type = %d; invalid column type", j, type);
      }
      return;
}

void glp_set_obj_coef(glp_prob *lp, int j, double coef)
{
      if (!(0 <= j && j <= lp->n))
         xerror("glp_set_obj_coef: j = %d; column number out of range", j);
      if (j == 0)
         lp->c0 = coef;
      else
         lp->col[j]->coef = coef;
      return;
}

void glp_load_matrix(glp_prob *lp, int ne, const int ia[],
      const int ja[], const double ar[])
{
      GLPROW *row;
      GLPCOL *col;
      GLPAIJ *aij, *next;
      int i, j, k;
      /* clear the constraint matrix */
      for (i = 1; i <= lp->m; i++)
      {  row = lp->row[i];
         while (row->ptr != NULL)
         {  aij = row->ptr;
            row->ptr = aij->r_next;
            dmp_free_atom(lp->pool, aij, sizeof(GLPAIJ)), lp->nnz--;
         }
      }
      xassert(lp->nnz == 0);
      for (j = 1; j <= lp->n; j++) lp->col[j]->ptr = NULL;
      /* load the new contents of the constraint matrix and build its
         row lists */
      if (ne < 0)
         xerror("glp_load_matrix: ne = %d; invalid number of constraint coefficients", ne);
      if (ne > NNZ_MAX)
         xerror("glp_load_matrix: ne = %d; too many constraint coefficients", ne);
      for (k = 1; k <= ne; k++)
      {  /* take indices of new element */
         i = ia[k], j = ja[k];
         /* obtain pointer to i-th row */
         if (!(1 <= i && i <= lp->m))
            xerror("glp_load_matrix: ia[%d] = %d; row index out of range", k, i);
         row = lp->row[i];
         /* obtain pointer to j-th column */
         if (!(1 <= j && j <= lp->n))
            xerror("glp_load_matrix: ja[%d] = %d; column index out of range", k, j);
         col = lp->col[j];
         /* create new element */
         aij = (GLPAIJ*)dmp_get_atom(lp->pool, sizeof(GLPAIJ)), lp->nnz++;
         aij->row = row;
         aij->col = col;
         aij->val = ar[k];
         /* add the new element to the beginning of i-th row list */
         aij->r_prev = NULL;
         aij->r_next = row->ptr;
         if (aij->r_next != NULL) aij->r_next->r_prev = aij;
         row->ptr = aij;
      }
      xassert(lp->nnz == ne);
      /* build column lists of the constraint matrix and check elements
         with identical indices */
      for (i = 1; i <= lp->m; i++)
      {  for (aij = lp->row[i]->ptr; aij != NULL; aij = aij->r_next)
         {  /* obtain pointer to corresponding column */
            col = aij->col;
            /* if there is element with identical indices, it can only
               be found in the beginning of j-th column list */
            if (col->ptr != NULL && col->ptr->row->i == i)
            {  for (k = 1; k <= ne; k++)
                  if (ia[k] == i && ja[k] == col->j) break;
               xerror("glp_load_mat: ia[%d] = %d; ja[%d] = %d; duplicate indices not allowed", k, i, k, col->j);
            }
            /* add the element to the beginning of j-th column list */
            aij->c_prev = NULL;
            aij->c_next = col->ptr;
            if (aij->c_next != NULL) aij->c_next->c_prev = aij;
            col->ptr = aij;
         }
      }
      /* remove zero elements from the constraint matrix */
      for (i = 1; i <= lp->m; i++)
      {  row = lp->row[i];
         for (aij = row->ptr; aij != NULL; aij = next)
         {  next = aij->r_next;
            if (aij->val == 0.0)
            {  /* remove the element from the row list */
               if (aij->r_prev == NULL)
                  row->ptr = next;
               else
                  aij->r_prev->r_next = next;
               if (next == NULL)
                  ;
               else
                  next->r_prev = aij->r_prev;
               /* remove the element from the column list */
               if (aij->c_prev == NULL)
                  aij->col->ptr = aij->c_next;
               else
                  aij->c_prev->c_next = aij->c_next;
               if (aij->c_next == NULL)
                  ;
               else
                  aij->c_next->c_prev = aij->c_prev;
               /* return the element to the memory pool */
               dmp_free_atom(lp->pool, aij, sizeof(GLPAIJ)), lp->nnz--;
            }
         }
      }
      /* invalidate the basis factorization */
      lp->valid = 0;
      return;
}

void delete_prob(glp_prob *lp);

void glp_erase_prob(glp_prob *lp)
{
      delete_prob(lp);
      create_prob(lp);
      return;
}

void delete_prob(glp_prob *lp)
{     lp->magic = 0x3F3F3F3F;
      dmp_delete_pool(lp->pool);
      if (lp->parms != NULL) xfree(lp->parms);
      xfree(lp->head);
      if (lp->bfcp != NULL) xfree(lp->bfcp);
      if (lp->bfd != NULL) bfd_delete_it(lp->bfd);
      return;
}

void glp_delete_prob(glp_prob *lp)
{
      delete_prob(lp);
      xfree(lp);
      return;
}

////////// END API-1


////////// START API-12

int glp_bf_exists(glp_prob *lp)
{     int ret;
      ret = (lp->m == 0 || lp->valid);
      return ret;
}

static int b_col(void *info, int j, int ind[], double val[])
{     glp_prob *lp = (glp_prob*)info;
      int m = lp->m;
      GLPAIJ *aij;
      int k, len;
      xassert(1 <= j && j <= m);
      /* determine the ordinal number of basic auxiliary or structural
         variable x[k] corresponding to basic variable xB[j] */
      k = lp->head[j];
      /* build j-th column of the basic matrix, which is k-th column of
         the scaled augmented matrix (I | -R*A*S) */
      if (k <= m)
      {  /* x[k] is auxiliary variable */
         len = 1;
         ind[1] = k;
         val[1] = 1.0;
      }
      else
      {  /* x[k] is structural variable */
         len = 0;
         for (aij = lp->col[k-m]->ptr; aij != NULL; aij = aij->c_next)
         {  len++;
            ind[len] = aij->row->i;
            val[len] = - aij->row->rii * aij->val * aij->col->sjj;
         }
      }
      return len;
}

static void copy_bfcp(glp_prob *lp);

int glp_factorize(glp_prob *lp)
{     int m = lp->m;
      int n = lp->n;
      GLPROW **row = lp->row;
      GLPCOL **col = lp->col;
      int *head = lp->head;
      int j, k, stat, ret;
      /* invalidate the basis factorization */
      lp->valid = 0;
      /* build the basis header */
      j = 0;
      for (k = 1; k <= m+n; k++)
      {  if (k <= m)
         {  stat = row[k]->stat;
            row[k]->bind = 0;
         }
         else
         {  stat = col[k-m]->stat;
            col[k-m]->bind = 0;
         }
         if (stat == GLP_BS)
         {  j++;
            if (j > m)
            {  /* too many basic variables */
               ret = GLP_EBADB;
               goto fini;
            }
            head[j] = k;
            if (k <= m)
               row[k]->bind = j;
            else
               col[k-m]->bind = j;
         }
      }
      if (j < m)
      {  /* too few basic variables */
         ret = GLP_EBADB;
         goto fini;
      }
      /* try to factorize the basis matrix */
      if (m > 0)
      {  if (lp->bfd == NULL)
         {  lp->bfd = bfd_create_it();
            copy_bfcp(lp);
         }
         switch (bfd_factorize(lp->bfd, m, lp->head, b_col, lp))
         {  case 0:
               /* ok */
               break;
            case BFD_ESING:
               /* singular matrix */
               ret = GLP_ESING;
               goto fini;
            case BFD_ECOND:
               /* ill-conditioned matrix */
               ret = GLP_ECOND;
               goto fini;
            default:
               xassert(lp != lp);
         }
         lp->valid = 1;
      }
      /* factorization successful */
      ret = 0;
fini: /* bring the return code to the calling program */
      return ret;
}

int glp_bf_updated(glp_prob *lp)
{     int cnt;
      if (!(lp->m == 0 || lp->valid))
         xerror("glp_bf_update: basis factorization does not exist");
      cnt = (lp->m == 0 ? 0 : bfd_get_count(lp->bfd));
      return cnt;
}

void glp_get_bfcp(glp_prob *lp, glp_bfcp *parm)
{     glp_bfcp *bfcp = lp->bfcp;
      if (bfcp == NULL)
      {  parm->type = GLP_BF_FT;
         parm->lu_size = 0;
         parm->piv_tol = 0.10;
         parm->piv_lim = 4;
         parm->suhl = GLP_ON;
         parm->eps_tol = 1e-15;
         parm->max_gro = 1e+10;
         parm->nfs_max = 100;
         parm->upd_tol = 1e-6;
         parm->nrs_max = 100;
         parm->rs_size = 0;
      }
      else
         memcpy(parm, bfcp, sizeof(glp_bfcp));
      return;
}

void copy_bfcp(glp_prob *lp)
{     glp_bfcp _parm, *parm = &_parm;
      glp_get_bfcp(lp, parm);
      bfd_set_parm(lp->bfd, parm);
      return;
}

void glp_set_bfcp(glp_prob *lp, const glp_bfcp *parm)
{     glp_bfcp *bfcp = lp->bfcp;
      if (parm == NULL)
      {  /* reset to default values */
         if (bfcp != NULL)
            xfree(bfcp), lp->bfcp = NULL;
      }
      else
      {  /* set to specified values */
         if (bfcp == NULL)
            bfcp = lp->bfcp = (glp_bfcp*)xmalloc(sizeof(glp_bfcp));
         memcpy(bfcp, parm, sizeof(glp_bfcp));
         if (!(bfcp->type == GLP_BF_FT || bfcp->type == GLP_BF_BG ||
               bfcp->type == GLP_BF_GR))
            xerror("glp_set_bfcp: type = %d; invalid parameter", bfcp->type);
         if (bfcp->lu_size < 0)
            xerror("glp_set_bfcp: lu_size = %d; invalid parameter", bfcp->lu_size);
         if (!(0.0 < bfcp->piv_tol && bfcp->piv_tol < 1.0))
            xerror("glp_set_bfcp: piv_tol = %f; invalid parameter", bfcp->piv_tol);
         if (bfcp->piv_lim < 1)
            xerror("glp_set_bfcp: piv_lim = %d; invalid parameter", bfcp->piv_lim);
         if (!(bfcp->suhl == GLP_ON || bfcp->suhl == GLP_OFF))
            xerror("glp_set_bfcp: suhl = %d; invalid parameter", bfcp->suhl);
         if (!(0.0 <= bfcp->eps_tol && bfcp->eps_tol <= 1e-6))
            xerror("glp_set_bfcp: eps_tol = %f; invalid parameter", bfcp->eps_tol);
         if (bfcp->max_gro < 1.0)
            xerror("glp_set_bfcp: max_gro = %f; invalid parameter", bfcp->max_gro);
         if (!(1 <= bfcp->nfs_max && bfcp->nfs_max <= 32767))
            xerror("glp_set_bfcp: nfs_max = %d; invalid parameter", bfcp->nfs_max);
         if (!(0.0 < bfcp->upd_tol && bfcp->upd_tol < 1.0))
            xerror("glp_set_bfcp: upd_tol = %f; invalid parameter", bfcp->upd_tol);
         if (!(1 <= bfcp->nrs_max && bfcp->nrs_max <= 32767))
            xerror("glp_set_bfcp: nrs_max = %d; invalid parameter", bfcp->nrs_max);
         if (bfcp->rs_size < 0)
            xerror("glp_set_bfcp: rs_size = %d; invalid parameter", bfcp->nrs_max);
         if (bfcp->rs_size == 0)
            bfcp->rs_size = 20 * bfcp->nrs_max;
      }
      if (lp->bfd != NULL) copy_bfcp(lp);
      return;
}

////////// END API-12


////////// START API-6

void glp_init_smcp(glp_smcp *parm)
{     parm->msg_lev = GLP_MSG_ALL;
      parm->meth = GLP_PRIMAL;
      parm->pricing = GLP_PT_PSE;
      parm->r_test = GLP_RT_HAR;
      parm->tol_bnd = 1e-7;
      parm->tol_dj = 1e-7;
      parm->tol_piv = 1e-10;
      parm->obj_ll = -DBL_MAX;
      parm->obj_ul = +DBL_MAX;
      parm->it_lim = INT_MAX;
      parm->tm_lim = INT_MAX;
      parm->out_frq = 500;
      parm->out_dly = 0;
      parm->presolve = GLP_OFF;
      return;
}

int solve_lp(glp_prob *P, const glp_smcp *parm)
{     /* solve LP directly without using the preprocessor */
      int ret;
      if (!glp_bf_exists(P))
      {  ret = glp_factorize(P);
         if (ret == 0)
            ;
         else if (ret == GLP_EBADB)
         {
         }
         else if (ret == GLP_ESING)
         {
         }
         else if (ret == GLP_ECOND)
         {
         }
         else
            xassert(ret != ret);
         if (ret != 0) goto done;
      }
         ret = spx_primal(P, parm);
done: return ret;
}

int glp_simplex(glp_prob *P, const glp_smcp *parm)
{     /* solve LP problem with the simplex method */
      glp_smcp _parm;
      int i, j, ret;
      /* check problem object */
      if (P == NULL || P->magic != (int)GLP_PROB_MAGIC)
         xerror("glp_simplex: P = %p; invalid problem object", P);
      /* check control parameters */
      if (parm == NULL)
         parm = &_parm, glp_init_smcp((glp_smcp *)parm);
      /* basic solution is currently undefined */
      P->pbs_stat = P->dbs_stat = GLP_UNDEF;
      P->obj_val = 0.0;
      P->some = 0;
      /* check bounds of double-bounded variables */
      for (i = 1; i <= P->m; i++)
      {  GLPROW *row = P->row[i];
         if (row->type == GLP_DB && row->lb >= row->ub)
         {  ret = GLP_EBOUND;
            goto done;
         }
      }
      for (j = 1; j <= P->n; j++)
      {  GLPCOL *col = P->col[j];
         if (col->type == GLP_DB && col->lb >= col->ub)
         {  ret = GLP_EBOUND;
            goto done;
         }
      }
      /* solve LP problem */
         ret = solve_lp(P, parm);
done: /* return to the application program */
      return ret;
}

void glp_set_row_stat(glp_prob *lp, int i, int stat)
{     GLPROW *row;
      if (!(1 <= i && i <= lp->m))
         xerror("glp_set_row_stat: i = %d; row number out of range", i);
      if (!(stat == GLP_BS || stat == GLP_NL || stat == GLP_NU ||
            stat == GLP_NF || stat == GLP_NS))
         xerror("glp_set_row_stat: i = %d; stat = %d; invalid status", i, stat);
      row = lp->row[i];
      if (stat != GLP_BS)
      {  switch (row->type)
         {  case GLP_FR: stat = GLP_NF; break;
            case GLP_LO: stat = GLP_NL; break;
            case GLP_UP: stat = GLP_NU; break;
            case GLP_DB: if (stat != GLP_NU) stat = GLP_NL; break;
            case GLP_FX: stat = GLP_NS; break;
            default: xassert(row != row);
         }
      }
      if ((row->stat == GLP_BS && stat != GLP_BS) ||
          (row->stat != GLP_BS && stat == GLP_BS))
      {  /* invalidate the basis factorization */
         lp->valid = 0;
      }
      row->stat = stat;
      return;
}

void glp_set_col_stat(glp_prob *lp, int j, int stat)
{     GLPCOL *col;
      if (!(1 <= j && j <= lp->n))
         xerror("glp_set_col_stat: j = %d; column number out of range", j);
      if (!(stat == GLP_BS || stat == GLP_NL || stat == GLP_NU ||
            stat == GLP_NF || stat == GLP_NS))
         xerror("glp_set_col_stat: j = %d; stat = %d; invalid status", j, stat);
      col = lp->col[j];
      if (stat != GLP_BS)
      {  switch (col->type)
         {  case GLP_FR: stat = GLP_NF; break;
            case GLP_LO: stat = GLP_NL; break;
            case GLP_UP: stat = GLP_NU; break;
            case GLP_DB: if (stat != GLP_NU) stat = GLP_NL; break;
            case GLP_FX: stat = GLP_NS; break;
            default: xassert(col != col);
         }
      }
      if ((col->stat == GLP_BS && stat != GLP_BS) ||
          (col->stat != GLP_BS && stat == GLP_BS))
      {  /* invalidate the basis factorization */
         lp->valid = 0;
      }
      col->stat = stat;
      return;
}

double glp_get_col_prim(glp_prob *lp, int j)
{     /*struct LPXCPS *cps = lp->cps;*/
      double prim;
      if (!(1 <= j && j <= lp->n))
         xerror("glp_get_col_prim: j = %d; column number out of range", j);
      prim = lp->col[j]->prim;
      /*if (cps->round && fabs(prim) < 1e-9) prim = 0.0;*/
      return prim;
}

////////// END API-6


} // namespace glpk


////////// START CLASS


LP::LP()
{ m_prob = glpk::glp_create_prob(); setMinimize(); }

LP::~LP()
{ glpk::glp_delete_prob(m_prob); }


void LP::setMinimize()
{ glpk::glp_set_obj_dir(m_prob, GLP_MIN); }

void LP::setMaximize()
{ glpk::glp_set_obj_dir(m_prob, GLP_MAX); }

void LP::addRows(unsigned int rows)
{ glpk::glp_add_rows(m_prob, rows); }

void LP::addColumns(unsigned int cols)
{ glpk::glp_add_cols(m_prob, cols); }

void LP::setObjectiveCoefficient(unsigned int col, double coeff)
{ glpk::glp_set_obj_coef(m_prob, col, coeff); }

void LP::setRowFree(unsigned int row)
{ glpk::glp_set_row_bnds(m_prob, row, GLP_FR, 0.0, 0.0); }

void LP::setRowLowerBounded(unsigned int row, double lower)
{ glpk::glp_set_row_bnds(m_prob, row, GLP_LO, lower, 0.0); }

void LP::setRowUpperBounded(unsigned int row, double upper)
{ glpk::glp_set_row_bnds(m_prob, row, GLP_UP, 0.0, upper); }

void LP::setRowDoubleBounded(unsigned int row, double lower, double upper)
{ glpk::glp_set_row_bnds(m_prob, row, GLP_DB, lower, upper); }

void LP::setRowFixed(unsigned int row, double value)
{ glpk::glp_set_row_bnds(m_prob, row, GLP_FX, value, value); }

void LP::setColumnFree(unsigned int col)
{ glpk::glp_set_col_bnds(m_prob, col, GLP_FR, 0.0, 0.0); }

void LP::setColumnLowerBounded(unsigned int col, double lower)
{ glpk::glp_set_col_bnds(m_prob, col, GLP_LO, lower, 0.0); }

void LP::setColumnUpperBounded(unsigned int col, double upper)
{ glpk::glp_set_col_bnds(m_prob, col, GLP_UP, 0.0, upper); }

void LP::setColumnDoubleBounded(unsigned int col, double lower, double upper)
{ glpk::glp_set_col_bnds(m_prob, col, GLP_DB, lower, upper); }

void LP::setColumnFixed(unsigned int col, double value)
{ glpk::glp_set_col_bnds(m_prob, col, GLP_FX, value, value); }

void LP::setConstraintMatrix(std::vector<unsigned int> const& row, std::vector<unsigned int> const& col, std::vector<double> const& value)
{
	unsigned int n = row.size();
	SHARK_CHECK(col.size() == n && value.size() == n, "[LP::setMatrix] vector sizes must agree");
	glpk::glp_load_matrix(m_prob, (int)n - 1, (int*)&row[0], (int*)&col[0], &value[0]);
}

void LP::setRowStatus(unsigned int row, bool basic)
{
	glpk::glp_set_row_stat(m_prob, row, basic ? GLP_BS : GLP_LO);
}

void LP::setColumnStatus(unsigned int col, bool basic)
{
	glpk::glp_set_col_stat(m_prob, col, basic ? GLP_BS : GLP_LO);
}

bool LP::solve()
{
	int ret = glp_simplex(m_prob, NULL);
	return (ret == GLP_OPT);
}

double LP::solution(unsigned int col)
{ return glp_get_col_prim(m_prob, col); }


////////// END CLASS


} // namespace shark
