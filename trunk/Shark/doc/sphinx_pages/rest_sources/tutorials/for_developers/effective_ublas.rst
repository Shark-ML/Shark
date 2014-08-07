Effective uBlas and General Code Optimization
*********************************************

Modern architectures are increasingly powerful for scientific computation. However, programmers
need to know the basic bottlenecks in order to leverage full performance. When using uBlas, it
is quite easy to achieve reasonable performance, since some basic optimization has already been
done. But there are still traps. This article tries to give general optimization advice in the
first part and answer a few questions regarding efficient use of uBlas in the second part.

Bottlenecks in modern architectures
========================================

Modern processors offer a wide variety of SIMD-Instructions. SIMD (Single Instruction, Multiple
Data) tries to apply the same instruction on a whole block of data. For example a scalar product
can be calculated using SIMD by first multiplying the components of both vectors as blocks and
adding these results together in the second step. SIMD instruction sets are better known as the
SSE-Family. This can add to the computational power of modern processors. And even though the
problem of vectorization -- the process of transforming normal code into SIMD instructions --
is not an easy one, it does not make sense to optimize for a minimum of instructions.
	
As a result, it is safe to say that the bottleneck on which to focus makes most sense is memory
rather than processor vectorization support. Let's look at the following two ways to compute a
matrix-matrix product C = AB::
  
  void prod1(const RealMatrix& C, const RealMatrix& A, const RealMatrix& B){
    for(std::size_t i = 0; i != A.size1(); ++i){
      for(std::size_t j = 0; j != A.size1(); ++j){
        for(std::size_t k = 0; k != A.size2(); ++k){
	  C(i,j)+= A(i,k) * B(k,j);
        }
      }
    }
  }
  
  void prod2(const RealMatrix& C, const RealMatrix& A, const RealMatrix& B){
    for(std::size_t i = 0; i != A.size1(); ++i){
      for(std::size_t k = 0; k != A.size2(); ++k){
        for(std::size_t j = 0; j != A.size1(); ++j){
	  C(i,j)+= A(i,k) * B(k,j);
        }
      }
    }
  }

The first implementation is the canonical one using the typical definition of a matrix product::
  
  C(i,j) = row(A,i) * column (B,j)

The second prod just swaps the j and k-loop. It is factor 5 faster for 768x768 matrices on a Core2 processor.
So, what is the difference? Cache misses! Processors are so fast that they need to call memory in blocks and 
cache it in a series of caches long before it is actually needed. The easiest way to do caching in advance is to
query the RAM for the next memory blocks after the one currently used. This means that once our code begins to
jump through memory, we effectively disable this caching mechanism, and the processor needs to wait until the
RAM delivered the last request. For modern processors this can be up to 100 cycles. Since we iterate over the 
columns, we jump over 768x8 Byte every time - that are a whooping 6 kilobyte of data! This is the main difference 
between prod1 and prod2.

Thus, the general optimization rule must be: Keep your memory aligned and iterate over it element by element, 
even if this might force you to calculate intermediate results multiple times! This is not true for very expensive 
operations like exponential functions or logarithms -- but a good rule of thumb nonetheless.

There are ways out of this dilemma using newer SIMD instructions which can access memory using strides. Still, 
these cannot achieve the same performance as under proper alignment.

General advice regarding uBlas and matrix products
===================================================
UBlas in general is optimized such that it avoids temporary results. This is especially good for vector-vector
operations, also called "BLAS1". For vector-vector operations, intermediate results are never needed. However,
as we have seen above, using no intermediate results in a matrix-matrix product hampers performance. In the following
table, you can see the results of uBlas prod, axpy_prod and block_prod, as well as the previous two algorithms for 
comparison, using 768x768 matrices on gcc 4.6.1 and optimization level ``-O3`` in the first row, and ``-O3 -mcore2``
in the second. All results are in seconds.

=============  ========  ==========  ===========  =======  ======
optimizations  prod      axpy_prod   block_prod   prod1    prod2
-------------  --------  ----------  -----------  -------  ------
-O3            2.05      0.88        0.75         4.95     0.88
-O3 -mcore2    1.18      0.85        0.75         5.03     0.86
=============  ========  ==========  ===========  =======  ======

`block_prod` is a special uBlas product where the size of the SIMD-block is given in advance. The choice of the
block size is crucial for the performance, and depends on both architecture as well as matrix size. The normal prod is a lot
faster than the canonical prod1, but still slower than everything else. Thus the advice is to use axpy_prod for
matrix products. Please note, that while there is no `noalias` needed for the axpy_prod, it assumes that
C is a different matrix than A and B!


Why is uBlas prod slow?
==========================
UBlas uses a lot of template magic and thus it makes sense to ask why axpy_prod is needed and/or faster?
The commands in comparison look like::

  noalias(A) = prod(B,C);
  axpy_prod(B,C,A,false);

The second notation is clearly a bit more clumsy, and we would like to use the first notation. Let's take a deeper look
into what happens in the prod case. uBlas uses a technique called expression templates. This means that the right
side of the equation itself calculates nothing, but instead is a complex object definition. The expression itself
can be roughly translated into::

  matrix_matrix_prod<RealMatrix,RealMatrix> expression(A,B);
  A.assign(expression);

If we used a more complex equation, the resulting type would also be a lot more complicated. Now, what happens when
assign is called? Again, this can be roughly translated into::
  
  RealMatrix::assign(const matrix_expression& expression){
    for(std::size_t i = 0; i != A.size1(); ++i){
      for(std::size_t j = 0; j != A.size1(); ++j){
	A(i,j) = expression.apply(i,j)
      }
    }
  }

Since the expression does not calculate the result in advance, the only way to define apply is using the inner loop
of prod1. Since it is highly optimized, performance is not as bad as expected. And we see how much modern compilers
can actually achieve given the proper optimization hints. Still the code cannot be as good as the axpy_prod, since 
the latter implements prod2 in our case which is optimal.
