#!/bin/ksh

function PerformTest
{
	echo $EO "testing $1 .... $EC"
if (./$1 $2 $3 $4 $5 $6 $7 $8 $9) >/dev/null 2>&1; then
    echo "correct!" 
    success=$(expr $success + 1)
else
    echo "is incorrect!"
    errors=$(expr $errors + 1)
fi
}


er=`echo -n ""`
if test "X$er" = "X-n "
then
    EC="\c"
    EO=""
else
    EC=""
    EO="-n"
fi

cd examples

echo ""
echo "testing ReClaM"
echo "--------------"
cd ReClaM
make >/dev/null 2>&1
errors=0
success=0
PerformTest CrossValidation
PerformTest KernelOptimization
PerformTest KM
PerformTest KNN
PerformTest LinearClassifierTest
PerformTest LinearRegressionTest
PerformTest PCAtest
PerformTest simpleFFNet
PerformTest simpleFFNetSource
PerformTest simpleMSERNNet
PerformTest simpleRBFNet
PerformTest simpleRNNet
PerformTest SvmApproximationExample
PerformTest SVMclassification
PerformTest SVMregression
ReClaM_errors=$errors
ReClaM_success=$success
cd ..

echo ""
echo "testing EALib"
echo "-------------"
cd EALib
make >/dev/null  2>&1
errors=0
success=0
PerformTest TSP_GA
PerformTest ackleyES
PerformTest countingOnes
PerformTest integerES
PerformTest paraboloidCMA
PerformTest paraboloidElitistCMA
PerformTest steadyState
PerformTest sphereGA
EALib_errors=$errors
EALib_success=$success
cd ..

echo ""
echo "testing LinAlg"
echo "--------------"
cd LinAlg
make >/dev/null  2>&1
errors=0
success=0
PerformTest linalg_simple_test
PerformTest cblnsrch_test
PerformTest dlinmin_test
PerformTest eigenerr_test
PerformTest eigensort_test
PerformTest eigensymm_test
PerformTest eigensymmJacobi_test
PerformTest eigensymmJacobi2_test
PerformTest fft_test
PerformTest g_inverse_matrix
PerformTest covar_corrcoef_test
PerformTest detsymm_test
PerformTest linmin_test
PerformTest lnsrch_test
PerformTest rank_decomp_test
PerformTest rank_test
PerformTest svd_test
PerformTest svdrank_test
PerformTest svdsort_test
LinAlg_errors=$errors
LinAlg_success=$success
cd ..

echo ""
echo "testing Mixture"
echo "---------------"
cd Mixture
make >/dev/null  2>&1
errors=0
success=0
PerformTest rbfn-example
Mixture_errors=$errors
Mixture_success=$success
cd ..

echo ""
echo "testing FileUtil"
echo "----------------"
cd FileUtil
make >/dev/null  2>&1
errors=0
success=0
PerformTest FileUtilSimple parameters.conf
FileUtil_errors=$errors
FileUtil_success=$success

errors=$(expr $ReClaM_errors + $LinAlg_errors + $EALib_errors + $Mixture_errors + $FileUtil_errors)
success=$(expr $ReClaM_success + $LinAlg_success + $EALib_success + $Mixture_success + $FileUtil_success)

echo ""
echo "selfPerformTest complete with $errors errors."
echo ""
