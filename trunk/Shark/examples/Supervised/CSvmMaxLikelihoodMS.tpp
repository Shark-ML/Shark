//###begin<includes_namespaces>
#include <shark/Data/Dataset.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Data/Statistics.h>
#include <shark/Models/Kernels/ArdKernel.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/SvmLogisticInterpretation.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>

using namespace std;
using namespace shark;
//###end<includes_namespaces>



//###begin<basic_dims>
// define the basic dimensionality of the problem
unsigned int useful_dim = 5;
unsigned int noise_dim = 5;
unsigned int total_dim = useful_dim + noise_dim;
//###end<basic_dims>

//###begin<function_one_trial_start>
RealVector run_one_trial( bool verbose) {
//###end<function_one_trial_start>

    //###begin<create_problem>
    // set up the classification problem from a DataDistribution
    PamiToy problem( useful_dim, noise_dim );

    // construct training and test sets from the problem distribution
    unsigned int train_size = 500;
    unsigned int test_size = 5000;
    ClassificationDataset train = problem.generateDataset( train_size );
    ClassificationDataset test = problem.generateDataset( test_size );
    //###end<create_problem>

    //###begin<normalize_data>
    // normalize data as usual
    Normalizer<> normalizer;
    NormalizeComponentsUnitVariance<> normalizationTrainer;
    normalizationTrainer.train( normalizer, train.inputs() );
    train = transformInputs( train, normalizer );
    test = transformInputs( test, normalizer );
    //###end<normalize_data>

    //###begin<setup_kernel>
    // set up the ArdKernel
    DenseARDKernel kernel( total_dim, 0.1 ); //for now with arbitrary value for gamma (gets properly initialized later)
    //###end<setup_kernel>

    //###begin<setup_cv>
    // set up partitions for cross-validation
    unsigned int num_folds = 5;
    CVFolds<ClassificationDataset> cv_folds = createCVIID( train, num_folds );
    //###end<setup_cv>

    //###begin<setup_mlms>
    // set up the learning machine
    bool log_enc_c = true; //use log encoding for the regularization parameter C
    QpStoppingCondition stop(1e-12); //use a very conservative stopping criterion for the individual SVM runs
    SvmLogisticInterpretation<> mlms( cv_folds, &kernel, log_enc_c, &stop ); //the main class for this tutorial
    //SvmLogisticInterpretation<> mlms( cv_folds, &kernel, log_enc_c ); //also possible without stopping criterion
    //###end<setup_mlms>

    //###begin<setup_starting_points>
    // set up a starting point for the optimization process
    RealVector start( total_dim+1 );
    if ( log_enc_c ) start( total_dim ) = 0.0; else start( total_dim ) = 1.0; //start at C = 1.0
    for ( unsigned int k=0; k<total_dim; k++ )
        start(k) = 0.5 / total_dim;
    //###end<setup_starting_points>

    //###begin<eval_init_point>
    // for illustration purposes, we also evalute the model selection criterion a single time at the starting point
    double start_value = mlms.eval( start );
    //###end<eval_init_point>

    //###begin<print_init_eval>
    if ( verbose ) {
        std::cout << "Value of model selection criterion at starting point: " << start_value << std::endl << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl;
        std::cout << " ----------- Beginning gradient-based optimization of MLMS criterion ------------ " << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl << std::endl;
    }
    //###end<print_init_eval>

    //###begin<setup_optimizer>
    // set up the optimizer
    IRpropPlus rprop;
    double stepsize = 0.1;
    double stop_delta = 1e-3;
    rprop.init( mlms, start, stepsize );
    unsigned int its = 50;
    //###end<setup_optimizer>

    //###begin<go_go_optimize>
    // start the optimization loop
    for (unsigned int i=0; i<its; i++) {
        rprop.step( mlms );
        if ( verbose )
            std::cout << "iteration " << i << ": current NCLL = " <<  rprop.solution().value << " at parameter: " << rprop.solution().point << std::endl;
        if ( rprop.maxDelta() < stop_delta ) {
            if ( verbose ) std::cout << "    Rprop quit pecause of small progress " << std::endl;
            break;
        }
    }
    //###end<go_go_optimize>

    //###begin<print_msg>
    if ( verbose ) {
        std::cout << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl;
        std::cout << " ----------- Done with gradient-based optimization of MLMS criterion ------------ " << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl << std::endl;
    }
    if ( verbose ) std::cout << std::endl << std::endl << " EVALUATION of hyperparameters found:" << std::endl << std::endl << std::endl;
    //###end<print_msg>

    //###begin<helper_vars>
    double C_reg; //will hold regularization parameter
    double test_error_v1, train_error_v1; //will hold errors determined via method 1
    double test_error_v2, train_error_v2; //will hold errors determined via method 2
    //###end<helper_vars>

    // BEGIN POSSIBILITY ONE OF HYPERPARAMETER COPY
    //###begin<possib_one_verbose>
    if ( verbose ) std::cout  << std::endl << " Possibility 1: copy kernel parameters via eval() and C by hand..." << std::endl << std::endl;
    //###end<possib_one_verbose>

    //###begin<possib_one_copy_results>
    // copy final parameters, variant one
    double end_value = mlms.eval( rprop.solution().point ); //this at the same time copies the most recent parameters from rprop to the kernel.
    C_reg = ( log_enc_c ? exp( rprop.solution().point(total_dim) ) : rprop.solution().point(total_dim) ); //ATTENTION: mind the encoding
    //###end<possib_one_copy_results>

    //###begin<possib_one_print_results>
    if ( verbose ) {
        std::cout << "    Value of model selection criterion at final point: " << end_value << std::endl;
        std::cout << "    Done optimizing the SVM hyperparameters. The final parameters (true/unencoded) are:" << std::endl << std::endl;
        std::cout << "        C = " << C_reg << std::endl;
        for ( unsigned int i=0; i<total_dim; i++ )
            std::cout << "        gamma(" << i << ") = " << kernel.parameterVector()(i)*kernel.parameterVector()(i) << std::endl;
        std::cout << std::endl << "    (as also given by kernel.gammaVector() : " << kernel.gammaVector() << " ) " << std::endl;
    }
    //###end<possib_one_print_results>

    //###begin<possib_one_final_training>
    // construct and train the final learner
    KernelExpansion<RealVector> svm_v1( &kernel, true );
    CSvmTrainer<RealVector> trainer_v1( &kernel, C_reg, log_enc_c ); //encoding does not really matter in this case b/c it does not affect the ctor
    if ( verbose ) {
        std::cout << std::endl << std::endl << "    Used mlms.eval(...) to copy kernel.parameterVector() " << kernel.parameterVector() << std::endl;
        std::cout << "    into trainer_v1.parameterVector() " << trainer_v1.parameterVector() << std::endl;
        std::cout << "    , where C (the last parameter) was set manually to " << trainer_v1.C() << std::endl << std::endl << std::endl;
    }
    trainer_v1.train( svm_v1, train ); //the kernel has the right parameters, and we copied C, so we are good to go
    //###end<possib_one_final_training>

    //###begin<possib_one_final_eval>
    // evaluate the final trained classifier on training and test set
    ZeroOneLoss<unsigned int, RealVector> loss_v1;
    Data<RealVector> output_v1; //real-valued output
    output_v1 = svm_v1( train.inputs() );
    train_error_v1 = loss_v1.eval( train.labels(), output_v1 );
    output_v1 = svm_v1( test.inputs() );
    test_error_v1 = loss_v1.eval( test.labels(), output_v1 );
    if ( verbose ) {
        std::cout << "    training error via possibility 1:  " <<  train_error_v1 << std::endl;
        std::cout << "    test error via possibility 1:      " << test_error_v1 << std::endl << std::endl << std::endl;
    }
    //###end<possib_one_final_eval>
    // END POSSIBILITY ONE OF HYPERPARAMETER COPY

    // BEGIN POSSIBILITY TWO OF HYPERPARAMETER COPY
    //###begin<possib_two_verbose>
    if ( verbose ) std::cout << std::endl << " Possibility 2: copy best parameters via solution().point()..." << std::endl << std::endl;
    //###end<possib_two_verbose>

    //###begin<possib_two_copy_results>
    KernelExpansion<RealVector> svm_v2( &kernel, true );
    CSvmTrainer<RealVector> trainer_v2( &kernel, 0.1, log_enc_c ); //ATTENTION: must be constructed with same log-encoding preference
    trainer_v2.setParameterVector( rprop.solution().point ); //copy best hyperparameters to svm trainer
    //###end<possib_two_copy_results>

    //###begin<possib_two_print_results>
    if ( verbose ) {
        std::cout << "    Copied rprop.solution().point = " << rprop.solution().point << std::endl;
        std::cout << "    into trainer_v2.parameterVector(), now = " << trainer_v2.parameterVector() << std::endl << std::endl << std::endl;
    }
    //###end<possib_two_print_results>

    //###begin<possib_two_final_training>
    trainer_v2.train( svm_v2, train );
    //###end<possib_two_final_training>

    //###begin<possib_two_final_eval>
    // evaluate the final trained classifier on training and test set
    ZeroOneLoss<unsigned int, RealVector> loss_v2;
    Data<RealVector> output_v2; //real-valued output
    output_v2 = svm_v2( train.inputs() );
    train_error_v2 = loss_v2.eval( train.labels(), output_v2 );
    output_v2 = svm_v2( test.inputs() );
    test_error_v2 = loss_v2.eval( test.labels(), output_v2 );
    if ( verbose ) {
        std::cout << "    training error via possibility 2:  " <<  train_error_v2 << std::endl;
        std::cout << "    test error via possibility 2:      " << test_error_v2 << std::endl << std::endl << std::endl;
        std::cout << std::endl << "That's all folks - we are done!" << std::endl;
    }
    //###end<possib_two_final_eval>
    // END POSSIBILITY TWO OF HYPERPARAMETER COPY

    //###begin<copy_params_for_averaging>
    // copy the best parameters, as well as performance values into averaging vector:
    RealVector final_params(total_dim+3);
    final_params(total_dim) = C_reg;
    for ( unsigned int i=0; i<total_dim; i++ )
        final_params(i) = rprop.solution().point(i)*rprop.solution().point(i);
    final_params(total_dim+1) = train_error_v1;
    final_params(total_dim+2) = test_error_v1;
    return final_params;
    //###end<copy_params_for_averaging>

//###begin<function_one_trial_end>
}
//###end<function_one_trial_end>


//###begin<main_start>
int main() {
//###end<main_start>

    //###begin<main_run_one_trial>
    // run one trial with output
    run_one_trial( true);
    std::cout << "\nNOW REPEAT WITH 100 TRIALS: now we do the exact same thing multiple times in a row, and note the average kernel weights. Please wait." << std::endl << std::endl;
    //###end<main_run_one_trial>

    //###begin<main_run_hundred_trials>
    // run several trials without output, and average the results
    unsigned int num_trials = 100;
    Data<RealVector> many_results(num_trials,RealVector(total_dim+3));//each row is one run of resulting hyperparameters
    for ( unsigned int i=0; i<num_trials; i++ ) {
        many_results.element(i) = run_one_trial(false);
        std::cout << "." << std::flush;
    }
    std::cout << "\n" << std::endl;
    //###end<main_run_hundred_trials>

    //###begin<main_calc_print_results>
    RealVector overall_mean, overall_variance;
    meanvar( many_results, overall_mean, overall_variance );
    for ( unsigned int i=0; i<total_dim+1; i++ ) {
        std::cout << "avg-param(" << i << ") = " << overall_mean(i) << " +- "<< overall_variance(i) << std::endl;
    }
    std::cout << std::endl << "avg-error-train = " << overall_mean(total_dim+1) << " +- "<< overall_variance(total_dim+1) << std::endl;
    std::cout << "avg-error-test  = " << overall_mean(total_dim+2) << " +- "<< overall_variance(total_dim+2) << std::endl;
    //###end<main_calc_print_results>

//###begin<main_end>
}
//###end<main_end>
