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



// define the basic dimensionality of the problem
unsigned int useful_dim = 5;
unsigned int noise_dim = 5;
unsigned int total_dim = useful_dim + noise_dim;

RealVector run_one_trial( bool verbose) {

        // set up the classification problem from a DataDistribution
    PamiToy problem( useful_dim, noise_dim );

    // construct training and test sets from the problem distribution
    unsigned int train_size = 500;
    unsigned int test_size = 5000;
    ClassificationDataset train = problem.generateDataset( train_size );
    ClassificationDataset test = problem.generateDataset( test_size );
    
        // normalize data as usual
    Normalizer<> normalizer;
    NormalizeComponentsUnitVariance<> normalizationTrainer(false);
    normalizationTrainer.train( normalizer, train.inputs() );
    train = transformInputs( train, normalizer );
    test = transformInputs( test, normalizer );
    
        // set up the ArdKernel
    DenseARDKernel kernel( total_dim, 0.1 ); //for now with arbitrary value for gamma (gets properly initialized later)
    
        // set up partitions for cross-validation
    unsigned int num_folds = 5;
    CVFolds<ClassificationDataset> cv_folds = createCVIID( train, num_folds );
    
        // set up the learning machine
    bool log_enc_c = true; //use log encoding for the regularization parameter C
    QpStoppingCondition stop(1e-12); //use a very conservative stopping criterion for the individual SVM runs
    SvmLogisticInterpretation<> mlms( cv_folds, &kernel, log_enc_c, &stop ); //the main class for this tutorial
    //SvmLogisticInterpretation<> mlms( cv_folds, &kernel, log_enc_c ); //also possible without stopping criterion
    
        // set up a starting point for the optimization process
    RealVector start( total_dim+1 );
    if ( log_enc_c ) start( total_dim ) = 0.0; else start( total_dim ) = 1.0; //start at C = 1.0
    for ( unsigned int k=0; k<total_dim; k++ )
        start(k) = 0.5 / total_dim;
    
        // for illustration purposes, we also evalute the model selection criterion a single time at the starting point
    double start_value = mlms.eval( start );
    
        if ( verbose ) {
        std::cout << "Value of model selection criterion at starting point: " << start_value << std::endl << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl;
        std::cout << " ----------- Beginning gradient-based optimization of MLMS criterion ------------ " << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl << std::endl;
    }
    
        // set up the optimizer
    Rprop<> rprop;
    double stepsize = 0.1;
    double stop_delta = 1e-3;
    mlms.init();
    rprop.init( mlms, start, stepsize );
    unsigned int its = 50;
    
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
    
        if ( verbose ) {
        std::cout << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl;
        std::cout << " ----------- Done with gradient-based optimization of MLMS criterion ------------ " << std::endl;
        std::cout << " -------------------------------------------------------------------------------- " << std::endl << std::endl;
    }
    if ( verbose ) std::cout << std::endl << std::endl << " EVALUATION of hyperparameters found:" << std::endl << std::endl << std::endl;
    
        double C_reg; //will hold regularization parameter
    double test_error_v1, train_error_v1; //will hold errors determined via method 1
    double test_error_v2, train_error_v2; //will hold errors determined via method 2
    
    // BEGIN POSSIBILITY ONE OF HYPERPARAMETER COPY
        if ( verbose ) std::cout  << std::endl << " Possibility 1: copy kernel parameters via eval() and C by hand..." << std::endl << std::endl;
    
        // copy final parameters, variant one
    double end_value = mlms.eval( rprop.solution().point ); //this at the same time copies the most recent parameters from rprop to the kernel.
    C_reg = ( log_enc_c ? exp( rprop.solution().point(total_dim) ) : rprop.solution().point(total_dim) ); //ATTENTION: mind the encoding
    
        if ( verbose ) {
        std::cout << "    Value of model selection criterion at final point: " << end_value << std::endl;
        std::cout << "    Done optimizing the SVM hyperparameters. The final parameters (true/unencoded) are:" << std::endl << std::endl;
        std::cout << "        C = " << C_reg << std::endl;
        for ( unsigned int i=0; i<total_dim; i++ )
            std::cout << "        gamma(" << i << ") = " << kernel.parameterVector()(i)*kernel.parameterVector()(i) << std::endl;
        std::cout << std::endl << "    (as also given by kernel.gammaVector() : " << kernel.gammaVector() << " ) " << std::endl;
    }
    
        // construct and train the final learner
    KernelClassifier<RealVector> svm_v1;
    CSvmTrainer<RealVector> trainer_v1( &kernel, C_reg, true, log_enc_c ); //encoding does not really matter in this case b/c it does not affect the ctor
    if ( verbose ) {
        std::cout << std::endl << std::endl << "    Used mlms.eval(...) to copy kernel.parameterVector() " << kernel.parameterVector() << std::endl;
        std::cout << "    into trainer_v1.parameterVector() " << trainer_v1.parameterVector() << std::endl;
        std::cout << "    , where C (the last parameter) was set manually to " << trainer_v1.C() << std::endl << std::endl << std::endl;
    }
    trainer_v1.train( svm_v1, train ); //the kernel has the right parameters, and we copied C, so we are good to go
    
        // evaluate the final trained classifier on training and test set
    ZeroOneLoss<unsigned int> loss_v1;
    Data<unsigned int> output_v1; //real-valued output
    output_v1 = svm_v1( train.inputs() );
    train_error_v1 = loss_v1.eval( train.labels(), output_v1 );
    output_v1 = svm_v1( test.inputs() );
    test_error_v1 = loss_v1.eval( test.labels(), output_v1 );
    if ( verbose ) {
        std::cout << "    training error via possibility 1:  " <<  train_error_v1 << std::endl;
        std::cout << "    test error via possibility 1:      " << test_error_v1 << std::endl << std::endl << std::endl;
    }
        // END POSSIBILITY ONE OF HYPERPARAMETER COPY

    // BEGIN POSSIBILITY TWO OF HYPERPARAMETER COPY
        if ( verbose ) std::cout << std::endl << " Possibility 2: copy best parameters via solution().point()..." << std::endl << std::endl;
    
        KernelClassifier<RealVector> svm_v2;
    CSvmTrainer<RealVector> trainer_v2( &kernel, 0.1, true, log_enc_c ); //ATTENTION: must be constructed with same log-encoding preference
    trainer_v2.setParameterVector( rprop.solution().point ); //copy best hyperparameters to svm trainer
    
        if ( verbose ) {
        std::cout << "    Copied rprop.solution().point = " << rprop.solution().point << std::endl;
        std::cout << "    into trainer_v2.parameterVector(), now = " << trainer_v2.parameterVector() << std::endl << std::endl << std::endl;
    }
    
        trainer_v2.train( svm_v2, train );
    
        // evaluate the final trained classifier on training and test set
    ZeroOneLoss<unsigned int> loss_v2;
    Data<unsigned int> output_v2; //real-valued output
    output_v2 = svm_v2( train.inputs() );
    train_error_v2 = loss_v2.eval( train.labels(), output_v2 );
    output_v2 = svm_v2( test.inputs() );
    test_error_v2 = loss_v2.eval( test.labels(), output_v2 );
    if ( verbose ) {
        std::cout << "    training error via possibility 2:  " <<  train_error_v2 << std::endl;
        std::cout << "    test error via possibility 2:      " << test_error_v2 << std::endl << std::endl << std::endl;
        std::cout << std::endl << "That's all folks - we are done!" << std::endl;
    }
        // END POSSIBILITY TWO OF HYPERPARAMETER COPY

        // copy the best parameters, as well as performance values into averaging vector:
    RealVector final_params(total_dim+3);
    final_params(total_dim) = C_reg;
    for ( unsigned int i=0; i<total_dim; i++ )
        final_params(i) = rprop.solution().point(i)*rprop.solution().point(i);
    final_params(total_dim+1) = train_error_v1;
    final_params(total_dim+2) = test_error_v1;
    return final_params;
    
}


int main() {

        // run one trial with output
    run_one_trial( true);
    std::cout << "\nNOW REPEAT WITH 100 TRIALS: now we do the exact same thing multiple times in a row, and note the average kernel weights. Please wait." << std::endl << std::endl;
    
        // run several trials without output, and average the results
    unsigned int num_trials = 100;
    Data<RealVector> many_results(num_trials,RealVector(total_dim+3));//each row is one run of resulting hyperparameters
    for ( unsigned int i=0; i<num_trials; i++ ) {
        many_results.element(i) = run_one_trial(false);
        std::cout << "." << std::flush;
    }
    std::cout << "\n" << std::endl;
    
        RealVector overall_mean, overall_variance;
    meanvar( many_results, overall_mean, overall_variance );
    for ( unsigned int i=0; i<total_dim+1; i++ ) {
        std::cout << "avg-param(" << i << ") = " << overall_mean(i) << " +- "<< overall_variance(i) << std::endl;
    }
    std::cout << std::endl << "avg-error-train = " << overall_mean(total_dim+1) << " +- "<< overall_variance(total_dim+1) << std::endl;
    std::cout << "avg-error-test  = " << overall_mean(total_dim+2) << " +- "<< overall_variance(total_dim+2) << std::endl;
    
}
