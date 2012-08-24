#include "fastlib/fastlib.h"
#include "cake_smo.h"

void GetTheFold(Dataset &cv_train_data,Dataset &cv_test_data,
		 index_t fold_num,int total_num_folds, 
		 ArrayList <int> &random_permutation_array_list,
		 Dataset &dset_train)
{
  
  // The crossvalidawhtion folds
  
  dset_train.SplitTrainTest(total_num_folds,fold_num,
			    random_permutation_array_list,
			    &cv_train_data,&cv_test_data);
  
}

int main(int argc, char *argv[]){


  fx_module *root = fx_init(argc, argv, NULL);
  Matrix train_data;
  Matrix test_data;
  
  
  // Read data first

  const char *train_data_file=fx_param_str_req(NULL,"train_data");
  const char *test_data_file=fx_param_str_req(NULL,"test_data");

  data::Load(train_data_file,&train_data);
  data::Load(test_data_file,&test_data);  
  
  int num_train_points=train_data.n_cols();

  bool cv_mode=fx_param_bool(NULL,"cv_mode",true);

  fx_timer_start(NULL,"timer");
  if(cv_mode){
  
    // Before we crossvalidate lets see what the actual settings are.
    bool true_test_densities_file= 
      fx_param_bool(NULL,"true_test_densities",true);
    
    bool regression=fx_param_bool(NULL,"regression",false);
    bool classification=fx_param_bool(NULL,"classification",false);

    // Now kill these parameters
    fx_set_param_bool(NULL,"true_test_densities",false);
    fx_set_param_bool(NULL,"regression",false);
    fx_set_param_bool(NULL,"classification",false);

    int num_folds=5;
    // In this case we will be crossvalidating and hence fill up the
    // regularization vector.
    
    int reg_param_len=5;
    Vector reg_param_vec;
    reg_param_vec.Init(reg_param_len);
    printf("number of train points=%d...\n",num_train_points);
    for(int i=0;i<reg_param_len;i++){
      
      reg_param_vec[i]=pow(5,-12+2*i)*pow(num_train_points,2);
    }
    printf("The regularization vector is ...\n");
    reg_param_vec.PrintDebug();
    
    // TODO: By reordering these loops we can be slightly more
    // efficient in our computations. Make the outerloop the fold and
    // the innter loop the regularization parameter

    Dataset dset_train;
    dset_train.matrix().Alias(train_data);
    
    double opt_reg_val;
    double opt_lscv_val=DBL_MAX;
    for(int param_num=0;param_num<reg_param_len;param_num++){

      double reg_param=reg_param_vec[param_num];

      void *random_permutation;
      
      random_permutation=(void *)malloc(num_train_points*sizeof(index_t));
      math::MakeRandomPermutation(num_train_points,
				  (index_t *) random_permutation);
      
      // Lets typecast random_permutation as ArrayList <index_t>
      
      ArrayList <int> random_permutation_array_list;
      random_permutation_array_list.InitAlias((int *)random_permutation,
					      num_train_points);
      
      
      double average_lscv_on_test_data=0;
      for(int fold_num=0;fold_num<num_folds;fold_num++){
	
	Dataset cv_train_data,cv_test_data;
	GetTheFold(cv_train_data,cv_test_data,
		    fold_num,num_folds,
		    random_permutation_array_list,dset_train);
	// Now call the optimization routine

	
	CAKESMO cake_smo;
	cake_smo.Init(cv_train_data.matrix(),cv_test_data.matrix(),reg_param);
	cake_smo.Estimate();
	double lscv=cake_smo.get_lscv_on_test_data();
	average_lscv_on_test_data+=lscv;
      }
      average_lscv_on_test_data/=num_folds;
      if(average_lscv_on_test_data<opt_lscv_val){

	// Set the optimal regularization value to this param
	
	opt_reg_val=reg_param_vec[param_num];
	opt_lscv_val=average_lscv_on_test_data;
      }

      printf("reg param=%f, LSCV =%f...\n",reg_param_vec[param_num],
	     average_lscv_on_test_data);
    }
    // We have finished crossvalidation
    printf("Optimal reg param is %f...\n",opt_reg_val);
    printf("Optimal LSCV value is %f...\n",opt_lscv_val);

    // Finally run with optimal parameter
    fx_set_param_bool(NULL,"true_test_densities",true_test_densities_file);
    fx_set_param_bool(NULL,"regression",regression);
    fx_set_param_bool(NULL,"classification",classification);

    CAKESMO cake_smo;
    cake_smo.Init(train_data,test_data,opt_reg_val);
    cake_smo.Estimate();
    //cake_smo.ClassificationTask();
    cake_smo.PrintTestDensities();
  }
  else{
    // This means we are not doing any crossvalidation.
    // Hence accept a regularization parameter from the user

    double optimal_reg_param=fx_param_double(NULL,"reg_param",51.2);
    printf("Regularization parameter provided is %f...\n",optimal_reg_param);
    CAKESMO cake_smo;
    cake_smo.Init(train_data,test_data,
		  optimal_reg_param);
    cake_smo.Estimate();
    cake_smo.PrintTestDensities();
  }
  //cake_smo.RegressionTask_();
  fx_timer_stop(NULL,"timer");
  fx_done(NULL);
}

