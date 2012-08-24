#include "fastlib/fastlib.h"
#include "cake_smo.h"

void UpdateZMatrix(Matrix &Z_mat,double val_to_add_to_diag){

 
  int num_rows=Z_mat.n_rows();
  for(int i=0;i<num_rows;i++){
    double prev_val=Z_mat.get(i,i);
    Z_mat.set(i,i,prev_val+val_to_add_to_diag);
  }
}

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

  fx_param_bool(NULL,"matrix_free",false);

  fx_timer_start(NULL,"timer");

  // Before we even begin lets form the vector v and matrix Z
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
   
    for(int i=0;i<reg_param_len;i++){
      
      reg_param_vec[i]=pow(5,-12+2*i)*pow(num_train_points,2);
      //reg_param_vec[i]=pow(10,-2+i);
    }
    printf("The regularization vector is ...\n");
    reg_param_vec.PrintDebug();
    
    Dataset dset_train;
    dset_train.matrix().Alias(train_data);
    
    
    Vector lscv_vec;
    lscv_vec.Init(reg_param_len);

    for(int i=0;i<lscv_vec.length();i++){
      
      lscv_vec[i]=0;
    }

    void *random_permutation;
    
    random_permutation=(void *)malloc(num_train_points*sizeof(index_t));
    math::MakeIdentityPermutation(num_train_points,
				  (index_t *) random_permutation);
    
    
    // Lets typecast random_permutation as ArrayList <index_t>
    
    ArrayList <int> random_permutation_array_list;
    random_permutation_array_list.InitAlias((int *)random_permutation, 
					    num_train_points);
    
    for(int fold_num=0;fold_num<num_folds;fold_num++){
      
      // They will be initialized during the first call.
      
      
      
      Dataset cv_train_data,cv_test_data;
      GetTheFold(cv_train_data,cv_test_data,
		 fold_num,num_folds,
		 random_permutation_array_list,dset_train);

      // Now go over all the possible reg parameter values      
      // Dummy variables
      Matrix dummy_mat;
      Vector dummy_vec;
      dummy_mat.Init(0,0);
      dummy_vec.Init(0);

      Matrix Z_mat;
      Vector v_vec;

      for(int param_num=0;param_num<reg_param_len;param_num++){
	
	double reg_param=reg_param_vec[param_num];

	
	CAKESMO cake_smo;

	double lscv;
	if(param_num==0){

	  // Because this is the first time we let the code calculate
	  // the Z matrix and the v vector
	  
	  cake_smo.Init(cv_train_data.matrix(),cv_test_data.matrix(),
			reg_param,0,dummy_mat,dummy_vec);
	  cake_smo.Estimate();
	  lscv=cake_smo.get_lscv_on_test_data();
	  cake_smo.get_Z_matrix(Z_mat);
	  cake_smo.get_v_vector(v_vec);
	}
	else{
	  // First update the Z matrix
	  double val_to_add_to_diag=
	    reg_param_vec[param_num]-reg_param_vec[param_num-1];
	  
	  UpdateZMatrix(Z_mat,val_to_add_to_diag);
	
	  
	  cake_smo.Init(cv_train_data.matrix(),cv_test_data.matrix(),
			reg_param,1,Z_mat,v_vec);

	  cake_smo.Estimate();
	  lscv=cake_smo.get_lscv_on_test_data();
	}
	lscv_vec[param_num]+=lscv;
      }
    }
    
    // Scale lscv_vec by 1/num_folds
    la::ScaleOverwrite(1.0/num_folds,lscv_vec,&lscv_vec);

    for(int i=0;i<reg_param_len;i++){
      
      printf("reg_param=%f, LSCV=%f...\n",reg_param_vec[i],lscv_vec[i]);

    }
    // We have finished crossvalidation
    double opt_lscv_val=DBL_MAX;
    double opt_reg_val;
    for(int i=0;i<reg_param_len;i++){
      
      if(lscv_vec[i]<opt_lscv_val){
	
	opt_lscv_val=lscv_vec[i];
	opt_reg_val=reg_param_vec[i];
      }
    }

    printf("Optimal reg param is %f...\n",opt_reg_val);
    printf("Optimal LSCV value is %f...\n",opt_lscv_val);

    // Finally run with optimal parameter
    fx_set_param_bool(NULL,"true_test_densities",true_test_densities_file);
    fx_set_param_bool(NULL,"regression",regression);
    fx_set_param_bool(NULL,"classification",classification);

    CAKESMO cake_smo;    
    Matrix dummy_mat;
    Vector dummy_vec;
    dummy_mat.Init(0,0);
    dummy_vec.Init(0);
    cake_smo.Init(train_data,test_data,opt_reg_val,0,dummy_mat,dummy_vec);
    cake_smo.Estimate();
    //cake_smo.ClassificationTask();
    //cake_smo.RegressionTask();
    cake_smo.PrintTestDensities();
  }
  else{
    // This means we are not doing any crossvalidation.
    // Hence accept a regularization parameter from the user

    double optimal_reg_param=fx_param_double(NULL,"reg_param",0.010486);
    printf("Regularization parameter provided is %f...\n",optimal_reg_param);
    CAKESMO cake_smo;
   
    // Some dummy variables
    Matrix dummy_mat;
    Vector dummy_vec;
    dummy_mat.Init(0,0);
    dummy_vec.Init(0);

    cake_smo.Init(train_data,test_data,
		  optimal_reg_param,0,dummy_mat,dummy_vec);
    cake_smo.Estimate();
    cake_smo.PrintTestDensities();
  }
  //cake_smo.RegressionTask_();
  fx_timer_stop(NULL,"timer");
  fx_done(NULL);
}

