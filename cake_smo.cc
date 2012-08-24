#include "fastlib/fastlib.h"
#include "cake_smo.h"

double CAKESMO::GetPrediction_(Vector &kernel_contrib_vec){
  
  double sum=0;
  for(int i=0;i<num_train_points_;i++){
    
    sum+=kernel_contrib_vec[i]*train_labels_[i];
  }

  return sum>0?+1:-1;
}

double CAKESMO::GetRegressionEstimate(Vector &kernel_contrib_vec){

  double reg_val=0;
  double total_kernel_contrib=0;
  for(int i=0;i<num_train_points_;i++){

    reg_val+=kernel_contrib_vec[i]*train_reg_val_[i];
    total_kernel_contrib+=kernel_contrib_vec[i];
  }
  double small=pow(10,-10);
  if(total_kernel_contrib>small){
    return reg_val/total_kernel_contrib;
  }
  else{

    return 0;
  }
}
void CAKESMO::CalculateTestDensities(){
  
  for(int i=0;i<num_test_points_;i++){

    double *test_point=test_data_.GetColumnPtr(i);
    
    double density=0;
    for(int j=0;j<num_train_points_;j++){
      
      double kernel_contrib=
	CalculateKernelContribution_(test_point,j);

      density+=kernel_contrib;
    }
    estimated_test_densities_[i]=density;
  }

  // Scale the test densities by 1/n

  la::Scale(1.0/num_train_points_,&estimated_test_densities_);
 

  // We will check if true test densities have been provided or not
  // and also we dont want to calculate RMSE when we are still
  // crossvalidating. Hence the second condition is in place.
  
  //printf("length of true densities=%d...\n",true_test_densities_.length());
  //printf("Estimated test densities length=%d...\n",
  // estimated_test_densities_.length());

  if(estimated_test_densities_.length()==true_test_densities_.length()){
    
    // We shall then calculate the RMSE value.
    
    double dist_sqd=
      la::DistanceSqEuclidean (estimated_test_densities_,
			       true_test_densities_); 
    double rmse=sqrt(dist_sqd/num_test_points_);
    printf("RMSE of the estimator is %f..\n",rmse);


    double sum=0;
    // Calculate Hellinger distance
    for(int i=0;i<num_test_points_;i++){
      
      sum+=sqrt(estimated_test_densities_[i]/true_test_densities_[i]); 
    }
    double  hellinger_dist=2-(2*sum/num_test_points_);
    printf("Hellinger distance=%f...\n",hellinger_dist);
  }
}


double CAKESMO::CalculateKernelContribution_(double *test_point,int index){
  
  double *train_point=train_data_.GetColumnPtr(index);
  double sqd_dist=la::DistanceSqEuclidean (num_dims_,train_point,test_point);
  
  double kernel_contrib=0;

  GaussianKernel gk;
  for(int i=0;i<num_base_kernels_;i++){
      
    double alpha_val=alpha_vec_[index*num_base_kernels_+i];
    double bw=bandwidths_[i];
    gk.Init(bw);
    double norm_const=gk.CalcNormConstant(num_dims_);
    double unnorm_val=gk.EvalUnnormOnSq (sqd_dist);
    kernel_contrib+=alpha_val*unnorm_val/norm_const;
  }
  return kernel_contrib;
}

void CAKESMO::ClassificationTask(){
  int num_mistakes=0;
  for(int i=0;i<num_test_points_;i++){

    Vector kernel_contrib_vec ;
    kernel_contrib_vec .Init(num_train_points_);
    
    double *test_point=test_data_.GetColumnPtr(i);
    for(int j=0;j<num_train_points_;j++){
      
      kernel_contrib_vec[j]=
      CalculateKernelContribution_(test_point,j);
    }
    
    double pred_label=GetPrediction_(kernel_contrib_vec);
    if(fabs(pred_label-true_test_labels_[i])>SMALL){

      num_mistakes++;
      // printf("Made mistake....\n");
    }
  }
  double error_rate=(double)num_mistakes/num_test_points_;
  printf("The error rate is %f...\n",error_rate); 

  printf("Finished classification task....\n");
}

void  CAKESMO::RegressionTask(){
  double rmse=0;
  Vector reg_estimate_vec;
  reg_estimate_vec.Init(num_test_points_);
  for(int i=0;i<num_test_points_;i++){

    Vector kernel_contrib_vec ;
    kernel_contrib_vec .Init(num_train_points_);
    
    double *test_point=test_data_.GetColumnPtr(i);
    for(int j=0;j<num_train_points_;j++){
      
      kernel_contrib_vec[j]=
      CalculateKernelContribution_(test_point,j);
    }
    
    double reg_estimate=GetRegressionEstimate(kernel_contrib_vec);
    reg_estimate_vec[i]=reg_estimate;
    rmse+=pow(reg_estimate-true_test_reg_val_[i],2);
    
  }
 
  rmse=rmse/num_test_points_;
  rmse=sqrt(rmse);
  printf("The rmse of regression estimator is %f...\n",rmse); 
  printf("Finished Regression task....\n");

  FILE *fp;
  fp=fopen("cake_reg_estimate.txt","w");
  for(int i=0;i<num_test_points_;i++){

    fprintf(fp,"%f..\n",reg_estimate_vec[i]);
  }
}

void CAKESMO::CheckKKTConditions_(){

  printf("Came to check KKT conditions...\n");
  
  // For each row the beta_up should be larger than beta_low
  
  for(int i=0;i<num_train_points_;i++){
    
    if(beta_up_[i]<beta_low_[i]-2*tau_){
      
      printf("beta_up is smaller than beta low...\n");
      printf("row=%d,beta_up=%f,beta_low=%f....\n",
	     current_row_,beta_up_[i],beta_low_[i]);
      exit(0);
    }
  }

  // Verify satisfaction of KKT conditions for elements in I1....
  for(int i=0;i<num_train_points_;i++){
    
    double beta_up=beta_up_[i];
    for(int j=0;j<I1_indices_[i].size();j++){
      
      double F_val=CalculateFValue_(i,I1_indices_[i][j]);
      if(F_val<beta_up-2*tau_){

	printf("F_val=%f..\n",F_val);
	printf("beta_up=%f...\n",beta_up);

	printf("The element in I1 for row=%d is not optimal...\n",i);
	exit(0);	
      }
    }
  }

  // Go over all rows and see if the F_values of the elements in I0
  // are all equal to each other

 
  for(int i=0;i<num_train_points_;i++){

    double beta_low=beta_low_[i];
    for(int j=0;j<I0_indices_[i].size();j++){

      double F_val=F_for_I0_[i][j];
      if(fabs(F_val-beta_low)>2*tau_){
	
	printf("F_val=%f..\n",F_val);
	printf("beta_low_=%f..\n",beta_low);
	printf("The element in I0 for row=%d is not optimal ...\n",i);
	printf("exiting....\n");
	exit(0);
      }
    }
  }
  //Finally check for feasibility
  
  for(int i=0;i<num_train_points_;i++){
    
    double sum=0;
    for(int j=0;j<num_base_kernels_;j++){

      int index=i*num_base_kernels_+j;
      sum+=alpha_vec_[index];
    }
    if(fabs(sum-1.0)>SMALL){
      
      printf("The sum of elements in row=%d doesn't add to 1..\n",i);
      printf("Infact the sum comes to %f..\n",sum);
    }
  } 
 printf("All KKT conditions are satisfied...\n");
  //printf("alpha vector is ..\n");
  //alpha_vec_.PrintDebug();
}



void CAKESMO::FillUpVVector_(){

  //Initialize first

  v_vector_.Init(num_var_);

  GaussianKernel gk;
 
  //v_vec[i,j]= \sum K_j(x_i-x_l/h_j)

  for(int i=0;i<num_train_points_;i++){

    double *x_i=train_data_.GetColumnPtr(i);

    for(int j=0;j<num_base_kernels_;j++){
      
      int index=i*num_base_kernels_+j;

      double sum=0;
      double bw=bandwidths_[j];
      
      gk.Init(bw);
      double norm_const=gk.CalcNormConstant(num_dims_);
     
      for(int l=0;l<num_train_points_;l++){
	
	double *x_l=train_data_.GetColumnPtr(l);
	
	if(l!=i){
	  
	  double dist_sqd=
	    la::DistanceSqEuclidean (num_dims_,x_i,x_l );
	  
	  sum+=gk.EvalUnnormOnSq(dist_sqd);
	}
      }
      v_vector_[index]=sum/norm_const;
    }
  }
}

/** Need to be filled with proper routine */
void CAKESMO:: FillUpZMatrix_(){
  
  // Initialize first
    Z_mat_.Init(num_var_,num_var_);
    for(int row=0;row<num_var_;row++){
      
      int i=row/num_base_kernels_;
      int j=row%num_base_kernels_;
      double *x_i=train_data_.GetColumnPtr(i);
      double sigma_j=bandwidths_[j];
      for(int col=row;col<num_var_;col++){
	
	int l=col/num_base_kernels_;
	int p=col%num_base_kernels_;
	
	
	double *x_l=train_data_.GetColumnPtr(l);
	double sigma_p=bandwidths_[p];
	
	double conv_bw=
	  sqrt(pow(sigma_j,2)+pow(sigma_p,2));
	
	double dist_sqd=
	  la::DistanceSqEuclidean (num_dims_,x_l,x_i);
	
	GaussianKernel gk;
	gk.Init(conv_bw);
	double kernel_val=gk.EvalUnnormOnSq(dist_sqd);
	double norm_const=gk.CalcNormConstant(num_dims_);
	double val=kernel_val/norm_const;
	if(row!=col){

	  // These are off-diagonal elements
	  Z_mat_.set(row,col,val);
	
	  //Due to symmetry
	  Z_mat_.set(col,row,val);
	  
	  //sum_elem+=2*Z_mat_.get(row,col);
	}
	else{
	 
	  Z_mat_.set(row,col,val+reg_param_);
	  //sum_elem+=Z_mat_.get(row,col);
	}
      }
    }   
}


void CAKESMO::FormPairwiseDistanceSqdMatrix_(){

  printf("NUmber of training points are %d...\n",num_train_points_);
  for(int i=0;i<num_train_points_;i++){

    double *x_i=train_data_.GetColumnPtr(i);
    for(int j=i+1;j<num_train_points_;j++){
    
      double *x_j=train_data_.GetColumnPtr(j);
      
      double dist_sqd=la::DistanceSqEuclidean(num_dims_,x_i,x_j);
      pairwise_dist_sqd_mat_.set(i,j,dist_sqd);
      pairwise_dist_sqd_mat_.set(j,i,dist_sqd);
    }
    pairwise_dist_sqd_mat_.set(i,i,0);
  }
  printf("Formed the pairwise distance matrix....\n");
}

void CAKESMO::FormNormalizationConstantMatrix_(){

  
  for(int i=0;i<num_base_kernels_;i++){

    double bw_i=bandwidths_[i];
    for(int j=i;j<num_base_kernels_;j++){

      double bw_j=bandwidths_[j];
      double conv_bw=sqrt(pow(bw_i,2)+pow(bw_j,2));
      
      GaussianKernel gk;
      gk.Init(conv_bw);

      // Add to convolution bandwidth matrix

      conv_bw_mat_.set(i,j,conv_bw);
      conv_bw_mat_.set(j,i,conv_bw);
      double norm_const=gk.CalcNormConstant(num_dims_);
      normalization_const_mat_.set(i,j,norm_const);
      normalization_const_mat_.set(j,i,norm_const);
    }
  }

  printf("Formed the normalization constant matrix and convoltuion bandwidth matrix...\n");
  conv_bw_mat_.PrintDebug();
}



double CAKESMO::CalculateElementInZMatrix_(int row,int col){
  
  int i=row/num_base_kernels_;
  int j=row%num_base_kernels_;
  
    
  int l=col/num_base_kernels_;
  int p=col%num_base_kernels_;
  
  
  double val;
  GaussianKernel gk;
  double conv_bw=conv_bw_mat_.get(j,p);
  gk.Init(conv_bw);
  double norm_const=normalization_const_mat_.get(j,p);

  if(i!=l){

    double dist_sqd=pairwise_dist_sqd_mat_.get(i,l);   
    double kernel_val=gk.EvalUnnormOnSq(dist_sqd);    
    val=kernel_val/norm_const;
  }
  else{
    
    // In this case we know that the unnormalized kernel value is 1
    
    val=1.0/norm_const;
  }
  if(row==col){
    
    // Dont forget to add the regularization term
    val=val+reg_param_;
  }
  if(fabs(val-Z_mat_.get(row,col))>SMALL){
    
    printf("Calculated wrong value for an element in the Z matrix....\n");
    printf("val=%f...\n",val);
    printf("The element in Z_matrix is %f...\n",Z_mat_.get(row,col));
    exit(0);
  }
  return val;
}


double CAKESMO::GetFromZMat_(int row,int col){
    
    // The matrix is present in main memory hence simply retrieve the
    // value
  
  if(!matrix_free_){
    
    //printf("This is not matrix free...\n");
    // In this case we already have the matrix in memory.
    return Z_mat_.get(row,col);
  }

  else{
    
    // Calculate the element 
    return CalculateElementInZMatrix_(row,col);
  }
    
}


double CAKESMO::CalculateStandardDeviation_(){

  // Calculate variance and return its square-root

  double mean_x_sqd=0;
  double mean_x=0;

  for(int i=0;i<num_train_points_;i++){
    
    double coord=train_data_.get(0,i);
    mean_x_sqd+=pow(coord,2);
    
    mean_x+=coord;
  }
  
  mean_x_sqd/=num_train_points_;
  mean_x/=num_train_points_;
  
  return sqrt(mean_x_sqd-pow(mean_x,2));
}

double CAKESMO::CalculatePluginBandwidth_(){
  
  // Assume dataset is whitened

  double std;
  //printf("number of dimensions is %d...\n",num_dims_);
  if(num_dims_==1){

    std=CalculateStandardDeviation_();    
    //printf("Standard dev=%f..\n",std);
  }
  else{
    std=1.0;
  }

  double bandwidth=pow(4.0/(num_dims_+2),1.0/(num_dims_+4))*
    pow( num_train_points_,-1.0/(num_dims_+4))*std;
  return bandwidth;
}


void CAKESMO::FillUpBandwidthsOfBaseKernels_(){

  double plugin_bandwidth=CalculatePluginBandwidth_();
  
  // First initialize
  bandwidths_.Init(num_base_kernels_);

  
  double min_bandwidth=plugin_bandwidth/3;
  double max_bandwidth=3*plugin_bandwidth;
  double gap=(max_bandwidth-min_bandwidth)/num_base_kernels_;

    
  for(int i=0;i<num_base_kernels_;i++){

    bandwidths_[i]=min_bandwidth+i*gap;
    
  }
}

//Note all positions are relative
int CAKESMO::CheckIfInI0_(int row, int position){

  for(int i=0;i<I0_indices_[row].size();i++){

    if(I0_indices_[row][i]==position){

      return i;
    }
  }
  return -1;
}


int CAKESMO::CheckIfInI1_(int row, int position){

  for(int i=0;i<I1_indices_[row].size();i++){

    if(I1_indices_[row][i]==position){

      return i;
    }
  }
  return -1;
}

void CAKESMO::DeleteFromI0_(int index){
  
  I0_indices_[current_row_].Remove(index);
  
}

void CAKESMO::DeleteFromFForI0_(int index){

  F_for_I0_[current_row_].Remove(index);
}


void CAKESMO::DeleteFromI1_(int index){

  I1_indices_[current_row_].Remove(index);
}

void CAKESMO::AddToI0_(int index){
  
  I0_indices_[current_row_].AddBack(1);
  int size=I0_indices_[current_row_].size();
  I0_indices_[current_row_][size-1]=index;
}

void CAKESMO::AddToI1_(int index){
  
  I1_indices_[current_row_].AddBack(1);
  int size=I1_indices_[current_row_].size();
  I1_indices_[current_row_][size-1]=index;
}

void CAKESMO::AddToFForI0_(double val){
  
  F_for_I0_[current_row_].PushBack(1);
  int size=F_for_I0_[current_row_].size();
  F_for_I0_[current_row_][size-1]=val;
}

void CAKESMO::GetRowInZMatrix_(int global_row, Vector &ret_vec){


  if(!matrix_free_){

    // Since this is not a matrix free setup, hence the Z matrix is already in memory.
    // Due to the symmetry of the matrix, it is enough to just return
    // the column of the matrix
    
    double *ptr= Z_mat_.GetColumnPtr(global_row); 
    ret_vec.Alias(ptr,num_var_);
   
  }
  else{
    
    ret_vec.Init(num_var_);
    
    for(int col=0;col<num_var_;col++){
      
      ret_vec[col]=CalculateElementInZMatrix_(global_row,col);
    }
    
    // Perform a check. This line will be removed
    
    // Vector temp;
//     double *ptr=Z_mat_.GetColumnPtr(global_row);
//     temp.Alias(ptr,num_var_);
//     double dist_sqd=la::DistanceSqEuclidean (ret_vec,temp);
//     if(fabs(dist_sqd)>SMALL){
      
//       printf("The row vector seems incorrect. Distance=%f..........\n",dist_sqd);
//       exit(0);
//     }
  }
}

void CAKESMO::UpdateFValues_(double alpha_wsv1_new,
			     double alpha_wsv2_new){

  for(int i=0;i<num_train_points_;i++){ //Over all rows (points)
    
    for(int j=0;j<I0_indices_[i].size();j++){ //over all columns (base kernels)

      // We are changing F values only for
      // those points in I0

      int pos=I0_indices_[i][j]; 
      int index=i*num_base_kernels_+pos;
      
      double old_contrib=
	2*(GetFromZMat_(index,global_index_of_wsv1_)*
	   alpha_vec_[global_index_of_wsv1_]+
	   GetFromZMat_(index,global_index_of_wsv2_)*
	   alpha_vec_[global_index_of_wsv2_]);

      double new_contrib=
	2*(GetFromZMat_(index,global_index_of_wsv1_)*alpha_wsv1_new+
	   GetFromZMat_(index,global_index_of_wsv2_)*alpha_wsv2_new);
      
      F_for_I0_[i][j]=F_for_I0_[i][j]-old_contrib+new_contrib;
    }
  }
}

void CAKESMO::UpdateFValuesOfWSV_(double alpha_wsv1_new,
			 double alpha_wsv2_new){

  // Update F value of wsv1
  
  double old_contrib=
    2*(GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv1_)*
       alpha_vec_[global_index_of_wsv1_]+
       GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv2_)*
       alpha_vec_[global_index_of_wsv2_]);

  double new_contrib=
    2*(GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv1_)*
       alpha_wsv1_new+
       GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv2_)*
       alpha_wsv2_new);
  
  F_wsv1_=F_wsv1_-old_contrib+new_contrib;
  
  // Update F value of wsv2
  old_contrib=
    2*(GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv1_)*
       alpha_vec_[global_index_of_wsv1_]+
       GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv2_)*
       alpha_vec_[global_index_of_wsv2_]);
  
  new_contrib=
    2*(GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv1_)*
       alpha_wsv1_new+
       GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv2_)*
       alpha_wsv2_new);
  
  F_wsv2_=F_wsv2_-old_contrib+new_contrib;
}

void CAKESMO::UpdateSetsForWSV_(double new_value, 
				double old_value,
				int *position_in_I0,
				int *position_in_I1,
				int position_in_current_row,
				double F_val,
				int which_variable){
  

  int delete_flag=-1;
  // So this element will now be in I1
  if(fabs(new_value)<SMALL){ 

    // This was originally in I0
    if(fabs(old_value)>SMALL){ 

      
      // Delete from I0.
      DeleteFromI0_(*position_in_I0);

      // Also delete from the F values 
      DeleteFromFForI0_(*position_in_I0);
      
      delete_flag=0;

      // Add to I1
      AddToI1_(position_in_current_row);
      
    }
    else{

      // This element was originally in I1, hence do not change
    }
  }
  else{ // This element was now be in I0
    
    if(fabs(old_value)<SMALL){
      // This element was originally in I1.
      // hence add to I0 and delete from I1
      AddToI0_(position_in_current_row);
      AddToFForI0_(F_val);
      DeleteFromI1_(*position_in_I1);
      delete_flag=1;
    }
    else{
      // The variable hasn't changed position so don't do anything
      
    }
  }

  if(which_variable==1){

    if(delete_flag==0){
      // So the variable was deleted from I0.

      if(position_of_wsv2_in_I0_>
	 position_of_wsv1_in_I0_){
	
	position_of_wsv2_in_I0_--;
      }
      
      *position_in_I0=-1; 
      *position_in_I1=I1_indices_[current_row_].size()-1;
    }
    
    if(delete_flag==1){
      // So this variable was deleted from I1.
      if(position_of_wsv2_in_I1_>
	 position_of_wsv1_in_I1_){
	
	position_of_wsv2_in_I1_--;
      }

      *position_in_I1=-1;
      *position_in_I0=I0_indices_[current_row_].size()-1;
    }
  }
  else{ //which variable =2

    if(delete_flag==0){
      if(position_of_wsv1_in_I0_>
	 position_of_wsv2_in_I0_){

	position_of_wsv1_in_I0_--;
      }
      *position_in_I0=-1; 
      *position_in_I1=I1_indices_[current_row_].size()-1;
    }
    if(delete_flag==1){
      
      if(position_of_wsv1_in_I1_>
	 position_of_wsv2_in_I1_){
	
	position_of_wsv1_in_I1_--;
      }

      *position_in_I1=-1;
      *position_in_I0=I0_indices_[current_row_].size()-1;
    }
  }
}

/** This update is over all rows **/
void CAKESMO::UpdateBetaUpAndBetaLowUsingI0AndWSV_(double alpha_wsv1_new,
					     double alpha_wsv2_new){


  //reset beta_up and beta_low everywhere
  
  beta_up_.SetAll(DBL_MAX);
  beta_low_.SetAll(-DBL_MAX);
  
 
  //  First update using I0

  for(int i=0;i<num_train_points_;i++){
    
    if(F_for_I0_[i].size()!=I0_indices_[i].size()){
      
      printf("Size of I0 and F for I0 do not match....\n");
      printf("There is a mistake here5 ...\n");
      exit(0);
    }
    
    for(int j=0;j<I0_indices_[i].size();j++){
     
      // Update beta_low
      if(F_for_I0_[i][j]>beta_low_[i]){
	
	beta_low_[i]=F_for_I0_[i][j];
	i_low_[i]=I0_indices_[i][j];
      }

      // Update beta up
      if(F_for_I0_[i][j]<beta_up_[i]){
	
	beta_up_[i]=F_for_I0_[i][j];
	i_up_[i]=I0_indices_[i][j];
      }
    }
  }
  
  // Update using wsv1

  if(fabs(alpha_wsv1_new)>SMALL){
    
    // This belongs to I0
   
    // Update beta_low
    if(F_wsv1_>beta_low_[current_row_]){

      beta_low_[current_row_]=F_wsv1_;
      i_low_[current_row_]=
	position_of_wsv1_in_current_row_;

      // printf("Updating beta low...\n");
      //printf("position_of_wsv1_in_current_row_=%d..\n",
      //     position_of_wsv1_in_current_row_);
    }

    //Update beta_up

    if(F_wsv1_<beta_up_[current_row_]){

      beta_up_[current_row_]=F_wsv1_;
      i_up_[current_row_]=
	position_of_wsv1_in_current_row_;
      
      //printf("Updating beta low...\n");
      //printf("position_of_wsv1_in_current_row_=%d..\n",
      //     position_of_wsv1_in_current_row_);
    }
  }
  
  else{
    // It belongs to I1
      
    if(F_wsv1_<beta_up_[current_row_]){
	
	beta_up_[current_row_]=F_wsv1_;
	i_up_[current_row_]=
	  position_of_wsv1_in_current_row_;
      }
  }

 
  // Update using wsv2
  if(fabs(alpha_wsv2_new)>SMALL){
    
    // This belongs to I0
    
    // Check F_wsv2_

    //Update beta_low
    if(F_wsv2_>beta_low_[current_row_]){
      
      beta_low_[current_row_]=F_wsv2_;
      i_low_[current_row_]=
	position_of_wsv2_in_current_row_;
    }

    // Update beta_up
    if(F_wsv2_<beta_up_[current_row_]){
      
      beta_up_[current_row_]=F_wsv2_;
      i_up_[current_row_]=
	position_of_wsv2_in_current_row_;
    }
  }
  else{
    // It belongs to I1
      if(F_wsv2_<beta_up_[current_row_]){
	
	beta_up_[current_row_]=F_wsv2_;
	i_up_[current_row_]=
	  position_of_wsv2_in_current_row_;
      }
  }
}


bool CAKESMO::TakeStep_(){
  

  if(global_index_of_wsv1_==global_index_of_wsv2_){
    
    return 0;
  }

  double sum=0.0;
  int base_index=current_row_*num_base_kernels_;
  for(int j=0;j<num_base_kernels_;j++){
    
    if(j!=position_of_wsv2_in_current_row_&&
       j!=position_of_wsv1_in_current_row_){
      
      
      sum+=alpha_vec_[base_index+j];
    }
  }

  double gamma=1-sum;
  
  // This is ij1
  
  double c_j1=
    F_wsv1_-2*(GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv1_)*
	       alpha_vec_[global_index_of_wsv1_]+
	       GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv2_)*
	       alpha_vec_[global_index_of_wsv2_]);
  
  
  double c_j2=
    F_wsv2_-2*(GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv1_)*
	       alpha_vec_[global_index_of_wsv1_]+
	       GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv2_)*
	       alpha_vec_[global_index_of_wsv2_]);
  
  double psi=
    c_j2-c_j1+2*gamma*(GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv2_)-
		       GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv2_));

  double xi=2*(GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv1_)+
	       GetFromZMat_(global_index_of_wsv2_,global_index_of_wsv2_)-
	       2*GetFromZMat_(global_index_of_wsv1_,global_index_of_wsv2_));
  
  double L=0;
  double H=gamma;

  double alpha_wsv1_new,alpha_wsv2_new;
  if(fabs(xi)>SMALL){
    
    // xi \neq 0
    
    alpha_wsv1_new=min(max(0.0,psi/xi),gamma);
  }
  else{
    
    if(psi>0){

      alpha_wsv1_new=H;
    }
    else{
      
      alpha_wsv1_new=L;
    }
  }
  alpha_wsv2_new=gamma-alpha_wsv1_new;

  if(fabs(alpha_wsv2_new-alpha_vec_[global_index_of_wsv2_]) < 
     eps_*(alpha_wsv2_new+alpha_vec_[global_index_of_wsv2_]+eps_)){


    return 0;
  }
  // Update F values of elements in I0 and the wsv
  UpdateFValues_(alpha_wsv1_new,alpha_wsv2_new);  
  UpdateFValuesOfWSV_(alpha_wsv1_new,alpha_wsv2_new);

  // Update sets for the different wsv
  double new_value=alpha_wsv1_new;
  double old_value=alpha_vec_[global_index_of_wsv1_];

  
  UpdateSetsForWSV_(new_value,old_value,
		    &position_of_wsv1_in_I0_,
		    &position_of_wsv1_in_I1_,
		    position_of_wsv1_in_current_row_,
		    F_wsv1_,1);
  
  new_value=alpha_wsv2_new;
  old_value=alpha_vec_[global_index_of_wsv2_];

  UpdateSetsForWSV_(new_value,old_value,
		    &position_of_wsv2_in_I0_,
		    &position_of_wsv2_in_I1_,
		    position_of_wsv2_in_current_row_,
		    F_wsv2_,2);
 
  UpdateBetaUpAndBetaLowUsingI0AndWSV_(alpha_wsv1_new,alpha_wsv2_new);

 
  alpha_vec_[global_index_of_wsv1_]=alpha_wsv1_new;
  alpha_vec_[global_index_of_wsv2_]=alpha_wsv2_new;

  // Note since alpha values have changed the optimality might have
  // been disrupted
  
  optimality_attained_flag_=false;
  return 1;
}



void CAKESMO::UpdateBetaUpAndBetaLow_(double F_value,int row,int position){
  
  if(position_of_wsv2_in_I1_!=-1 &&
     F_value< beta_up_[row]){
    
    beta_up_[row]=F_value;
    i_up_[row]=position;
  }

  if(position_of_wsv2_in_I0_!=-1 &&
     F_value> beta_low_[row]){
    
    beta_low_[row]=F_value;
    i_low_[row]=position;
  }

  if(position_of_wsv2_in_I0_!=-1 &&
     F_value< beta_up_[row]){
    
    beta_up_[row]=F_value;
    i_up_[row]=position;
  }
}

bool CAKESMO::CheckForOptimality_(int position, double F_value){
  
  bool optimality=true;
  if(position_of_wsv2_in_I0_!=-1){
    
    if(F_value>beta_up_[current_row_]+2*tau_){

      
      optimality=false;
    }
    if(F_value<beta_low_[current_row_]-2*tau_){
      
      optimality=false;
    }

    if(optimality==true){

      return 0;
    }
    else{

      if(F_value-beta_up_[current_row_]>beta_low_[current_row_]-F_value){
	// There is violation. Hence take 
	// wsv1 to be i_up
	
	position_of_wsv1_in_current_row_=i_up_[current_row_];
	position_of_wsv1_in_I0_=CheckIfInI0_(current_row_,i_up_[current_row_]);
	
	if(position_of_wsv1_in_I0_==-1){
	  
	  position_of_wsv1_in_I1_=
	    CheckIfInI1_(current_row_,i_up_[current_row_]);
	  
	  if(position_of_wsv1_in_I1_==-1){
	    
	    printf("There is a mistake here2...\n");
	    exit(0); 
	  }
	}
	else{

	  // The element has been found to be in I0. 
	  // Hence set its position in I1 to be -1

	  position_of_wsv1_in_I1_=-1;  
	}

	F_wsv1_=beta_up_[current_row_];
	optimality=false;
      }
      else{

	// There is violation. hence take wsv1 to be i_low
	
	position_of_wsv1_in_current_row_=i_low_[current_row_];
	position_of_wsv1_in_I0_=
	  CheckIfInI0_(current_row_,i_low_[current_row_]);

	if(position_of_wsv1_in_I0_==-1){
	  
	  position_of_wsv1_in_I1_=
	    CheckIfInI1_(current_row_,i_low_[current_row_]);
	  
	  if(position_of_wsv1_in_I1_!=-1){
	    
	    printf("There is a mistake here2. This element is in I1 but should have been I0...\n");
	    exit(0);
	    
	  }
	  else{

	    printf("There is a mistake here10. This element is nowhere.....\n");
	    exit(0);
	  }
	}
	else{
	  
	  // We are fine. Since this element is in I0 set its 
	  // position in I1 to be -1
	  position_of_wsv1_in_I1_=-1;
	}
	F_wsv1_=beta_low_[current_row_];
	optimality=false;
      }
    }
  }
  else{
    
    // The element belongs to I1
    if(F_value<beta_low_[current_row_]-2*tau_){

      position_of_wsv1_in_current_row_=i_low_[current_row_];
      
      position_of_wsv1_in_I0_=
	CheckIfInI0_(current_row_,i_low_[current_row_]);

      if(position_of_wsv1_in_I0_==-1){
      
	printf("current row is %d..\n",current_row_);
	printf("i_low is %d..\n",i_low_[current_row_]);
	printf("There is a mistake here3...\n");
	printf("position in I1 is %d..\n",CheckIfInI1_(current_row_,i_low_[current_row_]));
	exit(0);
      }
      else{

	// Remember to set the position of wsv1 in I1 to -1
	position_of_wsv1_in_I1_=-1;
      }
      
      // Also cache the F values
      F_wsv1_=beta_low_[current_row_];     
      optimality=false;
    }
    else{

      // This element is optimal
      return 0;
    }
  }
  global_index_of_wsv1_=
    current_row_*num_base_kernels_+position_of_wsv1_in_current_row_;
  // Now TakeStep

  return TakeStep_();
}

/** This examines the given alpha to find its set membership**/
/** The input is position relative to the current row that 
    is being investigated **/

bool CAKESMO::ExamineAlpha_(int position){

  // The first thing to do is to check if this element is in I0

  position_of_wsv2_in_I0_=
    CheckIfInI0_(current_row_,position);

  double F_value;

  if(position_of_wsv2_in_I0_!=-1){

    // This means this element is in I0.
    F_value=
      F_for_I0_[current_row_][position_of_wsv2_in_I0_];

    position_of_wsv2_in_I1_=-1;
    F_wsv2_=F_value;
  }

  else{
    // So this element is not in I0. Hence calculate its F-value from
    // scratch.
    
    F_value=CalculateFValue_(current_row_,position);
    F_wsv2_=F_value;
    position_of_wsv2_in_I0_=-1;
    position_of_wsv2_in_I1_=CheckIfInI1_(current_row_,position);    

    if(position_of_wsv2_in_I1_==-1){
      
      printf("There is a mistake here4. Hence exiting.....\n");
      exit(0);
    }
  }
  // Now lets update beta_up and beta_low
  UpdateBetaUpAndBetaLow_(F_value,current_row_,position); 
  bool ret_val=CheckForOptimality_(position,F_value);
  return ret_val;
}

/* From here onwards it is the usual SMO, i.e we are given a set of
working set variables (i.e alpha's corresponding to a particular
training point and the task is to choose 2 appropriate wsv.*/

void CAKESMO::OptimizeOverGivenRow_(){
  
  int num_iterations=0;
  int MAX_NUM_ITERATIONS_FOR_GIVEN_ROW=(int)pow(10,4);
  int num_changed=1;
  bool examine_all_alphas=true;
  while((num_changed!=0|| examine_all_alphas)&&
	num_iterations<MAX_NUM_ITERATIONS_FOR_GIVEN_ROW){
    
    num_changed=0;
    if(examine_all_alphas){
      
      for(int i=0;i<num_base_kernels_;i++){
	
	int position=i;
	position_of_wsv2_in_current_row_=position;

	global_index_of_wsv2_=
	  current_row_*num_base_kernels_+position;

	num_changed+=ExamineAlpha_(position);
      }
    }
    else{

      // Just iterate over those alpha's that are in I0
      for(int i=0;i<I0_indices_[current_row_].size();i++){

	int position=I0_indices_[current_row_][i];
	position_of_wsv2_in_current_row_=position;
	position_of_wsv2_in_I0_=i;
	position_of_wsv2_in_I1_=-1;

	global_index_of_wsv2_=
	  current_row_*num_base_kernels_+position;

	num_changed+=ExamineAlpha_(position);
      }
    }
    if(examine_all_alphas==true){
      examine_all_alphas=false;

    }
    else{
      
      if(num_changed==0){

	examine_all_alphas=true;
      }
    }
    num_iterations++;
  }
}

double CAKESMO::CalculateFValue_(int row, int col){

  // The first thing to be able to do is calculate Z\alpha. 

  int global_index=row*num_base_kernels_+col;
  Vector vec;
  GetRowInZMatrix_(global_index,vec);
  
  // Get the dot product between this and the alpha vector  
  if(vec.length()!=alpha_vec_.length()){
    
    printf("Sizes dont match...\n");
    vec.PrintDebug();
  }
  double Zalpha= la::Dot(vec,alpha_vec_);
  return 2*Zalpha-2*v_vector_[global_index];
}

/* The idea is to pick a row to optimize over and pick working set
variables from this row until optimality is reached or max number of
iterations are reached. Once this is done move over to the next column
and repeat this process. This is done until all rows are optimal.*/


void CAKESMO::Optimize(){
  
  // Maintain a flag to know if optimality has been reached in all
  // rows.
  int MAX_NUM_MASTER_ITERATIONS=(int)pow(10,6);
  int num_iterations=0;
  optimality_attained_flag_=false;
  while(!optimality_attained_flag_&&
	num_iterations<MAX_NUM_MASTER_ITERATIONS){
    
    // First pick the row, from where the working set variables will
    // come.

    // Reset the optimality attained flag to true. If in case this
    // flag reverts back to false, then we will have to iterate again
    // over all the rows.

    optimality_attained_flag_=true;
    for(current_row_=0;current_row_<num_train_points_;current_row_++){
      
      // Optimize over this row. flag tells us if optimality is
      // attained within that column.
      
	OptimizeOverGivenRow_();
    }
    num_iterations++;
  }

 
  // Finally check if the KKT conditions were satisfied or not
  //CheckKKTConditions_();
}

double CAKESMO::get_lscv_on_test_data(){

  // quad part=\alpha^TZ\alpha

  Vector Z_trans_alpha;
  la::MulInit (Z_mat_,alpha_vec_,&Z_trans_alpha);
  double quad_part=la::Dot(Z_trans_alpha,alpha_vec_);
  quad_part-=(reg_param_*pow(la::LengthEuclidean(alpha_vec_),2));
  quad_part/=pow(num_train_points_,2);
  
  // For the linear part simply calculate the mean of distances
  
 
  double sum_of_densities=0;
  for(int i=0;i<num_test_points_;i++){

    sum_of_densities+=estimated_test_densities_[i];
  }
  
  double linear_part=2*sum_of_densities/(num_test_points_);
  double lscv_score=quad_part-linear_part;
  return lscv_score;
}

void CAKESMO::get_Z_matrix(Matrix &mat_in){

  mat_in.Copy(Z_mat_);

}

void CAKESMO::get_v_vector(Vector &vec_in){

  vec_in.Copy(v_vector_);
}


void CAKESMO::Estimate(){
  
  // First optimize
  Optimize();
  CalculateTestDensities();
}

void CAKESMO::PrintTestDensities(){

   FILE *fp;
   fp=fopen("qp_and_boosted_kde/old_faithful_multi_qp_kde/cake_old_faithful_multi_boxed.txt","w");
   for(int i=0;i<num_test_points_;i++){
     
     fprintf(fp,"%f\n",estimated_test_densities_[i]);
   }
   printf("The first 5 estimated test densities are ...\n");
   printf("%f,%f,%f,%f,%f...\n",
	  estimated_test_densities_[0],
	  estimated_test_densities_[1],
	  estimated_test_densities_[2],
	  estimated_test_densities_[3],
	  estimated_test_densities_[4]);
  
}

void CAKESMO::Init(Matrix &train_data,Matrix &test_data,double reg_param,int params_available, Matrix &in_mat, Vector &in_vec){


  matrix_free_=false;

  // First copy the data
  train_data_.Alias(train_data);
  test_data_.Alias(test_data);

  reg_param_=reg_param;
  num_base_kernels_=
    fx_param_int(NULL,"num_base_kernels",10);

  tau_=pow(10,-2);
  eps_=pow(10,-10);
  mult_factor_for_bw_=3;
  
  num_train_points_=train_data.n_cols();
  num_test_points_=test_data.n_cols();

  num_dims_=train_data_.n_rows();
  num_var_=num_base_kernels_*num_train_points_;

  estimated_test_densities_.Init(num_test_points_);
  num_dims_=train_data_.n_rows();
  //printf("number of dimensions is %d..\n",num_dims_);

  //  printf("Number of test points are %d...\n",num_test_points_);
  
  // Also read if the true densities are present

  bool flag=
    fx_param_bool(NULL,"true_test_densities",false);

  if(flag){

    const char *true_densities_file=
      fx_param_str_req(NULL,"true_test_densities_file");

    Matrix true_test_densities_mat;
    data::Load(true_densities_file,&true_test_densities_mat);
    
    true_test_densities_.Init(num_test_points_);
  
    //Convert this into a vector

    for(int i=0;i<num_test_points_;i++){

      true_test_densities_[i]=
	true_test_densities_mat.get(0,i);
    }
  }
  else{
    
    true_test_densities_.Init(0);
  }

  estimated_test_densities_.Init(num_test_points_);
  flag=fx_param_bool(NULL,"classification",false);
  if(flag){

    const char *test_labels_file=
      fx_param_str_req(NULL,"test_labels_file");

    Matrix test_labels_mat;
    data::Load(test_labels_file,&test_labels_mat);
    true_test_labels_.Init(num_test_points_);
    
    //printf("size of test labels is %d...\n",test_labels_mat.n_cols());
    for(int i=0;i<num_test_points_;i++){
      
      true_test_labels_[i]=test_labels_mat.get(0,i);
    }

    const char *train_labels_file=
      fx_param_str_req(NULL,"train_labels_file");

    Matrix train_labels_mat;
    data::Load(train_labels_file,&train_labels_mat);

    train_labels_.Init(num_train_points_);
    for(int i=0;i<num_train_points_;i++){

      train_labels_[i]=train_labels_mat.get(0,i);
    }
  }

  else{

    true_test_labels_.Init(0);
    train_labels_.Init(0);
  }
  

  flag=fx_param_bool(NULL,"regression",false);

  if(flag){

    const char* test_reg_values_file=
      fx_param_str_req(NULL,"test_reg_values_file");

    Matrix test_reg_val_mat,train_reg_val_mat;
    data::Load(test_reg_values_file,&test_reg_val_mat);

   
    true_test_reg_val_.Init(num_test_points_);
    for(int i=0;i<num_test_points_;i++){

      true_test_reg_val_[i]=test_reg_val_mat.get(0,i);
    }
    
    const char *train_reg_values_file=
      fx_param_str_req(NULL,"train_reg_values_file");
    
    data::Load(train_reg_values_file,&train_reg_val_mat);
    train_reg_val_.Init(num_train_points_);
    for(int i=0;i<num_train_points_;i++){

      train_reg_val_[i]=train_reg_val_mat.get(0,i);
    }
  }
  else{

    train_reg_val_.Init(0);
    true_test_reg_val_.Init(0);

  }

  
  // Fill up the Z matrix and the v vector
  FillUpBandwidthsOfBaseKernels_();

  if(params_available==0){

    FillUpZMatrix_();
    FillUpVVector_();
  }
  else{
    //Z and v are available

    
    // Z_mat_.Init(num_var_,num_var_);
    //v_vector_.Init(num_var_);
    Z_mat_.Alias(in_mat);
    v_vector_.Alias(in_vec);
  }
 
  if(matrix_free_){
    
    // In this case we need to calculate paiwise distance between
    // points
    
    pairwise_dist_sqd_mat_.Init(num_train_points_,num_train_points_);
    normalization_const_mat_.Init(num_base_kernels_,num_base_kernels_);
    conv_bw_mat_.Init(num_base_kernels_,num_base_kernels_);

    FormPairwiseDistanceSqdMatrix_();

    // This routine fills up the normalization constant matrix and
    // also the convolution bandwidth matrix
    FormNormalizationConstantMatrix_();    
  }

  else{
    
    // Since we are not doing matrix free operations, hence initialize
    // matrices to empty ones
    pairwise_dist_sqd_mat_.Init(1,1);
    normalization_const_mat_.Init(1,1);
    conv_bw_mat_.Init(1,1);
  }

  // Let us first initialize alpha vector
  alpha_vec_.Init(num_var_);
  alpha_vec_.SetZero();
  
  for(int i=0;i<num_train_points_;i++){
    
    int index=i*num_base_kernels_;
    alpha_vec_[index]=1.0;
  }

  // Initialize I0 and I1
  
  I0_indices_.Init(num_train_points_);
  I1_indices_.Init(num_train_points_);

  for (int i=0;i<num_train_points_;i++){
    
    I0_indices_[i].Init(1);
    I1_indices_[i].Init(num_base_kernels_-1);
  }

  // Now fill up I0 and I1

  for (int i=0;i<num_train_points_;i++){
    
    I0_indices_[i][0]=0;
    for(int j=1;j<num_base_kernels_;j++){
      
      I1_indices_[i][j-1]=j;
    }
  } 

  //printf("filled up I0 and I1....\n");

  // Fill up F values for elements in I0
  
  F_for_I0_.Init(num_train_points_);

  for(int i=0;i<num_train_points_;i++){

    int row=i;
    int col=0;
    F_for_I0_[i].Init(1);   
    F_for_I0_[i][0]=CalculateFValue_(row,col);
  }

  beta_up_.Init(num_train_points_);
  beta_low_.Init(num_train_points_);

  beta_up_.SetAll(DBL_MAX);
  beta_low_.SetAll(-DBL_MAX);

  // Allocate memory for i_up and i_low
  i_up_.Init(num_train_points_);
  i_low_.Init(num_train_points_);

  // Initialize optimalty of rows bit vector to all 0's

  optimality_of_rows_bit_vec_.Init(num_train_points_);
  for(int i=0;i<num_train_points_;i++){

    optimality_of_rows_bit_vec_[i]=0;
  }
  
}
