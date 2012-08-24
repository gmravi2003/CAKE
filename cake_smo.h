#include "fastlib/fastlib.h"
#define SMALL pow(10,-6)
class CAKESMO{
  
 private:

  
  Matrix train_data_;
  Matrix test_data_;


  Matrix Z_mat_;

  Vector v_vector_;

  Matrix pairwise_dist_sqd_mat_;

  Matrix normalization_const_mat_;

  Matrix conv_bw_mat_;

  int num_train_points_;
  int num_test_points_;
  int num_base_kernels_;
  int num_dims_;
  int num_var_;
  int mult_factor_for_bw_;

  Vector beta_up_;
  Vector beta_low_;
  Vector bandwidths_;
 
  
  // Variables involved in the SMO optimization

  ArrayList <ArrayList <int> > I0_indices_;
  ArrayList <ArrayList <int> > I1_indices_;

  
  // Dictates in what mode we need to run SMO.
  bool matrix_free_;

  // The tolerances
  double tau_;

  // The reg parameter.
  double reg_param_;

  // Handy variables. Note all these positions are relative to the row

  int position_of_wsv1_in_I0_;
  int position_of_wsv1_in_I1_;

  int position_of_wsv2_in_I0_;
  int position_of_wsv2_in_I1_;

  // This basically tells us which working set variables (i.e base
  // kernels) we are using.

  int position_of_wsv1_in_current_row_;
  int position_of_wsv2_in_current_row_;

  int global_index_of_wsv1_;
  int global_index_of_wsv2_;
  
  
  // The row that we are working with right now
  
  int current_row_;

  // The F value for elements in I0
  
  ArrayList < ArrayList<double> > F_for_I0_;
  
  double F_wsv1_;
  double F_wsv2_;

  double eps_;

  Vector alpha_vec_;

  // This tells us what is the maximum duality gap
  double max_gap_in_violation_;

  ArrayList <bool> optimality_of_rows_bit_vec_;

  // The i_up and i_low

  ArrayList <int> i_up_;
  ArrayList <int> i_low_;

  // The flag which tells if optimality has been attained over all
  // variables

  bool optimality_attained_flag_;

  Vector train_labels_;
  Vector true_test_labels_;
  Vector estimated_test_densities_;
  Vector true_test_densities_;

  Vector true_test_reg_val_;
  Vector train_reg_val_;

 private:
  void Optimize();
  void UpdateBetaUpAndBetaLow_(double, int ,int);
  void GetRowInZMatrix_(int, Vector &);
  void FillUpZMatrix_();
  void FillUpBandwidthsOfBaseKernels_();
  void FillUpVVector_();
  void FillUpRegularizationParameterVector_();
  void UpdateFValues_(double,double);  
  void UpdateFValuesOfWSV_(double,double);
  void UpdateSetsForWSV1_(double,double,int,int,int,double,int);
  void AddToI0_(int);
  void AddToI1_(int);
  void AddToFForI0_(double);
  void DeleteFromI0_(int);
  void DeleteFromI1_(int);
  void DeleteFromFForI0_(int);
  void UpdateBetaUpAndBetaLow_();
  void UpdateSetsForWSV_(double,double,int *,
			 int *,int,double,int);

  void FormPairwiseDistanceSqdMatrix_();
  void FormNormalizationConstantMatrix_();
  void FormConvolutionBandwidthMatrix_();


  
  void UpdateBetaUpAndBetaLowUsingI0AndWSV_(double,double);
  void OptimizeOverGivenRow_(); 
  void CheckKKTConditions_();
 
  double CalculateFValue_(int, int);
  double CalculatePluginBandwidth_();
  double GetFromZMat_(int,int);
  double CalculateFromScratch_(int,int);
  double GetPrediction_(Vector &);
  double GetRegressionEstimate(Vector &);
  double CalculateKernelContribution_(double *,int);
  double CalculateStandardDeviation_();
  double CalculateElementInZMatrix_(int,int);


  int CheckOptimalityOfBitVector_();
  
  
  int CheckIfInI0_(int,int);
  int CheckIfInI1_(int,int);

 
  bool ExamineAlpha_(int);
  bool CheckForOptimality_(int,double);
  bool TakeStep_();
  


 public:
  
  void Init(Matrix&, Matrix&,double,int,Matrix &,Vector &);
  //void Init(Matrix&, Matrix&,double);
  void Estimate();
  void CalculateTestDensities();
  void ClassificationTask();
  void RegressionTask();
  void PrintTestDensities();
  double get_lscv_on_test_data();
  void get_Z_matrix(Matrix &);
  void get_v_vector(Vector &);
};
