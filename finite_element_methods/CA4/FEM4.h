
//Include files
//Data structures and solvers
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

//Mesh related classes
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

//Finite element implementation classes
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

//Standard C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <filesystem>
using namespace dealii;


const unsigned int order = 1;
const unsigned int quadRule = 2;

template <int dim>
class FEM
{
 public:
  //Class functions
  FEM (double Alpha, double DeltaT);  // Class constructor 
  ~FEM();                             //Class destructor

  //Solution steps
  void generate_mesh(std::vector<unsigned int> numberOfElements);
  void define_boundary_conds();
  void setup_system();
  void assemble_system();

  void solve_steady();
  void apply_initial_conditions();
  void solve_trans();
  void output_steady_results();
  void output_trans_results(unsigned int index);

  //Calculate the l2norm of the difference between the steady state and transient solution
  double l2norm();

	void setSavePath(); // Get path to save files to
	void writeL2VectorToCSV();

  //Class objects
  Triangulation<dim> triangulation;       
  FESystem<dim>      fe;                  
  DoFHandler<dim>    dof_handler;        
  QGauss<dim>  	     quadrature_formula;  

  //Data structures
  SparsityPattern      sparsity_pattern;                    
  SparseMatrix<double> M, K, system_matrix;               
  Vector<double>       D_steady, D_trans, V_trans, F, RHS;  

  Table<2,double>	              nodeLocation;         
  std::map<unsigned int,double> boundary_values_of_D; 
  std::map<unsigned int,double> boundary_values_of_V; 
	std::vector<std::vector<double>> l2norm_results;    

  double	alpha;    // Specifies the Euler method, 0 <= alpha <= 1
  double	delta_t;  // Specifies the timestep

  std::string savePath;       
  std::string alphaAsString;  
  std::string deltaAsString;  
  char sep;                   

  //Solution name array
  std::vector<std::string> nodal_solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
};

// Class constructor for a scalar field
template <int dim>
FEM<dim>::FEM (double Alpha, double DeltaT)
:
fe (FE_Q<dim>(order), 1),
  dof_handler (triangulation),
  quadrature_formula(quadRule)
{
  alpha = Alpha;
	delta_t = DeltaT;
	
	setSavePath();

  nodal_solution_names.push_back("D");
  nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
}

//Class destructor
template <int dim>
FEM<dim>::~FEM (){dof_handler.clear ();}

//Generate the mesh
template <int dim>
void FEM<dim>::generate_mesh(std::vector<unsigned int> numberOfElements){

  double  x_min =0. , 
          x_max =1., 
          y_min =0., 
          y_max =1., 
          z_min =0., 
          z_max =0.1; 

  Point<dim,double> min(x_min,y_min,z_min),
                      max(x_max,y_max,z_max);
  GridGenerator::subdivided_hyper_rectangle (triangulation, numberOfElements, min, max);
}


template <int dim>
void FEM<dim>::define_boundary_conds(){
  const unsigned int totalNodes = dof_handler.n_dofs(); 
  
  for (uint i = 0; i < totalNodes ; ++i)
  {
      // Left Dirichlet BC
      if (nodeLocation[i][0] == 0.0)
      {
            boundary_values_of_D[i] = 300.;
            boundary_values_of_V[i] = 0.;
      }
      // Right Dirichlet BC
      if (nodeLocation[i][0] == 1.)
      {
            boundary_values_of_D[i] = 310.;
            boundary_values_of_V[i] = 0.;
      }
    
  }
}

//Setup data structures (sparse matrix, vectors)
template <int dim>
void FEM<dim>::setup_system(){


  dof_handler.distribute_dofs (fe);

 
  MappingQ1<dim,dim> mapping;
  std::vector< Point<dim,double> > dof_coords(dof_handler.n_dofs());
  nodeLocation.reinit(dof_handler.n_dofs(),dim);
  DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords);
  for(unsigned int i=0; i<dof_coords.size(); i++){
    for(unsigned int j=0; j<dim; j++){
      nodeLocation[i][j] = dof_coords[i][j];
    }
  }
  
  define_boundary_conds();

  sparsity_pattern.reinit (dof_handler.n_dofs(),
			   dof_handler.n_dofs(),
			   dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  sparsity_pattern.compress();
  K.reinit (sparsity_pattern);
  M.reinit (sparsity_pattern);
  system_matrix.reinit (sparsity_pattern);
  D_steady.reinit(dof_handler.n_dofs());
  D_trans.reinit(dof_handler.n_dofs());
  V_trans.reinit(dof_handler.n_dofs());
  RHS.reinit(dof_handler.n_dofs());
  F.reinit(dof_handler.n_dofs());

  //Just some notes...
  std::cout << "   Number of active elems:       " << triangulation.n_active_cells() << std::endl;
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;   
}


template <int dim>
void FEM<dim>::assemble_system(){

  M=0; K=0; F=0;

  FEValues<dim> fe_values(fe,
			  quadrature_formula,
			  update_values | 
			  update_gradients | 
			  update_JxW_values);

  const unsigned int dofs_per_elem = fe.dofs_per_cell;        
  unsigned int 	     num_quad_pts = quadrature_formula.size(); 
  FullMatrix<double> Mlocal (dofs_per_elem, dofs_per_elem);
  FullMatrix<double> Klocal (dofs_per_elem, dofs_per_elem);
  Vector<double>     Flocal (dofs_per_elem);

  std::vector<unsigned int> local_dof_indices (dofs_per_elem); 
  double		    rho = 3.8151 * pow(10,6);                           

  //loop over elements  
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active (),
    endc = dof_handler.end();
  for (;elem!=endc; ++elem){
    elem->get_dof_indices (local_dof_indices);
    fe_values.reinit(elem); 
    elem->get_dof_indices (local_dof_indices);

    //Loop over local DOFs and quadrature points for Mlocal
    Mlocal = 0.;
    for(unsigned int q=0; q<num_quad_pts; q++){
      for(unsigned int A=0; A<fe.dofs_per_cell; A++){
        for(unsigned int B=0; B<fe.dofs_per_cell; B++){
          Mlocal(A,B) += rho * fe_values.shape_value(A, q) * fe_values.shape_value(B, q) * fe_values.JxW(q);
        }
      }
    }

    FullMatrix<double> kappa(dim,dim);
    kappa[0][0] = 385.;
    kappa[1][1] = 385.;
    kappa[2][2] = 385.;

    //Loop over local DOFs and quadrature points for Klocal
    Klocal = 0.;
    for(unsigned int A=0; A<fe.dofs_per_cell; A++){
      for(unsigned int B=0; B<fe.dofs_per_cell; B++){
        for(unsigned int q=0; q<num_quad_pts; q++){
          for(unsigned int i=0; i<dim; i++){
            for(unsigned int j=0; j<dim; j++){
              Klocal(A, B) += kappa[i][j] * fe_values.shape_grad(A, q)[i] * fe_values.shape_grad(B, q)[j] * fe_values.JxW(q);
            }
          }
        }
      }
    }

    for (unsigned int i=0; i<dofs_per_elem; ++i){
      for (unsigned int j=0; j<dofs_per_elem; ++j){
        M.add(local_dof_indices[i], local_dof_indices[j],Mlocal(i,j)); 
        K.add(local_dof_indices[i], local_dof_indices[j],Klocal(i,j)); 
      }
    }
  }
}

//Template for steady solution using D=K^-1F
template <int dim>
void FEM<dim>::solve_steady(){

  MatrixTools::apply_boundary_values (boundary_values_of_D, K, D_steady, F, false);
  
  SparseDirectUMFPACK  A;
  A.initialize(K);
  A.vmult (D_steady, F); 

  output_steady_results();
}

//Apply initial conditions for the transient problem
template <int dim>
void FEM<dim>::apply_initial_conditions(){

  const unsigned int totalNodes = dof_handler.n_dofs(); 

  for(unsigned int i=0; i<totalNodes; i++){
    if(nodeLocation[i][0] < 0.5){
      D_trans[i] = 300. ; 
    else{
      D_trans[i] = 300. + 20. * (nodeLocation[i][0] - 0.5); 
    }
  }
    
  //Find V_0 = M^{-1}*(F_0 - K*D_0)
  system_matrix.copy_from(M);
  K.vmult(RHS,D_trans); 
  RHS *= -1.; 	
  RHS.add(1.,F); 	
  MatrixTools::apply_boundary_values (boundary_values_of_V, system_matrix, V_trans, RHS, false);
  SparseDirectUMFPACK  A;
  A.initialize(system_matrix);
  A.vmult (V_trans, RHS); 
	
  output_trans_results(0);

  double current_l2norm = l2norm();
  std::vector<double> saveL2 = {0.0,current_l2norm};
  l2norm_results.push_back(saveL2);

}

//Template for transient solution using 
template <int dim>
void FEM<dim>::solve_trans(){

  apply_initial_conditions();

  std::cout << "Time step: " << delta_t << std::endl;
	std::cout << "Alpha: " << alpha << std::endl;


	const int phys_t      = 3000;	
	const int output_t    = 100;	

	const unsigned int num_steps 	= int(phys_t/delta_t);		
	const int mod_t               = int(output_t/delta_t);	

  const unsigned int totalNodes = dof_handler.n_dofs(); 
  Vector<double>     D_tilde(totalNodes);

   
  for(unsigned int t_step=1; t_step<num_steps+1; t_step++){ 
      for (size_t i = 0; i < D_trans.size(); ++i) {
      D_tilde[i] = D_trans[i] + delta_t * (1 - alpha) * V_trans[i];
    }

    system_matrix.copy_from(M); 
    system_matrix.add(alpha*delta_t,K);

    K.vmult(RHS,D_tilde); 
    RHS *= -1.; 	
    RHS.add(1.,F); 	

    MatrixTools::apply_boundary_values (boundary_values_of_V, system_matrix, V_trans, RHS, false);

    SparseDirectUMFPACK  A;
    A.initialize(system_matrix);
    A.vmult (V_trans, RHS); 

    for (size_t i = 0; i < D_trans.size(); ++i) {
      D_trans[i] = D_tilde[i] + delta_t * alpha * V_trans[i];
    }
      if(t_step%mod_t == 0){
        output_trans_results(t_step*delta_t);

        double current_l2norm = l2norm();
        std::vector<double> saveL2 = {t_step*delta_t,current_l2norm};
        l2norm_results.push_back(saveL2);
    }
  }
}


template <int dim>
void FEM<dim>::output_steady_results (){
  //Write results to VTK file
	std::string steadySavePath = "Alpha"+alphaAsString+sep+"solution_steady.vtk";
	std::ofstream output1 (steadySavePath);
  DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler);


  data_out.add_data_vector (D_steady, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation);
  data_out.build_patches (); data_out.write_vtk (output1); output1.close();
}


template <int dim>
void FEM<dim>::output_trans_results (unsigned int index){
  //This adds an index to your filename so that you can distinguish between time steps

  char filename[100];
  snprintf(filename, 100, "solution_%d.vtk", index);
  std::ofstream output1 (savePath+sep+filename);
  DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler);

  data_out.add_data_vector (D_trans, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation);
  data_out.build_patches (); data_out.write_vtk (output1); output1.close();
}

//Function to calculate the l2norm of the difference between the current and steady state solutions.
template <int dim>
double FEM<dim>::l2norm(){
  double l2norm = 0.;
  double u_steady, u_trans;

  FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values |
                            update_JxW_values);

  const unsigned int 				dofs_per_elem = fe.dofs_per_cell; 
  std::vector<unsigned int> local_dof_indices (dofs_per_elem);
  const unsigned int 				num_quad_pts = quadrature_formula.size(); 

  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active (), 
    endc = dof_handler.end();
  for (;elem!=endc; ++elem){
    elem->get_dof_indices (local_dof_indices);
    fe_values.reinit(elem);

    for(unsigned int q=0; q<num_quad_pts; q++){
      u_steady = 0.; u_trans = 0.;
      for(unsigned int A=0; A<dofs_per_elem; A++){

        u_steady += D_steady[local_dof_indices[A]]*fe_values.shape_value(A,q);
        u_trans += D_trans[local_dof_indices[A]]*fe_values.shape_value(A, q);
  
      }
      l2norm += (u_steady - u_trans) * (u_steady - u_trans) * fe_values.JxW(q);
    }
  }
  return sqrt(l2norm);
}

template <int dim>
void FEM<dim>::setSavePath(){
  sep = std::filesystem::path::preferred_separator;

  alphaAsString = std::to_string(alpha);
  alphaAsString = alphaAsString.replace(1,1,"_").substr(0,3);

  int endDelta = (delta_t >= 10.0) ? 2 : 3;
  if (delta_t >= 100.0){ endDelta = 3; }
  if (delta_t < 0.1){ endDelta = 4; }
  if (delta_t < 0.01){ endDelta = 5; }
  if (delta_t < 0.001){ endDelta = 6; }

    deltaAsString = std::to_string(delta_t);
  std::replace( deltaAsString.begin(), deltaAsString.end(), '.', '_');

  deltaAsString = deltaAsString.substr(0,endDelta);

  savePath = "Alpha"+alphaAsString+sep+"dt"+deltaAsString;

  std::filesystem::create_directory("Alpha"+alphaAsString);
  std::filesystem::create_directory(savePath);

}


template <int dim>
void FEM<dim>::writeL2VectorToCSV(){

	// Open a file to write
	std::ofstream myL2file;
	myL2file.open (savePath+sep+"L2norm.csv");
	myL2file << "t (s),L2\n";

  	for(unsigned int i=0; i<l2norm_results.size(); i++){
		myL2file << l2norm_results[i][0] << "," << l2norm_results[i][1] << "\n";
	}
	myL2file.close();

}