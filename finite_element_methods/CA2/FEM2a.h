/*This is a template file for use with 2D finite elements (scalar field).
  The portions of the code you need to fill in are marked with the comment "//EDIT".

  Do not change the name of any existing functions, but feel free
  to create additional functions, variables, and constants.
  It uses the deal.II FEM library.*/

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

using namespace dealii;

template <int dim>
class FEM
{
 public:
  //Class functions
  FEM(unsigned int problem);  // Class constructor 
  ~FEM();									    //Class destructor

  //Define your 2D basis functions and derivatives
  double basis_function(unsigned int node, double xi_1, double xi_2);
  std::vector<double> basis_gradient(unsigned int node, double xi_1, double xi_2);

  //Solution steps
  void generate_mesh(std::vector<unsigned int> numberOfElements);
  void define_boundary_conds();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results();

  //Function to calculate the l2 norm of the error in the finite element sol'n vs. the exact solution (problem 2)
  double l2norm_of_error();

  //Class objects
  Triangulation<dim>   triangulation; //mesh
  FESystem<dim>        fe;	          //FE element
  DoFHandler<dim>      dof_handler;   //Connectivity matrices

  //Gaussian quadrature - These will be defined in setup_system()
  unsigned int	      quadRule;    //quadrature rule, i.e. number of quadrature points
  std::vector<double> quad_points; //vector of Gauss quadrature points
  std::vector<double> quad_weight; //vector of the quadrature point weights
    
  //Data structures
  SparsityPattern               sparsity_pattern; //Sparse matrix pattern
  SparseMatrix<double>          K;                //Global stiffness (sparse) matrix
  Vector<double>                D, F;             //Global vectors - Solution vector (D) and Global force vector (F)
  Table<2,double>               nodeLocation;     //Table of the coordinates of nodes by global dof number
  std::map<unsigned int,double> boundary_values;  //Map of dirichlet boundary conditions 
	double prob;                                    //Problem number, defined in the main cc file

  //Solution name array
  std::vector<std::string> nodal_solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
};

// Class constructor for a scalar field
template <int dim>
FEM<dim>::FEM (unsigned int problem)
:
fe (FE_Q<dim>(1), 1), 
  dof_handler (triangulation)
{
	prob = problem;

  //Nodal Solution names - this is for writing the output file
  nodal_solution_names.push_back("D");
  nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
}

//Class destructor
template <int dim>
FEM<dim>::~FEM (){
  dof_handler.clear ();
}


template <int dim>
double FEM<dim>::basis_function(unsigned int node, double xi_1, double xi_2){

  double value = 0.; 

   if (node == 0) { // lower-left node
    value = 0.25 * (1 - xi_1) * (1 - xi_2);
  }
  else if (node == 1) { // lower-right node
    value = 0.25 * (1 + xi_1) * (1 - xi_2);
  }
  else if (node == 2) { // upper-left node
    value = 0.25 * (1 - xi_1) * (1 + xi_2);
  }
  else if (node == 3) { // upper-right node
    value = 0.25 * (1 + xi_1) * (1 + xi_2);
  }
  return value;
}

template <int dim>
std::vector<double> FEM<dim>::basis_gradient(unsigned int node, double xi_1, double xi_2){

  std::vector<double> values(dim,0.0);

 if (node == 0) { // lower-left node
    values[0] = -0.25 * (1 - xi_2);  
    values[1] = -0.25 * (1 - xi_1);
  }
  else if (node == 1) { // lower-right node
    values[0] = 0.25 * (1 - xi_2);
    values[1] = -0.25 * (1 + xi_1);
  }
  else if (node == 2) { // upper-left node
    values[0] = -0.25 * (1 + xi_2);
    values[1] = 0.25 * (1 - xi_1);
  }
  else if (node == 3) { // upper-right node
    values[0] = 0.25 * (1 + xi_2);
    values[1] = 0.25 * (1 + xi_1);
  }

  return values;
}

template <int dim>
void FEM<dim>::generate_mesh(std::vector<unsigned int> numberOfElements){

  //Define the limits of your domain
  double x_min = 0., //x is element of [0, 0.03]
    x_max = 0.03, 
    y_min = 0. , //y is element of [0, 0.08]
    y_max = 0.08;

  Point<dim,double> min(x_min,y_min), max(x_max,y_max);
  GridGenerator::subdivided_hyper_rectangle (triangulation, numberOfElements, min, max);
}

//Specify the Dirichlet boundary conditions
template <int dim>
void FEM<dim>::define_boundary_conds(){
	//Define the limits of your domain
  double x_min = 0., //x is element of [0, 0.03]
    x_max = 0.03, 
    y_min = 0. , //y is element of [0, 0.08]
    y_max = 0.08,
    c0 = 1./3.,
    c_0 = 8.,
    f = -10000.,
    kappa = 385.;
  /*Note: this will be very similiar to the define_boundary_conds function
    in the CA1 template. You will loop over all nodes and use "nodeLocations"
    to check if the node is on the boundary with a Dirichlet condition. If it is,
    then add the node number and the specified value (temperature in this problem)
    to the boundary values map, something like this:

    boundary_values[globalNodeIndex] = dirichletTemperatureValue

    Note that "nodeLocation" is now a Table instead of just a vector. The row index is
    the global node number; the column index refers to the x or y component (0 or 1 for 2D).
    e.g. nodeLocation[7][1] is the y coordinate of global node 7

		Problem 1 and problem 2 have different Dirichlet boundary conditions.*/

  const unsigned int totalNodes = dof_handler.n_dofs(); //Total number of nodes

	//Identify dirichlet boundary nodes and specify their values.


	if(prob == 1){
  for(unsigned int globalNode=0; globalNode<totalNodes; globalNode++){
    if(nodeLocation[globalNode][1] == y_min){
      boundary_values[globalNode] = 300. * (1. + (c0 * nodeLocation[globalNode][0]));
 
    }
    else if(nodeLocation[globalNode][1] == y_max){
      boundary_values[globalNode] = 310. * ( 1. + (c_0 * nodeLocation[globalNode][0] * nodeLocation[globalNode][0]));
    }
  }
	}
	else if(prob == 2){
  	//EDIT - Define the Dirichlet boundary conditions.
    for(unsigned int globalNode=0; globalNode<totalNodes; globalNode++){ //loop over the nodes 0 till L
    if(nodeLocation[globalNode][1] == y_min){
      boundary_values[globalNode] = 100. + (((f/(4. * kappa)) * nodeLocation[globalNode][0] * nodeLocation[globalNode][0]));
      
    }
    else if(nodeLocation[globalNode][0] == x_min){
      boundary_values[globalNode] = 100. + (((f/(4. * kappa)) * nodeLocation[globalNode][1] * nodeLocation[globalNode][1]));
   
    }
    if(nodeLocation[globalNode][1] == y_max){
      boundary_values[globalNode] = 100. + ((f/(4. * kappa)) * ((nodeLocation[globalNode][0] * nodeLocation[globalNode][0]) + 0.0064));
     
    }
    else if(nodeLocation[globalNode][0] == x_max){
      boundary_values[globalNode] = 100. + ((f/(4. * kappa)) * ((nodeLocation[globalNode][1] * nodeLocation[globalNode][1]) + 0.0009));
    
    }
  }
	}
}

//Setup data structures (sparse matrix, vectors)
template <int dim>
void FEM<dim>::setup_system(){

  //Let deal.II organize degrees of freedom
  dof_handler.distribute_dofs (fe);

  //Fill in the Table "nodeLocations" with the x and y coordinates of each node by its global index
  MappingQ1<dim,dim> mapping;
  std::vector< Point<dim,double> > dof_coords(dof_handler.n_dofs());
  nodeLocation.reinit(dof_handler.n_dofs(),dim);
  DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords);
  for(unsigned int i=0; i<dof_coords.size(); i++){
    for(unsigned int j=0; j<dim; j++){
      nodeLocation[i][j] = dof_coords[i][j];
    }
  }

  //Specify boundary condtions (call the function)
  define_boundary_conds();

  //Define the size of the global matrices and vectors
  sparsity_pattern.reinit (dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  sparsity_pattern.compress();
  K.reinit (sparsity_pattern);
  F.reinit (dof_handler.n_dofs());
  D.reinit (dof_handler.n_dofs());

  //Define quadrature rule - again, you decide what quad rule is needed
  quadRule = 5; //EDIT - Number of quadrature points along one dimension
  quad_points.resize(quadRule); quad_weight.resize(quadRule);

  quad_points[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.)); 
  quad_points[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.)); 
  quad_points[2] = 0.;
  quad_points[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.)); 
  quad_points[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.)); 

  quad_weight[0] = (322.-13.*sqrt(70.))/900.;
  quad_weight[1] = (322.+13.*sqrt(70.))/900.;
  quad_weight[2] = 128./225.;
  quad_weight[3] = (322.+13.*sqrt(70.))/900.;
  quad_weight[4] = (322.-13.*sqrt(70.))/900.;


  //Just some notes...
  std::cout << "   Number of active elems:       " << triangulation.n_active_cells() << std::endl;
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
  std::cout << "   Quadrule: " << quadRule << std::endl;
}

//Form elmental vectors and matrices and assemble to the global vector (F) and matrix (K)
template <int dim>
void FEM<dim>::assemble_system(){

  K=0; F=0;

  const unsigned int   	    dofs_per_elem = fe.dofs_per_cell; //This gives you number of degrees of freedom per element
  FullMatrix<double> 	    Klocal (dofs_per_elem, dofs_per_elem);
  Vector<double>      	    Flocal (dofs_per_elem);
  std::vector<unsigned int> local_dof_indices (dofs_per_elem);

  //loop over elements  
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end();
  for (;elem!=endc; ++elem){


    elem->get_dof_indices (local_dof_indices);

    //Loop over local DOFs and quadrature points to populate Flocal and Klocal.
    FullMatrix<double> Jacobian(dim,dim);
    double detJ, f = -10000.;

    //Loop over local DOFs and quadrature points to populate Flocal
    Flocal = 0.;
    for(unsigned int q1=0; q1<quadRule; q1++){
      for(unsigned int q2=0; q2<quadRule; q2++){
        Jacobian = 0.;
        for(unsigned int i=0;i<dim;i++){
          for(unsigned int j=0;j<dim;j++){
            for(unsigned int A=0; A<dofs_per_elem; A++){
              Jacobian[i][j] += nodeLocation[local_dof_indices[A]][i]*basis_gradient(A,quad_points[q1],quad_points[q2])[j];
            }
          }
        }
        detJ = Jacobian.determinant();
        for(unsigned int A=0; A<dofs_per_elem; A++){
          //EDIT - Define Flocal here.
          if (prob == 2){
          Flocal(A) -= detJ * f *  basis_function(A, quad_points[q1], quad_points[q2]) * quad_weight[q1] * quad_weight[q2];
          }
        }
      }
    }
    //Loop over local DOFs and quadrature points to populate Klocal
    FullMatrix<double> invJacob(dim,dim), kappa(dim,dim);

    //"kappa" is the conductivity tensor
    kappa = 0.;
    kappa[0][0] = 385.;
    kappa[1][1] = 385.;

    //Loop over local DOFs and quadrature points to populate Klocal
    Klocal = 0.;
    for(unsigned int q1=0; q1<quadRule; q1++){
      for(unsigned int q2=0; q2<quadRule; q2++){
        //Find the Jacobian at a quadrature point
        Jacobian = 0.;
        for(unsigned int i=0;i<dim;i++){
          for(unsigned int j=0;j<dim;j++){
            for(unsigned int A=0; A<dofs_per_elem; A++){
              Jacobian[i][j] += nodeLocation[local_dof_indices[A]][i]*basis_gradient(A,quad_points[q1],quad_points[q2])[j];
            }
          }
        }
        detJ = Jacobian.determinant();
        invJacob.invert(Jacobian);
        for(unsigned int A=0; A<dofs_per_elem; A++){
          for(unsigned int B=0; B<dofs_per_elem; B++){
            for(unsigned int i=0;i<dim;i++){
              for(unsigned int j=0;j<dim;j++){
                for(unsigned int I=0;I<dim;I++){
                  for(unsigned int J=0;J<dim;J++){
                    //EDIT - Define Klocal. You will need to use the inverse Jacobian ("invJacob") and "detJ"
                    Klocal(A, B) += kappa[i][j] * detJ * invJacob[I][i] * invJacob[J][j] * basis_gradient(A, quad_points[q1], quad_points[q2])[I] * basis_gradient(B, quad_points[q1], quad_points[q2])[J] * quad_weight[q1] * quad_weight[q2];
                }
               }
              }
            }
          }
        }
      }
    }
    for(unsigned int A=0; A<dofs_per_elem; A++){
	    F(local_dof_indices[A]) += Flocal(A);
      for(unsigned int B=0; B<dofs_per_elem; B++){
	      K.add(local_dof_indices[A], local_dof_indices[B],Klocal(A,B));
      }
    }
  }

  //Apply Dirichlet boundary conditions
  MatrixTools::apply_boundary_values (boundary_values, K, D, F, false);
}

//Solve for D in KD=F
template <int dim>
void FEM<dim>::solve(){

  //Solve for D
  SparseDirectUMFPACK  A;
  A.initialize(K);
  A.vmult (D, F); //D=K^{-1}*F

}

//Output results
template <int dim>
void FEM<dim>::output_results (){

  char solutionName[21];
  sprintf(solutionName, "CA2a_Problem%d",int(prob));

  std::string solutionFileName(solutionName); solutionFileName += ".vtk";
  
  std::cout << "Writing solution for Coding Assignment 2, prob " << prob << " to file : " << solutionFileName.c_str() << std::endl;

  //Write results to VTK file
  std::ofstream output1(solutionFileName.c_str());
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  //Add nodal DOF data
  data_out.add_data_vector(D, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation);
  data_out.build_patches();
  data_out.write_vtk(output1);
  output1.close();
}

template <int dim>
double FEM<dim>::l2norm_of_error(){
	
	double l2norm = 0.;

	//Find the l2 norm of the error between the finite element sol'n and the exact sol'n
	//(For problem 2 only)
	const unsigned int   			dofs_per_elem = fe.dofs_per_cell; //This gives you dofs per element
	std::vector<unsigned int> local_dof_indices (dofs_per_elem);
	double u_exact, u_h, x, y;
	FullMatrix<double> Jacobian(dim,dim);
	double detJ;
  	double f = -10000. , kappa = 385.;

	//loop over elements  
	typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active (), endc = dof_handler.end();
	for (;elem!=endc; ++elem){

		//Retrieve the effective "connectivity matrix" for this element
		elem->get_dof_indices (local_dof_indices);
		for(unsigned int q1=0; q1<quadRule; q1++){
			for(unsigned int q2=0; q2<quadRule; q2++){
				Jacobian = 0.;
        for(unsigned int i=0;i<dim;i++){
          for(unsigned int j=0;j<dim;j++){
            for(unsigned int A=0; A<dofs_per_elem; A++){
              Jacobian[i][j] += nodeLocation[local_dof_indices[A]][i]*basis_gradient(A,quad_points[q1],quad_points[q2])[j];
            }
          }
        }
        detJ = Jacobian.determinant();
				x = 0.; y = 0.; u_h = 0.;
				for(unsigned int B=0; B<dofs_per_elem; B++){
					x += nodeLocation[local_dof_indices[B]][0]*basis_function(B,quad_points[q1],quad_points[q2]);
          y  += nodeLocation[local_dof_indices[B]][1]*basis_function(B,quad_points[q1],quad_points[q2]);
          u_h += D[local_dof_indices[B]]*basis_function(B,quad_points[q1],quad_points[q2]);
				}
        if (prob == 2){
      u_exact = 100. + ((f/(4.* kappa)) * ((pow(x,2)) + pow(y,2)));
      }
	l2norm += pow((u_exact - u_h),2) * quad_weight[q1]* quad_weight[q2] * detJ;
			}
		}
	}
	std::cout<<"uexact for problem 2: "<< u_exact << std::endl;
      	std::cout<<"uh estimated: "<< u_h << std::endl;
      	std::cout<<"x: "<< x << std::endl;
      	std::cout<<"y: "<< y << std::endl;
      	std::cout<<"detJ: "<< detJ << std::endl;
	return sqrt(l2norm);
}
