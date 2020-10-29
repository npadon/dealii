#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <np_utilities.h>


using namespace dealii;

template <int dim>
class CaseStudy01
{
public:
	CaseStudy01 ();
  ~CaseStudy01 ();
  void run ();

private:
  void generate_initial_mesh();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle);

  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;
  FE_Q<dim>            fe;
  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       	solution;
  Vector<double>       	system_rhs;
  double				T_inf;
  double				T_hot;
  ConvergenceTable			table;
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
	RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;
};



template <int dim>
inline
double RightHandSide<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
	double radius = .005; //m
	//const double pi = 3.14159265359;
	//double volume = 4.0/3.0*pi*radius*radius*radius;
	Point<3> center(0,0,.05/2.0);
	if ((p-center).norm_square() < radius*radius)
		return 0.5;
	else
		return 0;
}

template <int dim>
void RightHandSide<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points,
          ExcDimensionMismatch (values.size(), n_points));

  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  for (unsigned int i=0; i<n_points; ++i)
	  values[i]=RightHandSide<dim>::value(points[i]);
}


template <int dim>
CaseStudy01<dim>::CaseStudy01 ()
  :
  dof_handler (triangulation),
  fe (1),
  T_inf(30),
  T_hot(100)
{
	table.clear();
}


template <int dim>
CaseStudy01<dim>::~CaseStudy01 ()
{
  dof_handler.clear ();
}


template <int dim>
void CaseStudy01<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            1,
                                            ConstantFunction<dim>(T_hot),
                                            constraints);
  constraints.close ();
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);
}

template <int dim>
void CaseStudy01<dim>::assemble_system ()
{
  const int gaussian_integration_points = 2;
  const QGauss<dim>  quadrature_formula(gaussian_integration_points);
  const QGauss<dim-1> face_quadrature_formula(gaussian_integration_points);

  double k = 0.2; //  J/[kg K]
  double h = 5.0;
  double alpha = 1.44e-7;

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                           update_values    |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   	dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   	n_q_points    = quadrature_formula.size();
  const unsigned int 	n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  RightHandSide<dim>	rhs;
  std::vector<double>	rhs_values(n_q_points);


  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit (cell);
      rhs.value_list(fe_values.get_quadrature_points(),rhs_values);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              cell_matrix(i,j) += (alpha*
            		  	  	  	   fe_values.shape_grad(i,q_index) *
                                   fe_values.shape_grad(j,q_index) *
                                   fe_values.JxW(q_index));
            }
            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            rhs_values[i] *
                            fe_values.JxW(q_index));
          }
      }
      for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
      {
    	  if (cell->face(face_number)->at_boundary() &&
               (cell->face(face_number)->boundary_id() == 2))
                  {
                    fe_face_values.reinit (cell, face_number);
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                      {
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                           for(unsigned int j=0; j<dofs_per_cell;++j)
                           {
                        	   cell_matrix(i,j)+=(alpha*h/k*
                        			   	   	   	 fe_face_values.shape_value(i,q_point)*
                        			   	   	   	 fe_face_values.shape_value(j,q_point)*
                        			   	   	   	 fe_face_values.JxW(q_point));
                           }
                        	cell_rhs(i) += (alpha*h/k*T_inf*
                                          fe_face_values.shape_value(i,q_point) *
                                          fe_face_values.JxW(q_point));
                        }
                      }
                  }
      }

      // Finally, transfer the contributions from @p cell_matrix and
      // @p cell_rhs into the global objects.
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
}


template <int dim>
void CaseStudy01<dim>::solve ()
{
  SolverControl      solver_control (1000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

  constraints.distribute (solution);
}


template <int dim>
void CaseStudy01<dim>::refine_grid ()
{
  /*Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);
  triangulation.execute_coarsening_and_refinement ();*/
	triangulation.refine_global(1);
}


template <int dim>
void CaseStudy01<dim>::output_results (const unsigned int cycle)
{
  Assert (cycle < 10, ExcNotImplemented());

  std::string filename = "solution-";
  filename += ('0' + cycle);
  filename += ".vtk";
  std::ofstream output (filename.c_str());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  data_out.write_vtk (output);


  table.add_value("cycle",cycle);
  table.add_value("cells", triangulation.n_active_cells());
  table.add_value("dofs", dof_handler.n_dofs());
  table.add_value("value", VectorTools::point_value(dof_handler,solution,Point<dim>(0,0,0.05/4)));
}

template <int dim>
void CaseStudy01<dim>::generate_initial_mesh ()
{
	std::cout << "Generating coarse mesh..." << std::endl;
	double radius=.01,height=.05,n_cells=1;
	Point<2> center(0,0);
	Triangulation<2,2> temp;
	GridGenerator::hyper_ball(temp,center,radius);
	GridGenerator::extrude_triangulation(temp,n_cells+1,height,triangulation);
	//make a cylindrical manifold in the z-axis (x=0,y=1,z=2)
	static const CylindricalManifold<dim> cyl_boundary(2);
	triangulation.set_all_manifold_ids_on_boundary(0);
	triangulation.set_manifold(0,cyl_boundary);
	triangulation.refine_global(1);

	//set the boundary ids of the triangulation
	typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
	for (;cell != triangulation.end();++cell)
	{
	  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	    if (cell->face(f)->at_boundary())
	    {
	    //check the cell faces with z=0, and set the boundary indicator equal to 1
	      if (std::fabs(cell->face(f)->center()[2] - (0.0)) < 1e-10)
	    	  cell->face(f)->set_boundary_id (1);
	      else
	    	  cell->face(f)->set_boundary_id(2);
	    }
	}

	//output the boundary IDs
	print_boundary_indicators(triangulation);

}
template <int dim>
void CaseStudy01<dim>::run ()
{
  for (unsigned int cycle=0; cycle<5; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        generate_initial_mesh();
      else
        refine_grid ();


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      setup_system ();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      assemble_system ();
      solve ();
      output_results (cycle);
    }
  std::ofstream table_output_file("convergence_output.txt");
  table.write_text(table_output_file);
  table_output_file.close();
  std::cout << "Mesh convergence table written." << std::endl;
}


int main ()
{

  try
    {
      deallog.depth_console (0);
      CaseStudy01<3> prob;
      prob.run ();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
