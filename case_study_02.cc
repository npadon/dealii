#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/config.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "np_utilities.h"

namespace CaseStudy02
{
	using namespace dealii;
	template <int dim>
	SymmetricTensor<4,dim>
	get_stress_strain_tensor (double lambda=0, double mu=0)
	{
		const double	youngs_modulus=200e9,poisson=0.3;
		lambda = 		poisson*youngs_modulus/((1+poisson)*(1-2*poisson));
		mu =			youngs_modulus/(2*(1+poisson));

		SymmetricTensor<4,dim> tmp;
		for (unsigned int i=0; i<dim; ++i)
			for (unsigned int j=0; j<dim; ++j)
				for (unsigned int k=0; k<dim; ++k)
					for (unsigned int l=0; l<dim; ++l)
						tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
											((i==l) && (j==k) ? mu : 0.0) +
											((i==j) && (k==l) ? lambda : 0.0));
		return tmp;
	}
	template <int dim>
	inline
	SymmetricTensor<2,dim>
	get_strain (const FEValues<dim> &fe_values,
			  const unsigned int   shape_func,
			  const unsigned int   q_point)
	{
	SymmetricTensor<2,dim> tmp;
	for (unsigned int i=0; i<dim; ++i)
	  tmp[i][i] = fe_values.shape_grad_component (shape_func,q_point,i)[i];
	for (unsigned int i=0; i<dim; ++i)
	  for (unsigned int j=i+1; j<dim; ++j)
		tmp[i][j]
		  = (fe_values.shape_grad_component (shape_func,q_point,i)[j] +
			 fe_values.shape_grad_component (shape_func,q_point,j)[i]) / 2;
	return tmp;
	}

const char* get_direction(const int coordinate)
{
	Assert(coordinate<=2,ExcNotImplemented());
	if(coordinate==0)
		return "x";
	else if(coordinate==1)
		return "y";
	else if (coordinate==2)
		return "z";
	return "INVALID";
}

template<int dim>
double get_vonMisesStress(const SymmetricTensor<2,dim>& stress_tensor)
{
	SymmetricTensor<2,dim> deviatoric_stress_tensor = deviator(stress_tensor);
	return sqrt(3/2*deviatoric_stress_tensor*deviatoric_stress_tensor);
}
	  template <int dim>
	  class BodyForce :  public Function<dim>
	  {
	  public:
	    BodyForce ();
	    virtual
	    void
	    vector_value (const Point<dim> &p,
	                  Vector<double>   &values) const;
	    virtual
	    void
	    vector_value_list (const std::vector<Point<dim> > &points,
	                       std::vector<Vector<double> >   &value_list) const;
	  };
	  template <int dim>
	  BodyForce<dim>::BodyForce ()
	    :
	    Function<dim> (dim)
	  {}
	  template <int dim>
	  inline
	  void
	  BodyForce<dim>::vector_value (const Point<dim> &/*p*/,
	                                Vector<double>   &values) const
	  {
	    Assert (values.size() == dim,
	            ExcDimensionMismatch (values.size(), dim));
	   // const double g   = 9.81;
	    //const double rho = 7700;
	    values = 0;
	    values(dim-1) = 0; //-rho * g;
	  }
	  template <int dim>
	  void
	  BodyForce<dim>::vector_value_list (const std::vector<Point<dim> > &points,
	                                     std::vector<Vector<double> >   &value_list) const
	  {
	    const unsigned int n_points = points.size();
	    Assert (value_list.size() == n_points,
	            ExcDimensionMismatch (value_list.size(), n_points));
	    for (unsigned int p=0; p<n_points; ++p)
	      BodyForce<dim>::vector_value (points[p],
	                                    value_list[p]);
	  }

		template <int dim>
		class TractionForce :  public Function<dim>
		{
			public:
			TractionForce ();
			virtual
			void
			vector_value (const Point<dim> &p,
						  Vector<double>   &values) const;
			virtual
			void
			vector_value_list (const std::vector<Point<dim> > &points,
							   std::vector<Vector<double> >   &value_list) const;
		};

		template <int dim>
		TractionForce<dim>::TractionForce ()
			:
			Function<dim> (dim)
			{}

		template <int dim>
		inline
		void
		TractionForce<dim>::vector_value (const Point<dim> &/*p*/,
									Vector<double>   &values) const
		{
			Assert (values.size() == dim,
					ExcDimensionMismatch (values.size(), dim));

			values = 0;
			values(dim-1) = -1e6;
		}
	template <int dim>
	void
	TractionForce<dim>::vector_value_list (const std::vector<Point<dim> > &points,
									 std::vector<Vector<double> >   &value_list) const
	{
		const unsigned int n_points = points.size();
		Assert (value_list.size() == n_points,
				ExcDimensionMismatch (value_list.size(), n_points));
		for (unsigned int p=0; p<n_points; ++p)
			TractionForce<dim>::vector_value (points[p],
										value_list[p]);
	}
	template<int dim> class PointForce : public Function<dim>
	{
	public:
		PointForce();
		virtual void vector_value(const Point<dim>& p,
									Vector<double>& values) const;
		virtual void vector_value_list(const std::vector<Point<dim>>& points,
										std::vector<Vector<double>>& value_list) const;
	};
	template<int dim> PointForce<dim>::PointForce():
			Function<dim> (dim)
	{

	}

	template<int dim>
	inline
	void PointForce<dim>::vector_value(const Point<dim>& p,
									Vector<double>& values) const
	{
		Assert(values.size()==dim,ExcDimensionMismatch(values.size(),dim));
		Assert(dim==2,ExcNotImplemented());

		Point<dim> apply_force_point(1.0,.1);

		//apply a force along the top
		if(equivalent_points(apply_force_point,p))
		{
			values(0)=0;
			values(1)=0;//-10000;
		}
		else
		{
			values(0)=0;
			values(1)=0;
		}
	}
	template<int dim>
	inline
	void PointForce<dim>::vector_value_list(const std::vector<Point<dim>>& points,
											std::vector<Vector<double>>& value_list) const
	{
		Assert(value_list.size()==points.size(),ExcDimensionMismatch(value_list.size(),points.size()));
		const unsigned int n_points = points.size();
		for(int i=0;i<n_points;++i)
			PointForce<dim>::vector_value(points[i],value_list[i]);
	}


	template<int dim> class BoundaryValues: Function<dim>
	{
	public:
		BoundaryValues();
		virtual void vector_value(const Point<dim>& p,Vector<double>& values) const;
		virtual void vector_value_list(const std::vector<Point<dim>>& points,
										std::vector<Vector<double>>& value_list) const;
	};
	template<int dim> BoundaryValues<dim>::BoundaryValues():
			Function<dim>(dim)
	{

	}
	template<int dim> void BoundaryValues<dim>::vector_value(const Point<dim>& p,Vector<double>& values) const
		{
			Assert(values.size()==dim,ExcDimensionMismatch(values.size(),dim));
			Assert(dim ==2,ExcNotImplemented());
			double boundary_value = 0.0;
			for(unsigned int i=0;i<dim;i++)
				values[i]=0;
		}
	template<int dim> void BoundaryValues<dim>::vector_value_list(const std::vector<Point<dim>>& points,
			std::vector<Vector<double>>& value_list) const
	{
		Assert(value_list.size()==points.size(),ExcDimensionMismatch(value_list.size(),points.size()));
		const unsigned int n_points = points.size();
		for(int i=0;i<n_points;++i)
			BoundaryValues<dim>::vector_value(points[i],value_list[i]);
	}

	template <int dim> class PlaneStrain
	{
	public:
		PlaneStrain();
		~PlaneStrain();
		void run();
	private:
		void setup_system();
		void assemble_system();
		void solve();
		void refine_grid();
		void output_results(const int cycle);

		Triangulation<dim> 		triangulation;
		DoFHandler<dim>			dof_handler;
		FESystem<dim>			fe;
		ConstraintMatrix		hanging_node_constraints;
		SparsityPattern			sparsity_pattern;
		SparseMatrix<double> 	system_matrix;
		Vector<double>			solution;
		Vector<double>			system_rhs;
		static const SymmetricTensor<4,dim> stress_strain_tensor;
		int 					gaussian_integration_points;
		ConvergenceTable        convergence_table;

	};

	template <int dim>
	const SymmetricTensor<4,dim>
	PlaneStrain<dim>::stress_strain_tensor = get_stress_strain_tensor<dim> ();

	template <int dim> PlaneStrain<dim>::PlaneStrain()
	:
	dof_handler(triangulation),
	fe(FE_Q<dim>(1),dim)
	{
		gaussian_integration_points = 2;
	}
	template <int dim> PlaneStrain<dim>::~PlaneStrain()
	{
		dof_handler.clear();
	}
	template <int dim> void PlaneStrain<dim>::setup_system()
	{
	    dof_handler.distribute_dofs (fe);
	    hanging_node_constraints.clear ();
	    DoFTools::make_hanging_node_constraints (dof_handler,
	                                             hanging_node_constraints);
	    hanging_node_constraints.close ();

	    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	    DoFTools::make_sparsity_pattern(dof_handler,
	                                    dsp,
	                                    hanging_node_constraints,
	                                    /*keep_constrained_dofs = */ true);
	    sparsity_pattern.copy_from (dsp);
	    //make the dimensions of each matrix consistent
	    system_matrix.reinit (sparsity_pattern);
	    solution.reinit (dof_handler.n_dofs());
	    system_rhs.reinit (dof_handler.n_dofs());
	}
	template <int dim> void PlaneStrain<dim>::assemble_system()
	{
		const QGauss<dim>  quadrature_formula(gaussian_integration_points);
	    const QGauss<dim-1> face_quadrature_formula(gaussian_integration_points);

	    FEValues<dim> fe_values (fe, quadrature_formula,
	                             update_values   | update_gradients |
	                             update_quadrature_points | update_JxW_values);
	    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
	                             update_values    |
	                             update_quadrature_points  |  update_JxW_values);

	    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	    const unsigned int   n_q_points    = quadrature_formula.size();
	    const unsigned int	 n_face_q_points = face_quadrature_formula.size();

	    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	    Vector<double>       cell_rhs (dofs_per_cell);

	    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	    PointForce<dim>      					point_forces;
	    Vector<double>							point_force_values(fe.dofs_per_vertex);
	    std::vector<bool>						vertices_touched(triangulation.n_used_vertices(),false);


	    BodyForce<dim>      					body_force;
	    std::vector<Vector<double> > 			body_force_values (n_q_points,
	                                                       	   	   Vector<double>(dim));

	    TractionForce<dim>      				traction_force;
	    std::vector<Vector<double> > 			traction_force_values (n_face_q_points,
	                                                       	   	   Vector<double>(dim));
	    // Now we can begin with the loop over all cells:
	    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
	                                                   endc = dof_handler.end();
	    for (; cell!=endc; ++cell)
	      {
	        cell_matrix = 0;
	        cell_rhs = 0;

	        fe_values.reinit (cell);
	        body_force.vector_value_list (fe_values.get_quadrature_points(),
	                                        body_force_values);
	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	        {
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
					 for (unsigned int q_point=0; q_point<n_q_points;++q_point)
					   {
						 const SymmetricTensor<2,dim>
						 eps_phi_i = get_strain (fe_values, i, q_point),
						 eps_phi_j = get_strain (fe_values, j, q_point);
						 cell_matrix(i,j)
						 += (eps_phi_i * stress_strain_tensor * eps_phi_j *
							 fe_values.JxW (q_point));
					   }
				}
	        }
	        // Assembling the cell rhs - point values first

	        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
            {
            	double global_vertex_location = cell->vertex_index(v);
            	AssertIndexRange(global_vertex_location,vertices_touched.size());
            	if(vertices_touched[global_vertex_location]==false)
            	{
					vertices_touched[global_vertex_location] = true;
					point_forces.vector_value(cell->vertex(v),point_force_values);
					for(unsigned int i=0;i<fe.dofs_per_vertex;++i)
						cell_rhs(fe.dofs_per_vertex*v+i) += point_force_values(i);
            	}
            }
            //...then add any body forces
	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          {
	            const unsigned int
	            component_i = fe.system_to_component_index(i).first;

	            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	              cell_rhs(i) += fe_values.shape_value(i,q_point) *
	                             body_force_values[q_point](component_i) *
	                             fe_values.JxW(q_point);
	          }
	        //...then add any surface tractions
	        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
	        {
	      	  if (cell->face(face_number)->at_boundary() &&
	                 (cell->face(face_number)->boundary_id() == 2))
	                    {
	                      fe_face_values.reinit (cell, face_number);
	          	        traction_force.vector_value_list (fe_face_values.get_quadrature_points(),
	          	                                        traction_force_values);
	                      for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
	                        {
	                          for (unsigned int i=0; i<dofs_per_cell; ++i)
	                          {
	              	            const unsigned int
	              	            component_i = fe.system_to_component_index(i).first;
	                        	  cell_rhs(i) += (traction_force_values[q_point](component_i)*
	                                            fe_face_values.shape_value(i,q_point) *
	                                            fe_face_values.JxW(q_point));
	                          }
	                        }
	                    }
	        }
	        //go from element indices (k) to global indices (K)
	        cell->get_dof_indices (local_dof_indices);
	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          {
	        	//assemble the big matrix, K
	            for (unsigned int j=0; j<dofs_per_cell; ++j)
	            {
	              system_matrix.add (local_dof_indices[i],
	                                 local_dof_indices[j],
	                                 cell_matrix(i,j));
	            }

	            system_rhs(local_dof_indices[i]) += cell_rhs(i);
	          }

	      }

	    hanging_node_constraints.condense (system_matrix);
	    hanging_node_constraints.condense (system_rhs);

	    // The interpolation of the boundary values needs a small modification:
	    // since the solution function is vector-valued, so need to be the
	    // boundary values. The <code>ZeroFunction</code> constructor accepts a
	    // parameter that tells it that it shall represent a vector valued,
	    // constant zero function with that many components. By default, this
	    // parameter is equal to one, in which case the <code>ZeroFunction</code>
	    // object would represent a scalar function. Since the solution vector has
	    // <code>dim</code> components, we need to pass <code>dim</code> as number
	    // of components to the zero function as well.
	    std::map<types::global_dof_index,double> boundary_values;
	    VectorTools::interpolate_boundary_values (dof_handler,
	                                              0,
	                                              ZeroFunction<dim>(dim),
	                                              boundary_values);
	    MatrixTools::apply_boundary_values (boundary_values,
	                                        system_matrix,
	                                        solution,
	                                        system_rhs);

	}
	template <int dim> void PlaneStrain<dim>::solve()
	{
		SolverControl 			solver_control(5000,1e-5);
		SolverCG<>				cg(solver_control);
		PreconditionSSOR<>		preconditioner;

		preconditioner.initialize(system_matrix,1.1);
		cg.solve(system_matrix,solution,system_rhs,preconditioner);
		hanging_node_constraints.distribute(solution);
	}
	template <int dim> void PlaneStrain<dim>::refine_grid()
	{
		/*Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
		KellyErrorEstimator<dim>::estimate (dof_handler,
											QGauss<dim-1>(2),
											typename FunctionMap<dim>::type(),
											solution,
											estimated_error_per_cell);
		GridRefinement::refine_and_coarsen_fixed_number (triangulation,
															estimated_error_per_cell,
															0.3,
															0.03);
		triangulation.execute_coarsening_and_refinement();*/
		triangulation.refine_global(1);

	}

	template <int dim> void PlaneStrain<dim>::output_results(const int cycle)
	{
		bool plot_deformed_configuration = false;
		std::string sol_filename = "solution-";
		sol_filename += ('0'+cycle);
		sol_filename += ".vtk";
		Assert(cycle < 10, ExcInternalError());
		std::ofstream sol_output(sol_filename.c_str());

		DataOut<dim> data_out;
		int mapping_degree = fe.base_element(0).tensor_degree();

		MappingQEulerian<dim> q_mapping(mapping_degree, dof_handler, solution);

		data_out.attach_dof_handler (dof_handler);
		std::vector<std::string> solution_names;
		std::vector<std::string> rhs_names;
		std::vector<std::string> stresses_cell_names;
		std::vector<std::string> direct_stresses_nodes_names;
		std::vector<std::string> shear_stresses_nodes_names;

		int number_stress_components = 0;


		switch(dim)
		{
			case 1:
				solution_names.push_back("displacement");
				rhs_names.push_back("force");
				stresses_cell_names.push_back("stress_cell");

				direct_stresses_nodes_names.push_back("stress_node");
				shear_stresses_nodes_names.push_back("null");
				number_stress_components = 1;
				break;
			case 2:
				solution_names.push_back("x_displacement");
				solution_names.push_back("y_displacement");
				rhs_names.push_back("x_force");
				rhs_names.push_back("y_force");
				stresses_cell_names.push_back("stress_xx_cell");
				stresses_cell_names.push_back("stress_xy_cell");
				stresses_cell_names.push_back("stress_yy_cell");

				direct_stresses_nodes_names.push_back("stress_xx_nodes");
				direct_stresses_nodes_names.push_back("stress_yy_nodes");
				shear_stresses_nodes_names.push_back("stress_xy_nodes");
				shear_stresses_nodes_names.push_back("null");
				number_stress_components = 3;
				break;
			case 3:
				solution_names.push_back("x_displacement");
				solution_names.push_back("y_displacement");
				solution_names.push_back("z_displacement");
				rhs_names.push_back("x_force");
				rhs_names.push_back("y_force");
				rhs_names.push_back("z_force");
				stresses_cell_names.push_back("stress_xx_cell");
				stresses_cell_names.push_back("stress_xy_cell");
				stresses_cell_names.push_back("stress_xz_cell");
				stresses_cell_names.push_back("stress_yy_cell");
				stresses_cell_names.push_back("stress_yz_cell");
				stresses_cell_names.push_back("stress_zz_cell");

				direct_stresses_nodes_names.push_back("stress_xx_nodes");
				direct_stresses_nodes_names.push_back("stress_yy_nodes");
				direct_stresses_nodes_names.push_back("stress_zz_nodes");
				shear_stresses_nodes_names.push_back("stress_xy_nodes");
				shear_stresses_nodes_names.push_back("stress_xz_nodes");
				shear_stresses_nodes_names.push_back("stress_yz_nodes");
				number_stress_components = 6;
				break;
			default:
				Assert(false,ExcNotImplemented());
		}
		data_out.add_data_vector(solution,solution_names);
		data_out.add_data_vector(system_rhs,rhs_names);

		const std::vector<Point<dim>> support_points = fe.base_element(0).get_unit_support_points();
		std::vector<double>		weights(support_points.size(),1);
		const Quadrature<dim> node_quadrature_formula(support_points,weights);


		double num_cells = triangulation.n_active_cells();
		types::global_dof_index n_dofs = dof_handler.n_dofs();
		unsigned int n_node_q_points = node_quadrature_formula.size();
		double num_vertices = triangulation.n_used_vertices();

		std::cout << "Num cells: " << num_cells << std::endl;
		std::cout << "Num vertices: " << num_vertices << std::endl;
		std::cout << "Total DOF: " << n_dofs << std::endl;
		std::cout << "Length of solution vector:" << solution.size() << std::endl;
		std::cout << "Number of support points: " << support_points.size() << std::endl;
		std::cout << "Number of nodal quad points:" << node_quadrature_formula.size() << std::endl;

		FEValues<dim> fe_node_values (fe, node_quadrature_formula,
			                             update_values   | update_gradients |
			                             update_quadrature_points | update_JxW_values);

		typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
					                                                  endc = dof_handler.end();

		std::vector< SymmetricTensor<2,dim>> node_strain_results(n_node_q_points);
		SymmetricTensor<2,dim> cell_strain;
		SymmetricTensor<2,dim> cell_stress;

		std::vector<SymmetricTensor<2,dim>> sigma_cells(num_cells);
		std::vector<SymmetricTensor<2,dim>> sigma_nodes(n_dofs);
		Vector<double>	vonMisesCells(num_cells);
		Vector<double>  vonMisesNodes(n_dofs);
		std::vector<int> count_quad_points_written (n_dofs,0);

		std::vector<types::global_dof_index> global_dof_indices(fe.dofs_per_cell);
		int cell_index =0;

		//Evaluate a single point for mesh convergence purposes
		const Point<dim> convergence_point(0.04999,0.0999);
		SymmetricTensor<2,dim> point_strain;
		point_strain.clear();
		SymmetricTensor<2,dim> point_stress;
		std::vector<Tensor<1,dim,double>> convergence_point_gradient(dim,Tensor<1,dim,double>(dim));
		VectorTools::point_gradient(q_mapping,dof_handler,solution,convergence_point,convergence_point_gradient);
		for(int i=0;i<dim;++i)
		{
			for(int j=0;j<dim;++j)
			{
				point_strain[i][j] = 0.5*(convergence_point_gradient[i][j]+convergence_point_gradient[j][i]);
				//std::cout << i << ", " << j << ": " << point_strain[i][j] << std::endl;
			}
		}
		point_stress = stress_strain_tensor*point_strain;
		convergence_table.add_value("cycle",cycle);
		convergence_table.add_value("cells", num_cells);
		convergence_table.add_value("dofs", n_dofs);
		convergence_table.add_value("value", point_stress[0][0]);


		const FEValuesExtractors::Vector displacements (0);

		for(;cell!=endc;++cell)
		{
			fe_node_values.reinit(cell);
			cell_strain.clear();

			//get the strains at each support point
			fe_node_values[displacements].get_function_symmetric_gradients(solution,node_strain_results);

			//compute the stresses at each integration point and the average stresses on each cell
			cell->get_dof_indices(global_dof_indices);
			for(int q=0;q<n_node_q_points;++q)
			{
				cell_stress.clear();
				//compute the stress at the support point
				cell_stress = (stress_strain_tensor)*node_strain_results[q];

				//add the stress at the support point to the correct dof index
				//NOTE: because support points can be shared between cells, these
				//vectors will need to be averaged based on the number of times they've
				//been written to
				sigma_nodes[global_dof_indices[q*dim]] += cell_stress;
				count_quad_points_written[global_dof_indices[q*dim]] += 1;
				//add the strains up at each evaluation point
				cell_strain += node_strain_results[q];
			}

			//multiply the stress/strain tensor divide by the number of quadrature points
			//to get the average stress in the cell
			cell_stress.clear();
			cell_stress = (stress_strain_tensor*cell_strain)/n_node_q_points;
			//assign each stress to the right cell
			sigma_cells[cell_index] = cell_stress;
			vonMisesCells(cell_index) = get_vonMisesStress(cell_stress);
			cell_index++;
		}

		//compute actual nodal stress through simple averaging and compute von Mises
		for(types::global_dof_index i=0;i<n_dofs;i+=dim)
		{
				sigma_nodes[i] = sigma_nodes[i]/count_quad_points_written[i];
				vonMisesNodes(i) = get_vonMisesStress(sigma_nodes[i]);
		}
		data_out.add_data_vector(vonMisesCells,"vonMises_cell_stresses");
		data_out.add_data_vector(vonMisesNodes,"vonMises_nodal_stresses");


		std::vector<Vector<double>> stresses_cells(number_stress_components,Vector<double>(num_cells));
		Vector<double> direct_stresses_nodes(n_dofs); //dim DOF
		Vector<double> shear_stresses_nodes(n_dofs);
		for(int k=0;k<num_cells;++k)
		{
			for(int l=0;l<number_stress_components;++l)
			{
				if(l<dim)
					stresses_cells[l](k)=sigma_cells[k][0][l];
				else if(l<(2*dim-1))
					stresses_cells[l](k)=sigma_cells[k][1][l-dim+1];
				else
					stresses_cells[l](k)=sigma_cells[k][2][2];
			}
		}
		for(int i=0;i<number_stress_components;++i)
			data_out.add_data_vector(stresses_cells[i],stresses_cell_names[i]);

		for(types::global_dof_index k=0;k<n_dofs;k+=dim)
		{
			for(int i=0;i<dim;++i)
			{
				direct_stresses_nodes(k+i) = sigma_nodes[k][i][i];
				for(int j=0;j<dim;++j)
					shear_stresses_nodes(k+j) = sigma_nodes[k][i][j];
			}
		}
		data_out.add_data_vector(direct_stresses_nodes,direct_stresses_nodes_names);
		data_out.add_data_vector(shear_stresses_nodes,shear_stresses_nodes_names);

		if(plot_deformed_configuration==true)
			data_out.build_patches(q_mapping,mapping_degree);
		else
			data_out.build_patches();
		data_out.write_vtk(sol_output);
		std::cout << "Cycle " << cycle << " written to file..." << std::endl;

	}

	template <int dim>
	void print_mesh_info(const Triangulation<dim> &tria,
	                     const std::string        &filename,
	                     const std::string			&gnufilename)
	{
		std::cout << "Mesh info:" << std::endl
				<< " dimension: " << dim << std::endl
				<< " no. of cells: " << tria.n_active_cells() << std::endl;
		std::map<unsigned int, unsigned int> boundary_count;
		typename Triangulation<dim>::active_cell_iterator
		cell = tria.begin_active(),
		endc = tria.end();
		for (; cell!=endc; ++cell)
		{
		  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
			{
			  if (cell->face(face)->at_boundary())
				boundary_count[cell->face(face)->boundary_id()]++;
			}
		}
		std::cout << " boundary indicators: ";
		for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
		   it!=boundary_count.end();
		   ++it)
		{
		  std::cout << it->first << "(" << it->second << " times) ";
		}
		std::cout << std::endl;
		std::ofstream out (filename.c_str()),gnu_out (gnufilename.c_str());
		GridOut grid_out;
		grid_out.write_eps (tria, out);
		GridOutFlags::Gnuplot	gnuplot_flags(true);
		grid_out.set_flags(gnuplot_flags);
		grid_out.write_gnuplot(tria,gnu_out);
		std::cout << " written to " << filename << " and " << gnufilename << std::endl;
	}

	template <int dim> void PlaneStrain<dim>::run()
	{

		for (unsigned int cycle=0; cycle<7; ++cycle)
		{
			std::cout << "Cycle " << cycle << ':' << std::endl;
			if (cycle == 0)
			  {
				GridGenerator::hyper_rectangle(triangulation,Point<dim>(0,0),Point<dim>(1,.1));
				triangulation.refine_global(1);
				std::cout << "Number of dof per cell:" << fe.dofs_per_cell << "." << std::endl;
				typename Triangulation<dim>::cell_iterator
				cell = triangulation.begin (),
				endc = triangulation.end();
				for (; cell!=endc; ++cell)
				{
					for (unsigned int face_number=0;face_number<GeometryInfo<dim>::faces_per_cell;++face_number)
					{
						if(cell->face(face_number)->at_boundary())
						{
							//set all boundary faces with x greater than zero to 1
							if(cell->face(face_number)->center()(0)>0.0)
								cell->face(face_number)->set_boundary_id (1);
							//set all boundary faces with y==.1 equal to 2
							if(std::fabs(cell->face(face_number)->center()(1)-.1) < 1e-10)
								cell->face(face_number)->set_boundary_id (2);
						}
					}
				}
				print_mesh_info(triangulation,"mesh_info.eps","cell_data");
			  }
			else
			  refine_grid ();

			setup_system();
			assemble_system ();
			solve ();
			output_results (cycle);
		  }
		  std::ofstream table_output_file("convergence_output.txt");
		  convergence_table.write_text(table_output_file);
		  table_output_file.close();
		  std::cout << "Mesh convergence table written." << std::endl;

	}

}

int main()
{
	CaseStudy02::PlaneStrain<2> beam_prob;
	try
	{
	  dealii::deallog.depth_console (0);
	  beam_prob.run ();
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
