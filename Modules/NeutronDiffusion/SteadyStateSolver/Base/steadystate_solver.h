#ifndef STEADYSTATE_SOLVER_H
#define STEADYSTATE_SOLVER_H

#include "NeutronDiffusion/boundaries.h"
#include "NeutronDiffusion/Groupset/groupset.h"

#include "Grid/mesh.h"
#include "Discretization/discretization.h"

#include "material.h"
#include "CrossSections/cross_sections.h"

#include "vector.h"
#include "matrix.h"

#include "LinearSolvers/linear_solver.h"


namespace neutron_diffusion
{

/// Algorithms to solve the multigroup diffusion problem.
enum class SolutionTechnique
{
  FULL_SYSTEM = 0,   ///< Solve the full multigroup system.
  GROUPSET_WISE = 1  ///< Iteratively solve by groupset.
};

//######################################################################

typedef math::LinearSolverType LinearSolverType;

struct Options
{
  double tolerance = 1.0e-10;
  size_t max_iterations = 100;
  LinearSolverType linear_solver_type = LinearSolverType::LU;
  SolutionTechnique solution_technique = SolutionTechnique::GROUPSET_WISE;
};

//######################################################################

/// A steady state solver for multigroup neutron diffusion applications.
class SteadyStateSolver
{
protected:
  const std::string solver_string = "diffusion::SteadyStateSolver::";

protected:
  typedef grid::Mesh Mesh;
  typedef math::DiscretizationMethod DiscretizationMethod;
  typedef math::Discretization Discretization;

  typedef physics::Material Material;
  typedef physics::MaterialPropertyType MaterialPropertyType;
  typedef physics::CrossSections CrossSections;
  typedef physics::IsotropicMultiGroupSource IsotropicMGSource;

  typedef std::vector<double> RobinBndryVals;
  typedef std::shared_ptr<Boundary> BndryPtr;

  typedef math::LinearSolverType LinearSolverType;
  typedef math::LinearSolver LinearSolver;

public:

  /*---------- Options ----------*/
  Options options;

  /*---------- General Information ----------*/
  size_t n_groups = 0;
  size_t n_precursors = 0;
  bool use_precursors = false;

  /*---------- Groupsets and Groups ----------*/
  std::vector<size_t> groups;
  std::vector<Groupset> groupsets;

  /*---------- Spatial Grid Information ----------*/
  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<Discretization> discretization;

  /*---------- Material Information ----------*/
public:
  std::vector<std::shared_ptr<Material>> materials;
  std::vector<std::shared_ptr<CrossSections>> material_xs;
  std::vector<std::shared_ptr<IsotropicMGSource>> material_src;

protected:
  /** The maximum number of precursors that live on a material.
   *  This is used to promote sparsity in the precursor vector for problems with
   *  many different materials and precursor sets, such as in burnup
   *  applications. */
  size_t max_precursors_per_material = 0;

  /** Map a material ID to a particular CrossSection object.
   *  This mapping alleviates the need to store multiple copies of the
   *  CrossSections objects when one property appears more than once. */
  std::vector<int> matid_to_xs_map;

  /** Map a material ID to a particular IsotropicMultiGroupSource object.
   *  This mapping alleviates the need to store multiple copies of the
   *  IsotropicMultiGroupSource objects when one property appears more than
   *  once. */
  std::vector<int> matid_to_src_map;

  /*---------- Boundary Information ----------*/
public:
  /** A list containing a pair with the boundary type and index corresponding
   *  to the location of the boundary values within the boundary values vector.
   *  This is similar to the matid_to_xs_map attribute. */
  std::vector<std::pair<BoundaryType, size_t>> boundary_info;

  /** The multigroup boundary values. The outer index corresponds to the
   *  boundary index, the middle to the group, and the last to the boundary
   *  value index. For non-Robin boundaries, this always has one entry at the
   *  innermost lever. For Robin boundaries, three entries in the order of
   *  <tt>(a, b, f)</tt> are used. */
  std::vector<std::vector<std::vector<double>>> boundary_values;

protected:
  /** The multigroup boundary conditions. This is a vector of vectors of
   *  pointers to Boundary objects. The outer indexing corresponds to the
   *  boundary index and the inner index to the group. These are created at
   *  solver initialization. */
  std::vector<std::vector<BndryPtr>> boundaries;

public:
  /*---------- Solutions ----------*/
  math::Vector phi;
  math::Vector precursors;

public:
  void initialize();
  void execute();
  void solve_groupset(Groupset& groupset);

protected:
  /// Virtual function for assembling a groupset matrix.
  virtual void assemble_matrix(Groupset& groupset) = 0;
  /// Virtual function for setting a groupset source.
  virtual void set_source(Groupset& groupset, math::Vector& b) = 0;

protected:
  void input_checks();

  void initialize_materials();
  void initialize_boundaries();

  /// Virtual function for creating a discretization.
  virtual void initialize_discretization() = 0;

protected:
  virtual void scoped_transfer(Groupset& groupset,
                               const math::Vector& x,
                               math::Vector& y) = 0;
};

}

#endif //STEADYSTATE_SOLVER_H