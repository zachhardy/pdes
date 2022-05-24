#include "steadystate_solver.h"

#include "LinearSolvers/Direct/lu.h"
#include "LinearSolvers/Direct/cholesky.h"
#include "LinearSolvers/Direct/sparse_lu.h"
#include "LinearSolvers/Direct/sparse_cholesky.h"

#include <iomanip>
#include <fstream>

using namespace pdes;

/// Run the steady state multigroup diffusion simulation.
void NeutronDiffusion::SteadyStateSolver:: execute()
{
  std::cout << "Executing solver...\n";

  // Initialize matrices
  for (auto& gs : groupsets)
   assemble_matrix(gs);

  SourceFlags source_flags = APPLY_MATERIAL_SOURCE;
  if (solution_technique == SolutionTechnique::GROUPSET_WISE)
    source_flags = source_flags | APPLY_WGS_SCATTER_SOURCE |
                   APPLY_AGS_SCATTER_SOURCE | APPLY_WGS_FISSION_SOURCE |
                   APPLY_AGS_FISSION_SOURCE;

  //======================================== Loop over groupsets
  for (auto& groupset : groupsets)
    solve_groupset(groupset, source_flags);

  std::cout << "\nDone executing solver.\n";
}

//######################################################################

/// Converge the system for the current groupset, lagging couplings from others.
void NeutronDiffusion::SteadyStateSolver::
solve_groupset(Groupset& groupset, SourceFlags source_flags)
{
  std::cout << "\n***** Solving Groupset " << groupset.id << "\n\n";

  double change = 1.0;
  bool converged = false;

  Math::SparseCholesky solver = groupset.matrix;
  solver.factorize();

  //======================================== Start iterations
  for (size_t nit = 0; nit < groupset.max_iterations; ++nit)
  {
    // Compute the RHS and solve
    groupset.rhs *= 0.0;
    set_source(groupset, groupset.rhs, source_flags);
    auto x = solver.solve(groupset.rhs);

    // Convergence check, finalize iteration
    scoped_transfer(groupset, x, phi);
    change = compute_change(groupset);
    scoped_copy(groupset, phi, phi_ell);

    if (change < groupset.tolerance)
      converged = true;

    // Print iteration information
    std::stringstream iter_info;
    iter_info << "Iteration: " << std::setw(3) << nit << " "
              << "Change: " << change;
    if (converged) iter_info << " CONVERGED";
    std::cout << iter_info.str() << "\n";

    if (converged) break;
  }
}
