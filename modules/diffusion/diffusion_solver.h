#pragma once
#include "modules/diffusion/diffusion_model.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/sparse_matrix.h"
#include "framework/math/vector.h"
#include "framework/math/linear_solver/preconditioner/precondition_ssor.h"

namespace pdes
{
  class DiffusionSolver
  {
  public:
    DiffusionSolver(const DiffusionModel& model,
                    const std::shared_ptr<SpatialDiscretization>& discretization,
                    const std::shared_ptr<LinearSolver<SparseMatrix<>>>& solver,
                    const std::shared_ptr<Preconditioner<SparseMatrix<>>>& preconditioner);

    /**
     * Assemble the linear system and rebuild the preconditioner.
     *
     * Any call to this function must be followed by a preconditioner
     * rebuild before solving; this requirement is handled internally by
     * invoking the preconditioner's build routine at the end of the
     * assembly process.
     */
    void assemble();

    void solve();

  private:
    const DiffusionModel& model_;
    const std::shared_ptr<SpatialDiscretization> discretization_;

    std::shared_ptr<LinearSolver<SparseMatrix<>>> solver_;
    std::shared_ptr<Preconditioner<SparseMatrix<>>> preconditioner_;

    SparseMatrix<> A_;
    Vector<> b_, x_;
  };
}
