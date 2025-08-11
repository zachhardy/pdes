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

    void assemble();
    void solve();

    const Vector<>& solution() const { return x_; }
    Vector<>& solution() { return x_; }

  private:
    const DiffusionModel& model_;
    const std::shared_ptr<SpatialDiscretization> discretization_;

    std::shared_ptr<LinearSolver<SparseMatrix<>>> solver_;
    std::shared_ptr<Preconditioner<SparseMatrix<>>> preconditioner_;

    SparseMatrix<> A_;
    Vector<> b_, x_;
  };
}
