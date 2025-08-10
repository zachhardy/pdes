#pragma once
#include "modules/diffusion/diffusion_model.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/math/sparse_matrix.h"
#include "framework/math/vector.h"

namespace pdes
{
  class DiffusionSolver
  {
  public:
    DiffusionSolver(const DiffusionModel& model,
                    const std::shared_ptr<SpatialDiscretization>& discretization);

    void assemble();

  private:
    const DiffusionModel& model_;
    const std::shared_ptr<SpatialDiscretization> discretization_;

    SparseMatrix<> A_;
    Vector<> b_;
  };
}
