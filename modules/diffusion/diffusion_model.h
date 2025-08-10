#pragma once
#include "framework/mesh/mesh.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"

namespace pdes
{
  class DiffusionModel
  {
  protected:
    using SourceFunction = std::function<types::real(const MeshVector<>&)>;

  public:
    DiffusionModel(types::real diffusion_coefficient,
                   const SourceFunction& source_function);

    types::real k() const { return diff_coeff_; }
    types::real q(const MeshVector<>& x) const { return q_func_(x); }

  protected:
    const types::real diff_coeff_;
    const std::function<types::real(const MeshVector<>&)> q_func_;
  };

  inline
  DiffusionModel::DiffusionModel(const double diffusion_coefficient,
                                 const SourceFunction& source_function)
    : diff_coeff_(diffusion_coefficient),
      q_func_(source_function)
  {
  }
}
