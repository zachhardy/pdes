#pragma once
#include "framework/mesh/mesh.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"

namespace pdes
{
  class DiffusionModel
  {
  public:
    struct BoundaryCondition
    {
      enum class Type { DIRICHLET, NEUMANN, ROBIN };
      struct Dirichlet { types::real f; };
      struct Neumann { types::real g; };
      struct Robin { types::real a, b, f; };

      Type type;
      Dirichlet dirichlet;
      Neumann neumann;
      Robin robin;
    };

  protected:
    using SourceFunction = std::function<types::real(const MeshVector<>&)>;

  public:
    DiffusionModel(types::real diffusion_coefficient,
                   const SourceFunction& source_function,
                   const std::map<unsigned int, BoundaryCondition>& bcs);

    types::real k() const { return diff_coeff_; }
    types::real q(const MeshVector<>& x) const { return q_func_(x); }
    const BoundaryCondition& bc(const unsigned int bid) const { return bcs_.at(bid); }

  protected:
    const types::real diff_coeff_;
    const std::function<types::real(const MeshVector<>&)> q_func_;
    const std::map<unsigned int, BoundaryCondition> bcs_;
  };

  inline
  DiffusionModel::DiffusionModel(const double diffusion_coefficient,
                                 const SourceFunction& source_function,
                                 const std::map<unsigned int, BoundaryCondition>& bcs)
    : diff_coeff_(diffusion_coefficient),
      q_func_(source_function),
      bcs_(bcs)
  {
  }
}
