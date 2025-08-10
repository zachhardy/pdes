#pragma once
#include "framework/math/spatial_discretization/spatial_discretization.h"

namespace pdes
{
  class FiniteVolume final : public SpatialDiscretization
  {
  public:
    explicit FiniteVolume(const Mesh& mesh);
    ~FiniteVolume() override = default;

    unsigned int num_local_nodes() const override { return mesh_->num_local_cells(); }
    unsigned int num_cell_nodes(const Cell&) const override { return 1; }
    std::vector<MeshVector<>> cell_node_locations(const Cell& cell) const override;
  };

  /* -------------------- inline definitions -------------------- */

  inline
  FiniteVolume::FiniteVolume(const Mesh& mesh)
    : SpatialDiscretization(mesh, SpatialDiscretizationType::FINITE_VOLUME)
  {
  }

  inline std::vector<MeshVector<>>
  FiniteVolume::cell_node_locations(const Cell& cell) const
  {
    return {cell.centroid()};
  }
}
