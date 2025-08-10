#pragma once
#include "framework/mesh/mesh.h"

namespace pdes
{
  enum class SpatialDiscretizationType
  {
    FINITE_DIFFERENCE = 0,
    FINITE_VOLUME = 1,
    CONTINUOUS_FINITE_ELEMENT = 2,
    DISCONTINUOUS_FINITE_ELEMENT = 3,
    UNDEFINED = 4
  };


  class SpatialDiscretization
  {
  public:
    SpatialDiscretization(const Mesh& mesh,
                          const SpatialDiscretizationType type)
      : mesh_(std::make_unique<Mesh>(mesh)),
        type_(type)
    {
    }

    virtual ~SpatialDiscretization() = default;

    const std::shared_ptr<Mesh>& mesh() const { return mesh_; }
    SpatialDiscretizationType type() const { return type_; }

    virtual unsigned int num_local_nodes() const = 0;
    virtual unsigned int num_cell_nodes(const Cell& cell) const = 0;
    virtual std::vector<MeshVector<>> cell_node_locations(const Cell& cell) const = 0;

  protected:
    const std::shared_ptr<Mesh> mesh_;
    const SpatialDiscretizationType type_;
  };
}
