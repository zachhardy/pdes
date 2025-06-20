#include "framework/mesh/face.h"


namespace pdes
{
  void
  Face::set_neighbor_id(const types::global_index neighbor_id)
  {
    has_neighbor_ = true;
    neighbor_id_ = neighbor_id;
  }

  void
  Face::set_boundary_id(const unsigned int boundary_id)
  {
    has_neighbor_ = false;
    boundary_id_ = boundary_id;
  }

  void
  Face::set_vertex_ids(std::vector<types::global_index> vertex_ids)
  {
    vertex_ids_ = std::move(vertex_ids);
  }

  types::global_index
  Face::vertex_ids(const unsigned int vid) const
  {
    return vertex_ids_.at(vid);
  }

  const std::vector<types::global_index>&
  Face::vertex_ids() const
  {
    return vertex_ids_;
  }
}
