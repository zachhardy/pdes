#include "framework/mesh/cell.h"

namespace pdes
{
  Cell::Cell(const Type type, const Type sub_type)
    : type_(type),
      sub_type_(sub_type)
  {
  }

  void
  Cell::set_global_id(const types::global_index global_id)
  {
    global_id_ = global_id;
  }

  void
  Cell::set_partition_id(const unsigned int partition_id)
  {
    partition_id_ = partition_id;
  }


  void
  Cell::set_vertex_ids(std::vector<types::global_index>&& vertex_ids)
  {
    vertex_ids_ = std::move(vertex_ids);
  }

  types::global_index
  Cell::vertex_ids(const unsigned int vid) const
  {
    return vertex_ids_.at(vid);
  }


  const std::vector<types::global_index>&
  Cell::vertex_ids() const
  {
    return vertex_ids_;
  }

  void
  Cell::add_faces(std::vector<Face>&& faces)
  {
    for (auto& face: faces)
      faces_.push_back(std::move(face));
  }
}
