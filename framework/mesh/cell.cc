#include "framework/mesh/cell.h"

namespace pdes
{
  Cell::Cell(const Type type, const Type sub_type)
    : type_(type),
      sub_type_(sub_type)
  {
  }

  void
  Cell::add_faces(std::vector<Face>&& faces)
  {
    for (auto& face: faces)
      faces_.push_back(std::move(face));
  }
}
