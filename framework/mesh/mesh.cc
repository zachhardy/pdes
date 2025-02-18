#include "framework/mesh/mesh.h"
#include "framework/mesh/cell.h"

namespace pdes
{
  Mesh::Mesh(const unsigned int dim, const CoordinateSystem coord_sys)
    : dim_(dim),
      coord_sys_(coord_sys)
  {
  }

  void
  Mesh::add_vertices(std::vector<MeshVector<>>&& vertices)
  {
    for (auto vertex: vertices)
      this->add_vertex(std::move(vertex));
  }

  void
  Mesh::add_cells(std::vector<Cell>&& cells)
  {
    for (auto& cell: cells)
      this->add_cell(std::move(cell));
  }
}
