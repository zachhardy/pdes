#include "framework/mesh/mesh.h"
#include "framework/mesh/cell.h"

namespace pdes
{
  void
  Mesh::set_ortho_attributes(const unsigned int nx,
                             const unsigned int ny,
                             const unsigned int nz)
  {
    orthogonal_ = true;
    ortho_attr_.nx = nx;
    ortho_attr_.ny = ny;
    ortho_attr_.nz = nz;
  }

  void
  Mesh::add_vertex(types::global_index i,
                   MeshVector<> vertex,
                   const bool allow_replace)
  {
    if (not allow_replace and vertices_.count(i) > 0)
      throw std::runtime_error(
        "Vertex with global ID " + std::to_string(i) + " already exists");

    vertices_.emplace(i, std::move(vertex));
  }

  void
  Mesh::add_vertices(VertexMap vertices, const bool allow_replace)
  {
    for (auto& [i, vertex]: vertices)
      this->add_vertex(i, std::move(vertex), allow_replace);
  }

  void
  Mesh::add_cells(std::vector<Cell> cells)
  {
    for (auto& cell: cells)
      this->add_cell(std::move(cell));
  }
}
