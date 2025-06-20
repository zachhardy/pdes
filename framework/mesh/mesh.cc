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
                   MeshVector<>&& vertex,
                   const bool allow_replace)
  {
    if (not allow_replace and vertices_.count(i) > 0)
      throw std::runtime_error(
        "Vertex with global ID " + std::to_string(i) + " already exists");

    vertices_.emplace(i, std::move(vertex));
  }

  void
  Mesh::add_vertices(std::map<types::global_index, MeshVector<>>&& vertices,
                     bool allow_replace)
  {
    for (auto& [i, vertex]: vertices)
      this->add_vertex(i, std::move(vertex), allow_replace);
  }

  MeshVector<>&
  Mesh::vertices(const types::global_index i)
  {
    return vertices_.at(i);
  }

  const MeshVector<>&
  Mesh::vertices(const types::global_index i) const
  {
    return vertices_.at(i);
  }

  std::map<types::global_index, MeshVector<>>&
  Mesh::vertices()
  {
    return vertices_;
  }

  const std::map<types::global_index, MeshVector<>>&
  Mesh::vertices() const
  {
    return vertices_;
  }

  void
  Mesh::add_cells(std::vector<Cell>&& cells)
  {
    for (auto& cell: cells)
      this->add_cell(std::move(cell));
  }

  Cell&
  Mesh::cells(const unsigned int local_id)
  {
    return local_cells_.at(local_id);
  }

  const Cell&
  Mesh::cells(const unsigned int local_id) const
  {
    return local_cells_.at(local_id);
  }
}
