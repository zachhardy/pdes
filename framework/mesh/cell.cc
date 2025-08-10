#include "framework/mesh/cell.h"
#include "framework/mesh/mesh.h"

namespace pdes
{
  void
  Cell::compute_geometric_info(const Mesh& mesh)
  {
    centroid_ = 0.0;
    for (const auto& v_id: vertex_ids_)
      centroid_ += mesh.vertex(v_id);
    centroid_ /= static_cast<double>(vertex_ids_.size());

    switch (type_)
    {
      case Type::SLAB:
      {
        const auto xl = mesh.vertex(vertex_ids_.at(0))(0);
        const auto xr = mesh.vertex(vertex_ids_.at(1))(0);

        switch (mesh.coordinate_system())
        {
          case Mesh::CoordinateSystem::CARTESIAN:
            volume_ = xr - xl;
            break;
          case Mesh::CoordinateSystem::CYLINDRICAL:
            volume_ = M_PI * (pow(xr, 2) - pow(xl, 2));
            break;
          case Mesh::CoordinateSystem::SPHERICAL:
            volume_ = 4.0 / 3.0 * M_PI * (pow(xr, 3) - pow(xl, 3));
            break;
        }
        break;
      }
      case Type::QUADRILATERAL:
      {
        const auto v0 = mesh.vertex(vertex_ids_.at(0));
        const auto v1 = mesh.vertex(vertex_ids_.at(1));
        const auto v2 = mesh.vertex(vertex_ids_.at(2));
        const auto v3 = mesh.vertex(vertex_ids_.at(3));

        const auto dx01 = v1 - v0;
        const auto dx02 = v2 - v0;
        const auto dx03 = v3 - v0;

        switch (mesh.coordinate_system())
        {
          case Mesh::CoordinateSystem::CARTESIAN:
          {
            const auto area1 = 0.5 * (dx01(0) * dx02(1) - dx02(0) * dx01(1));
            const auto area2 = 0.5 * (dx02(0) * dx03(1) - dx03(0) * dx02(1));
            volume_ = std::abs(area1 + area2);
            break;
          }
          default:
            throw std::runtime_error("Only Cartesian 2D is implemented.");
        }
        break;
      }
      default:
        throw std::runtime_error("Only slab and quadrilateral cells are implemented.");
    }

    unsigned int f = 0;
    for (auto& face: faces_)
      face.compute_geometric_data(mesh, f++);
  }
}
