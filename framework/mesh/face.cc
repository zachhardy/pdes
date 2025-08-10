#include "framework/mesh/mesh.h"
#include "framework/mesh/face.h"

#include <iostream>


namespace pdes
{
  void
  Face::compute_geometric_data(const Mesh& mesh, const unsigned int f)
  {
    centroid_ = 0.0;
    for (const auto& v_id: vertex_ids_)
      centroid_ += mesh.vertex(v_id);
    centroid_ /= static_cast<double>(vertex_ids_.size());

    if (vertex_ids_.size() == 1)
    {
      const auto r = centroid_(0);

      switch (mesh.coordinate_system())
      {
        case Mesh::CoordinateSystem::CARTESIAN:
          area_ = 1.0;
          break;
        case Mesh::CoordinateSystem::CYLINDRICAL:
          area_ = 2.0 * M_PI * r;
          break;
        case Mesh::CoordinateSystem::SPHERICAL:
          area_ = 4.0 * M_PI * r * r;
          break;
      }
      normal_ = MeshVector<>(f == 0 ? -1.0 : 1.0);
    }
    else if (vertex_ids_.size() == 2)
    {
      const auto& v0 = mesh.vertex(vertex_ids_[0]);
      const auto& v1 = mesh.vertex(vertex_ids_[1]);

      switch (mesh.coordinate_system())
      {
        case Mesh::CoordinateSystem::CARTESIAN:
          area_ = v1.distance(v0);
          break;
        default:
          throw std::runtime_error("Only Cartesian 2D is implemented.");
      }

      const auto khat = MeshVector<>(0.0, 0.0, 1.0);
      normal_ = (v1 - v0).cross(khat).direction();
    }
    else
      throw std::runtime_error("Only 1D and 2D Cartesian faces are implemented.");
  }
}
