#pragma once
#include "framework/types.h"
#include "framework/numbers.h"
#include "framework/mesh/mesh_vector.h"
#include <vector>


namespace pdes
{
  class Mesh;

  class Face final
  {
  public:
    enum class Type
    {
      INTERNAL,
      BOUNDARY
    };

    Face(const Type type,
         const types::global_index neighbor_or_boundary_id,
         std::vector<types::global_index> vertex_ids)
      : type_(type),
        neighbor_id_(type == Type::INTERNAL
                       ? neighbor_or_boundary_id
                       : numbers::invalid_global_index),
        boundary_id_(type == Type::BOUNDARY
                       ? neighbor_or_boundary_id
                       : numbers::invalid_unsigned_int),
        vertex_ids_(std::move(vertex_ids)),
        area_(0)
    {}

    Face(const Face&) = default;
    Face(Face&&) noexcept = default;
    ~Face() = default;

    Face& operator=(const Face&) = delete;
    Face& operator=(Face&&) noexcept = delete;

    bool has_neighbor() const { return type_ == Type::INTERNAL; }
    bool is_boundary() const { return type_ == Type::BOUNDARY; }

    types::global_index neighbor_id() const { return neighbor_id_; }
    unsigned int boundary_id() const { return boundary_id_; }

    const MeshVector<>& centroid() const { return centroid_; }
    const MeshVector<>& normal() const { return normal_; }
    types::real area() const { return area_; }

    types::global_index vertex_ids(const unsigned int vid) const { return vertex_ids_.at(vid); }
    const std::vector<types::global_index>& vertex_ids() const { return vertex_ids_; }

    void compute_geometric_data(const Mesh& mesh);

  private:
    const Type type_;
    types::global_index neighbor_id_;
    unsigned int boundary_id_;

    std::vector<types::global_index> vertex_ids_;

    MeshVector<> centroid_;
    MeshVector<> normal_;
    types::real area_;
  };
}
