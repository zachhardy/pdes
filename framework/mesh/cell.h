#pragma once
#include "framework/types.h"
#include "framework/mesh/face.h"
#include "framework/mesh/mesh_vector.h"
#include <vector>


namespace pdes
{
  class Mesh;

  class Cell final
  {
  public:
    enum class Type
    {
      SLAB,
      POLYGON,
      TRIANGLE,
      QUADRILATERAL,
      HEXAHEDRON
    };

    Cell(const Type type,
         const Type sub_type,
         const types::global_index global_id,
         std::vector<types::global_index> vertex_ids,
         std::vector<Face> faces,
         const unsigned int block_id = 0,
         const unsigned int partition_id = 0)
      : type_(type),
        sub_type_(sub_type),
        global_id_(global_id),
        local_id_(numbers::invalid_unsigned_int),
        block_id_(block_id),
        partition_id_(partition_id),
        faces_(std::move(faces)),
        vertex_ids_(std::move(vertex_ids)),
        volume_(0)
    {}

    Cell(const Cell&) = default;
    Cell(Cell&&) noexcept = default;
    ~Cell() = default;

    Cell& operator=(const Cell&) = delete;
    Cell& operator=(Cell&&) noexcept = delete;

    Type type() const { return type_; }
    Type sub_type() const { return sub_type_; }

    unsigned int local_id() const { return local_id_; }
    types::global_index global_id() const { return global_id_; }
    unsigned int block_id() const { return block_id_; }
    unsigned int partition_id() const { return partition_id_; }

    types::global_index vertex_ids(unsigned int vid) const { return vertex_ids_.at(vid); }
    const std::vector<types::global_index>& vertex_ids() const { return vertex_ids_; }

    const MeshVector<>& centroid() const { return centroid_; }
    types::real volume() const { return volume_; }

    const Face& faces(const unsigned int f) const { return faces_.at(f); }
    const std::vector<Face>& faces() const { return faces_; }

    void compute_geometric_info(const Mesh& mesh);

  private:
    void set_local_id(const unsigned int local_id) { local_id_ = local_id; }
    void set_block_id(const unsigned int block_id) { block_id_ = block_id; }
    void set_partition_id(const unsigned int partition_id) { partition_id_ = partition_id; }

    const Type type_;
    const Type sub_type_;

    types::global_index global_id_;
    unsigned int local_id_;
    unsigned int block_id_;
    unsigned int partition_id_;

    std::vector<Face> faces_;
    std::vector<types::global_index> vertex_ids_;

    MeshVector<> centroid_;
    types::real volume_;

    friend class OrthoMeshGenerator;
  };
}
