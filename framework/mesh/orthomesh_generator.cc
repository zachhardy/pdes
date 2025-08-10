#include "framework/mesh/orthomesh_generator.h"

#include "framework/mesh/mesh.h"
#include "framework/mesh/cell.h"
#include "framework/mesh/face.h"
#include "framework/mesh/mesh_vector.h"

namespace pdes
{
  namespace
  {
    // Face ordering for line elements (deal.II / Gmsh / VTK convention):
    //
    //   Face 0: -X (left endpoint)  → vertex 0
    //   Face 1: +X (right endpoint) → vertex 1
    //
    // Local vertex indices for a line element:
    //
    //   0 -------- 1     (along +x direction)
    //
    // Notes:
    // - Each 1D cell (interval) has 2 vertices and 2 faces (endpoints)
    // - Face 0 is at the left end (x_min), face 1 is at the right end (x_max)
    // - Boundary IDs follow direction convention: 0 = -X, 1 = +X
    struct FaceDescription1D
    {
      unsigned int cell_vertex_map;
      int neighbor_offset;
      unsigned int boundary_id;
    };

    constexpr std::array<FaceDescription1D, 2> table_1d = {
      {
        {0, -1, 0},
        {1, +1, 1}
      }
    };

    // Face ordering for quadrilateral cells (deal.II / Gmsh / VTK convention):
    //
    //   Face 0: -Y (bottom edge) → vertices 0 → 1
    //   Face 1: +X (right edge)  → vertices 1 → 2
    //   Face 2: +Y (top edge)    → vertices 2 → 3
    //   Face 3: -X (left edge)   → vertices 3 → 0
    //
    // Local vertex indices for a quadrilateral:
    //
    //       3 -------- 2      ↑ +Y (top)
    //       |          |
    //       |          |      → +X (right)
    //       0 -------- 1
    //
    // Notes:
    // - Vertices are ordered counter-clockwise starting at bottom-left (0)
    // - Face normals point outward from the cell
    // - Face order and orientation affect boundary ID assignment and BC logic
    struct FaceDescription2D
    {
      unsigned int cell_vertex_map[2];
      int neighbor_offset[2];
      unsigned int boundary_id;
    };

    constexpr std::array<FaceDescription2D, 4> table_2d = {
      {
        {{0, 1}, {0, -1}, 0},
        {{1, 2}, {+1, 0}, 1},
        {{2, 3}, {0, +1}, 2},
        {{3, 0}, {-1, 0}, 3}
      }
    };

    // Face ordering for hexahedra (deal.II / Gmsh / VTK convention):
    //
    //   Face 0: -Z (bottom) → vertices 0,1,2,3
    //   Face 1: +Z (top)    → vertices 4,5,6,7
    //   Face 2: -Y (back)   → vertices 0,1,5,4
    //   Face 3: +Y (front)  → vertices 3,2,6,7
    //   Face 4: -X (left)   → vertices 0,3,7,4
    //   Face 5: +X (right)  → vertices 1,2,6,5
    //
    // Local vertex indices for a hexahedron:
    //         7--------6       z+ (top)
    //        /|       /|
    //       4--------5 |        ^
    //       | |      | |        |
    //       | 3------|-2        +--→ y+ (front)
    //       |/       |/        /
    //       0--------1        x+ (right)
    //
    // Notes:
    // - Vertices 0–3 lie in the z=0 (bottom) plane, ordered CCW from (0,0,0)
    // - Vertices 4–7 are directly above 0–3, in the z=1 (top) plane
    // - Face normals point outward from the hexahedron
    // - Face order matters for consistent integration and BC application
    struct FaceDescription3D
    {
      unsigned int cell_vertex_map[4];
      int neighbor_offset[3];
      unsigned int boundary_id;
    };

    constexpr std::array<FaceDescription3D, 6> table_3d = {
      {
        {{0, 1, 2, 3}, {0, 0, -1}, 0},
        {{4, 5, 6, 7}, {0, 0, +1}, 1},
        {{0, 1, 5, 4}, {0, -1, 0}, 2},
        {{3, 2, 6, 7}, {0, +1, 0}, 3},
        {{0, 3, 7, 4}, {-1, 0, 0}, 4},
        {{1, 2, 6, 5}, {+1, 0, 0}, 5}
      }
    };
  }


  Mesh
  OrthoMeshGenerator::create_1d_orthomesh(const std::vector<types::real>& x_coords,
                                          const Mesh::CoordinateSystem coord_sys)
  {
    const auto nv = x_coords.size();
    const auto nc = nv - 1;

    // Initialize the mesh
    Mesh mesh(1, coord_sys);
    mesh.set_ortho_attributes(nc);

    // Create vertices
    for (types::global_index i = 0; i < nv; ++i)
    {
      if (i > 0 and x_coords[i] <= x_coords[i - 1])
        throw std::logic_error("The x-coordinates must be in ascending order.");
      mesh.add_vertex(i, MeshVector(x_coords.at(i)));
    }

    // Define the mesh
    for (types::global_index c = 0; c < nc; ++c)
    {
      const std::vector vertex_ids = {c, c + 1};

      std::vector<Face> faces;
      for (int f = 0; f < 2; ++f)
      {
        const auto cvid = table_1d[f].cell_vertex_map;
        const auto nbr_offset = table_1d[f].neighbor_offset;
        const auto bid = table_1d[f].boundary_id;

        const std::vector face_vertex_ids({vertex_ids[cvid]});
        const int neighbor_id = static_cast<int>(c) + nbr_offset;
        if (neighbor_id < 0 or neighbor_id >= nc)
          faces.emplace_back(Face::Type::BOUNDARY, bid, face_vertex_ids);
        else
          faces.emplace_back(Face::Type::INTERNAL, neighbor_id, face_vertex_ids);
      }

      Cell cell(Cell::Type::SLAB,
                Cell::Type::SLAB,
                c,
                vertex_ids,
                faces);

      cell.set_local_id(c);
      cell.compute_geometric_info(mesh);
      mesh.add_cell(std::move(cell));
    }
    return mesh;
  }

  Mesh
  OrthoMeshGenerator::create_1d_orthomesh(const std::vector<types::real>& region_edges,
                                          const std::vector<unsigned int>& cells_per_regions,
                                          const std::vector<unsigned int>& block_ids,
                                          const Mesh::CoordinateSystem coord_sys)
  {
    const auto nr = cells_per_regions.size();

    if (region_edges.size() - 1 != nr)
      throw std::logic_error("Incompatible region edges and cell counts.");

    // Define the vertices on the mesh
    std::vector<types::real> verts;
    for (unsigned int r = 0; r < nr; ++r)
    {
      const auto width = region_edges[r + 1] - region_edges[r];
      const auto dx = width / static_cast<types::real>(cells_per_regions[r]);
      for (unsigned int c = 0; c < cells_per_regions[r]; ++c)
        verts.emplace_back(region_edges[r] + c * dx);
    }
    verts.emplace_back(region_edges.back());

    // Create a mesh from the vertices
    auto mesh = create_1d_orthomesh(verts, coord_sys);

    // Assign block ids to each region
    unsigned int idx = 0;
    for (unsigned int r = 0; r < nr; ++r)
      for (unsigned int c = 0; c < cells_per_regions[r]; ++c)
        mesh.local_cell(idx++).set_block_id(block_ids[r]);

    return mesh;
  }

  Mesh
  OrthoMeshGenerator::create_2d_orthomesh(const std::vector<types::real>& x_coords,
                                          const std::vector<types::real>& y_coords,
                                          const Mesh::CoordinateSystem coord_sys)
  {
    const auto nxv = x_coords.size();
    const auto nyv = y_coords.size();

    const auto nxc = nxv - 1;
    const auto nyc = nyv - 1;

    const auto vertex_id = [&](const int i, const int j) {
      return static_cast<types::global_index>(j * nxv + i);
    };
    const auto cell_id = [&](const int i, const int j) {
      return static_cast<types::global_index>(j * nxc + i);
    };

    // Initialize the mesh
    Mesh mesh(2, coord_sys);
    mesh.set_ortho_attributes(nxc, nyc);

    // Add vertices to the mesh
    for (unsigned int j = 0; j < nyv; ++j)
      for (unsigned int i = 0; i < nxv; ++i)
      {
        if (i > 0 and x_coords[i] <= x_coords[i - 1])
          throw std::logic_error("The x-coordinates must be in ascending order.");
        if (j > 0 and y_coords[j] <= y_coords[j - 1])
          throw std::logic_error("The y-coordinates must be in ascending order.");

        const MeshVector vertex(x_coords.at(i), y_coords.at(j));
        mesh.add_vertex(vertex_id(i, j), vertex);
      }

    // Create cells ordered in the same manner. The cell vertices are
    // ordered counter-clockwise from the bottom left vertex.  The faces
    // are ordered similarly from the bottom face.
    for (unsigned int j = 0; j < nyc; ++j)
      for (unsigned int i = 0; i < nxc; ++i)
      {
        // Vertices are defined CCW: bl-br-tr-tl
        std::vector vertex_ids({
          vertex_id(i, j),
          vertex_id(i + 1, j),
          vertex_id(i + 1, j + 1),
          vertex_id(i, j + 1)
        });

        // Define the faces
        std::vector<Face> faces;
        for (int f = 0; f < 4; ++f)
        {
          const std::vector face_vertex_ids = {
            vertex_ids[table_2d[f].cell_vertex_map[0]],
            vertex_ids[table_2d[f].cell_vertex_map[1]]
          };

          const int ni = static_cast<int>(i) + table_2d[f].neighbor_offset[0];
          const int nj = static_cast<int>(j) + table_2d[f].neighbor_offset[1];
          if (ni < 0 or ni >= nxc or nj < 0 or nj >= nyc)
            faces.emplace_back(Face::Type::BOUNDARY, table_2d[f].boundary_id, face_vertex_ids);
          else
            faces.emplace_back(Face::Type::INTERNAL, cell_id(ni, nj), face_vertex_ids);
        }

        Cell cell(Cell::Type::QUADRILATERAL,
                  Cell::Type::QUADRILATERAL,
                  cell_id(i, j),
                  vertex_ids,
                  faces);

        cell.set_local_id(cell_id(i, j));
        cell.compute_geometric_info(mesh);
        mesh.add_cell(std::move(cell));
      }

    return mesh;
  }

  Mesh
  OrthoMeshGenerator::create_2d_orthomesh(const std::vector<types::real>& x_region_edges,
                                          const std::vector<types::real>& y_region_edges,
                                          const std::vector<unsigned int>& cells_per_x_region,
                                          const std::vector<unsigned int>& cells_per_y_region,
                                          const std::vector<unsigned int>& block_ids,
                                          const Mesh::CoordinateSystem coord_sys)
  {
    const auto nxr = cells_per_x_region.size();
    const auto nyr = cells_per_y_region.size();

    if (x_region_edges.size() - 1 != nxr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in x-dimension.");
    if (y_region_edges.size() - 1 != nyr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in y-dimension.");
    if (block_ids.size() != nxr * nyr)
      throw std::logic_error("Incompatible number of regions and block IDs.");

    std::vector<types::real> x_verts;
    for (unsigned int r = 0; r < nxr; ++r)
    {
      const auto width = x_region_edges[r + 1] - x_region_edges[r];
      const auto dx = width / static_cast<types::real>(cells_per_x_region[r]);
      for (unsigned int c = 0; c < cells_per_x_region[r]; ++c)
        x_verts.emplace_back(x_region_edges[r] + c * dx);
    }
    x_verts.emplace_back(x_region_edges.back());

    std::vector<types::real> y_verts;
    for (unsigned int r = 0; r < nyr; ++r)
    {
      const auto width = y_region_edges[r + 1] - y_region_edges[r];
      const auto dy = width / static_cast<types::real>(cells_per_y_region[r]);
      for (unsigned int c = 0; c < cells_per_y_region[r]; ++c)
        y_verts.emplace_back(y_region_edges[r] + c * dy);
    }
    y_verts.emplace_back(y_region_edges.back());

    // Create a mesh from the verticles
    auto mesh = create_2d_orthomesh(x_verts, y_verts, coord_sys);

    // Assign block ids to each region
    unsigned int idx = 0;
    for (unsigned int ry = 0; ry < nyr; ++ry)
      for (unsigned int cy = 0; cy < cells_per_y_region[ry]; ++cy)
        for (unsigned int rx = 0; rx < nxr; ++rx)
          for (unsigned int cx = 0; cx < cells_per_x_region[rx]; ++cx)
            mesh.local_cell(idx++).set_block_id(block_ids[ry * nxr + rx]);

    return mesh;
  }

  Mesh
  OrthoMeshGenerator::create_3d_orthomesh(const std::vector<types::real>& x_coords,
                                          const std::vector<types::real>& y_coords,
                                          const std::vector<types::real>& z_coords,
                                          const Mesh::CoordinateSystem coord_sys)
  {
    const auto nxv = x_coords.size();
    const auto nyv = y_coords.size();
    const auto nzv = z_coords.size();

    const auto nxc = nxv - 1;
    const auto nyc = nyv - 1;
    const auto nzc = nzv - 1;

    const auto vertex_id = [&](const int i, const int j, const int k) {
      return static_cast<types::global_index>((k * nyv + j) * nxv + i);
    };
    const auto cell_id = [&](const int i, const int j, const int k) {
      return static_cast<types::global_index>((k * nyc + j) * nxc + i);
    };

    // Initialize the mesh
    Mesh mesh(3, coord_sys);
    mesh.set_ortho_attributes(nxc, nyc, nzc);

    // Create the vertices
    for (unsigned int k = 0; k < nzv; ++k)
      for (unsigned int j = 0; j < nyv; ++j)
        for (unsigned int i = 0; i < nxv; ++i)
        {
          if (i > 0 and x_coords[i] <= x_coords[i - 1])
            throw std::logic_error("The x-coordinates must be in ascending order.");
          if (j > 0 and y_coords[j] <= y_coords[j - 1])
            throw std::logic_error("The y-coordinates must be in ascending order.");
          if (k > 0 and z_coords[k] <= z_coords[k - 1])
            throw std::logic_error("The k-coordinates must be in ascending order.");

          MeshVector vertex(x_coords.at(i), y_coords.at(j), z_coords.at(k));
          mesh.add_vertex(vertex_id(i, j, k), std::move(vertex));
        }

    // Create cells ordered in the same manner. The cell vertices are
    // ordered counter-clockwise from the bottom left vertex.  The faces
    // are ordered similarly from the bottom face.
    for (unsigned int k = 0; k < nzc; ++k)
      for (unsigned int j = 0; j < nyc; ++j)
        for (unsigned int i = 0; i < nxc; ++i)
        {
          const std::vector vertex_ids = {
            vertex_id(i, j, k),
            vertex_id(i + 1, j, k),
            vertex_id(i + 1, j + 1, k),
            vertex_id(i, j + 1, k),
            vertex_id(i, j, k + 1),
            vertex_id(i + 1, j, k + 1),
            vertex_id(i + 1, j + 1, k + 1),
            vertex_id(i, j + 1, k + 1)
          };

          // Define the faces
          std::vector<Face> faces;
          for (int f = 0; f < 6; ++f)
          {
            const std::vector face_vertex_ids = {
              vertex_ids[table_3d[f].cell_vertex_map[0]],
              vertex_ids[table_3d[f].cell_vertex_map[1]],
              vertex_ids[table_3d[f].cell_vertex_map[2]],
              vertex_ids[table_3d[f].cell_vertex_map[3]],
            };

            const int ni = static_cast<int>(i) + table_3d[f].neighbor_offset[0];
            const int nj = static_cast<int>(j) + table_3d[f].neighbor_offset[1];
            const int nk = static_cast<int>(k) + table_3d[f].neighbor_offset[2];
            if (ni < 0 or ni >= nxc or nj < 0 or nj >= nyc or nk < 0 or nk >= nzc)
              faces.emplace_back(Face::Type::BOUNDARY, table_3d[f].boundary_id, face_vertex_ids);
            else
              faces.emplace_back(Face::Type::INTERNAL, cell_id(ni, nj, nk), face_vertex_ids);
          }

          Cell cell(Cell::Type::HEXAHEDRON,
                    Cell::Type::HEXAHEDRON,
                    cell_id(i, j, k),
                    vertex_ids,
                    faces);

          cell.set_local_id(cell_id(i, j, k));
          cell.compute_geometric_info(mesh);
          mesh.add_cell(std::move(cell));
        } // for i,j,k cell

    return mesh;
  }

  Mesh
  OrthoMeshGenerator::create_3d_orthomesh(const std::vector<types::real>& x_region_edges,
                                          const std::vector<types::real>& y_region_edges,
                                          const std::vector<types::real>& z_region_edges,
                                          const std::vector<unsigned int>& cells_per_x_region,
                                          const std::vector<unsigned int>& cells_per_y_region,
                                          const std::vector<unsigned int>& cells_per_z_region,
                                          const std::vector<unsigned int>& block_ids,
                                          const Mesh::CoordinateSystem coord_sys)
  {
    const auto nxr = cells_per_x_region.size();
    const auto nyr = cells_per_y_region.size();
    const auto nzr = cells_per_z_region.size();

    if (x_region_edges.size() - 1 != nxr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in x-dimension.");
    if (y_region_edges.size() - 1 != nyr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in y-dimension.");
    if (z_region_edges.size() - 1 != nzr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in z-dimension.");
    if (block_ids.size() != nxr * nyr * nzr)
      throw std::logic_error("Incompatible number of regions and block IDs.");

    std::vector<types::real> x_verts;
    for (unsigned int r = 0; r < nxr; ++r)
    {
      const auto width = x_region_edges[r + 1] - x_region_edges[r];
      const auto dx = width / static_cast<types::real>(cells_per_x_region[r]);
      for (unsigned int c = 0; c < cells_per_x_region[r]; ++c)
        x_verts.emplace_back(x_region_edges[r] + c * dx);
    }
    x_verts.emplace_back(x_region_edges.back());

    std::vector<types::real> y_verts;
    for (unsigned int r = 0; r < nyr; ++r)
    {
      const auto width = y_region_edges[r + 1] - y_region_edges[r];
      const auto dy = width / static_cast<types::real>(cells_per_y_region[r]);
      for (unsigned int c = 0; c < cells_per_y_region[r]; ++c)
        y_verts.emplace_back(y_region_edges[r] + c * dy);
    }
    y_verts.emplace_back(y_region_edges.back());

    std::vector<types::real> z_verts;
    for (unsigned int r = 0; r < nzr; ++r)
    {
      const auto width = z_region_edges[r + 1] - z_region_edges[r];
      const auto dz = width / static_cast<types::real>(cells_per_z_region[r]);
      for (unsigned int c = 0; c < cells_per_z_region[r]; ++c)
        z_verts.emplace_back(z_region_edges[r] + c * dz);
    }
    z_verts.emplace_back(z_region_edges.back());

    // Create a mesh from the vertices
    auto mesh = create_3d_orthomesh(x_verts, y_verts, z_verts, coord_sys);

    // Assign block ids to each region
    unsigned int idx = 0;
    for (unsigned int rz = 0; rz < nzr; ++rz)
      for (unsigned int cz = 0; cz < cells_per_z_region[rz]; ++cz)
        for (unsigned int ry = 0; ry < nyr; ++ry)
          for (unsigned int cy = 0; cy < cells_per_y_region[ry]; ++cy)
            for (unsigned int rx = 0; rx < nxr; ++rx)
              for (unsigned int cx = 0; cx < cells_per_x_region[rx]; ++cx)
              {
                const auto bid = (rz * nyr + ry) * nxr + rx;
                mesh.local_cell(idx++).set_block_id(block_ids[bid]);
              }

    return mesh;
  }
}
