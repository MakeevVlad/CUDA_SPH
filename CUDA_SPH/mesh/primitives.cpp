#ifndef PRIMITIVES_CPP
#define PRIMITIVES_CPP

#include "primitives.h"
#include "../info.cuh"
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <limits>
#include <unordered_map>


// Utils
std::string get_file_extension(std::string fname) {
  int n = fname.rfind(".");
  if(n == std::string::npos) {
    return "";
  } else {
    return fname.substr(n+1);
  };
};

void skip_to_tag(std::istream& in, std::string tag) {
  std::string line;
  while((line != tag) && !in.eof()) {
    std::getline(in, line);
  };
};

void skip_lines(std::istream& in, int n) {
  std::string line;
  for(int i = 0; i < n; ++i) {
    std::getline(in, line);
  };
};

// Point
Point::Point() : x{std::vector<real_t>(3)} {};
Point::Point(real_t x, real_t y, real_t z) : x{x, y, z} {};

real_t& Point::operator[](int i) {
  return x[i];
};

std::istream& Point::operator>>(std::istream& in) {
  in >> x[0] >> x[1] >> x[2];
  return in;
};


// Triangle
Triangle::Triangle() : points{0, 0, 0} {};
Triangle::Triangle(index_t p1, index_t p2, index_t p3) : points{p1, p2, p3} {};

std::istream& Triangle::operator>>(std::istream& in) {
  in >> points[0] >> points[1] >> points[2];
  return in;
};

index_t& Triangle::operator[](int i) {
  return points[i];
};


// Tetrahedra
Tetrahedra::Tetrahedra() : 
    points{index_t(), index_t(), index_t(), index_t()}, 
    material(index_t()) {};

Tetrahedra::Tetrahedra(index_t p1, index_t p2, index_t p3, 
                       index_t p4, index_t material) : 
    points{p1, p2, p3, p4}, material(material) {};

index_t& Tetrahedra::operator[](int i) {
  return points[i];
};

index_t Tetrahedra::get_material_id() {
  return material;
};


// SurfaceMesh
SurfaceMesh::SurfaceMesh() = default;

void SurfaceMesh::construct_from_file(std::string fname) {
  std::string ext = get_file_extension(fname);
  if(ext == "off") {
    std::ifstream f_in(fname);
    if(!f_in.is_open()) {
      std::cout << "[ERROR]: File \"" << fname << "\" not found\n";
      throw "File not found";
    };
    f_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    int node_count, faces_count, edge_count;
    f_in >> node_count >> faces_count >> edge_count;
    f_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    pts.reserve(node_count);
    for(int i = 0; i < node_count; ++i) {
      double x, y, z;
      f_in >> x >> y >> z;
      pts.push_back(Point(x, y, z));
    };

    trs.reserve(faces_count);
    for(int i = 0; i < faces_count; ++i) {
      int p1, p2, p3, dim;
      f_in >> dim >> p1 >> p2 >> p3;
      trs.push_back(Triangle(p1, p2, p3));
    };
  } else {
    std::cout << "[ERROR]: Wrong file format(" << ext <<
          ") for surface mesh providen\n";
  };
};

index_t SurfaceMesh::get_triangles_count() {
  return trs.size();
};

void SurfaceMesh::get_triangle_as_array(real_t** tr, index_t id) {
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < 3; ++j) {
      tr[i][j] = pts[trs[id][i]][j];
    };
  };
};

void SurfaceMesh::get_all_triangles(real_t*** trarr, index_t N) {
  if(N > trs.size()) {
    std::cout << "[ERROR]: Request of " << N << " triangles, which is more than " <<
                  trs.size() << " existing in the mesh\n";
    throw "Request of too many triangles";
  };
  for(int i = 0; i < N; ++i) {
    get_triangle_as_array(trarr[i], i);
  };
};


// VolumeMesh
VolumeMesh::VolumeMesh() = default;

void VolumeMesh::construct_materials(std::string fmat) {
  std::ifstream mat_in(fmat);
  if(!mat_in.is_open()) {
    std::cout << "[ERROR]: File \"" << fmat << "\" not found\n";
    throw "File not found";
  };
  // for now .mat is "ph_tag1 dens1\nph_tag2 dens2\n..."
  std::unordered_map<int, real_t> props;
  int ph_tag;
  while(!mat_in.eof()) {
    mat_in >> ph_tag;
    mat_in >> props[ph_tag];
  };
  for(int i = 0; i < mats.size(); ++i) {
    mats[i].set_parametres(props[mats[i].get_tag()]);
  };
};

void VolumeMesh::construct_from_file(std::string fmesh, std::string fmat) {
  std::string ext = get_file_extension(fmesh);
  if(ext == "msh") {
    std::ifstream f_in(fmesh);
    if(!f_in.is_open()) {
      std::cout << "[ERROR]: File \"" << fmesh << "\" not found\n";
      throw "File not found";
    };

    // Materiall read
    skip_to_tag(f_in, "$PhysicalNames");
    int ph_ts_n;
    f_in >> ph_ts_n;
    std::unordered_map<int, index_t> mat_map;
    for(int i = 0; i < ph_ts_n; ++i) {
      int dim, ph_tag;
      std::string name;
      f_in >> dim >> ph_tag;
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      std::getline(f_in, name);
      if(dim == 3) {
        mats.push_back(Material(name, ph_tag));
        mat_map[ph_tag] = mats.size() - 1;
      };
    };

    construct_materials(fmat);

    // Entities read
    skip_to_tag(f_in, "$Entities");
    int d0, d1, d2, d3;
    f_in >> d0 >> d1 >> d2 >> d3;
    skip_lines(f_in, d0 + d1 + d2 + 1); // and the first remained \n symbol
    std::unordered_map<int, index_t> entities;
    for(int i = 0; i < d3; ++i) {
      int tag, phsn, ntophsn;
      f_in >> tag;
      // ignore next 6 floats and lleading space
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      f_in >> phsn >> ntophsn;
      if(phsn < 1) {
        std::cout << "[ERROR]: No material provided for entity" <<
            " with tag " << tag << '\n';
        throw "No material";
      };
      if(phsn > 1) {
        std::cout << "[WARNING]: More than one material provided" <<
            " for entity with tag  " << tag << 
            ". The first material given will be used\n";
      };
      entities[tag] = mat_map[ph_ts_n];
    };

    // Node reading
    skip_to_tag(f_in, "$Nodes");
    int ebn, ptsn;
    f_in >> ebn >> ptsn;
    pts.reserve(ptsn);
    f_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::unordered_map<int, index_t> nodes;
    int abs_n = 0;
    for(int e = 0; e < ebn; ++e) {
      int d, t, p, nn;
      f_in >> d >> t >> p >> nn;
      for(int i = 0; i < nn; ++i) {
        int nid;
        f_in >> nid;
        nodes[nid] = i + abs_n;
      };
      for(int i = 0; i < nn; ++i) {
        real_t x,y,z;
        f_in >> x >> y >> z;
        pts.push_back(Point(x, y, z));
      };
      abs_n += nn;
    };

    // Tetrahedra(elements) reading
    skip_to_tag(f_in, "$Elements");
    int eln;
    f_in >> ebn >> eln;
    f_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    tetrs.reserve(eln);
    for(int e = 0; e < ebn; ++e) {
      int d, t, type, nn;
      f_in >> d >> t >> type >> nn;
      if(d == 3) {
        if(type != 4) {
          std::cout << "[ERROR]: File contain non-tetrahedral domain";
          throw "Non tetrahedral domain";
        };
        index_t id, p1, p2, p3, p4;
        for(int i = 0; i < nn; ++i) {
          f_in >> id >> p1 >> p2 >> p3 >> p4;
          tetrs.push_back(Tetrahedra(nodes[p1], nodes[p2], 
                                    nodes[p3], nodes[p4], entities[id]));
        };
      } else {
        skip_lines(f_in, nn+1);
      };
    };


  } else {
    std::cout << "[ERROR]: Wrong file format(" << ext <<
          ") for volume mesh providen\n";
  };
};

#endif