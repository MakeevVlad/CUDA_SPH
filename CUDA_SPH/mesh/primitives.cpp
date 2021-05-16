#ifndef PRIMITIVES_CPP
#define PRIMITIVES_CPP

#include "primitives.h"
#include "../info.cuh"
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <limits>


// Utils
std::string get_file_extension(std::string fname) {
  int n = fname.rfind(".");
  if(n == std::string::npos) {
    return "";
  } else {
    return fname.substr(n+1);
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


#endif