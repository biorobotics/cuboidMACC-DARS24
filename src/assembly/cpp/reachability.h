#ifndef DEF_REACHABILITY_H
#define DEF_REACHABILITY_H

#include <unordered_set>
#include <utility>
#include <pybind11/numpy.h>
#include "boost/multi_array.hpp"
namespace py=pybind11;

using Cell=std::array<int,3>;
using Location=std::array<int,4>;
bool in_bounds(size_t x_dim,size_t y_dim,size_t z_dim,const Cell& cell){
    return cell[0]>=0 && cell[0]<x_dim && cell[1]>=0 && cell[1]<y_dim && cell[2]>=0 && cell[2]<z_dim;
}
bool in_bounds(size_t x_dim,size_t y_dim,size_t z_dim,const Location& location){
    return location[0]>=0 && location[0]<x_dim && location[1]>=0 && location[1]<y_dim && location[2]>=0 && location[2]<z_dim;
}
std::vector<Location> all_next_locations(const Location& location,const py::array_t<bool,py::array::c_style>& occu_grid);
bool can_stand_at(int x,int y,int z,const py::array_t<bool,py::array::c_style>& occu_grid);
std::vector<Cell> cells_occupied(int,int,int,int,int);
py::array_t<bool> occupancy_grid(const std::vector<std::vector<int>>&,int,int,int);
bool collides(int x,int y,int z,const py::array_t<bool,py::array::c_style>& occu_grid);

class LocationSet{
public:
    LocationSet();
    LocationSet(size_t,size_t,size_t);
    LocationSet(size_t,size_t,size_t,const std::vector<Location>&);

    bool count(const Location&) const;
    void insert(const Location&);
    template<class Iterator>
    void insert(Iterator begin,Iterator end){
        for(auto it=begin;it!=end;it++){
            insert(*it);
        }
    }

    std::vector<Location>::const_iterator cbegin() const;
    std::vector<Location>::const_iterator cend() const;

    LocationSet& operator=(const LocationSet&);

    std::vector<Location> elements;
    boost::multi_array<bool,4> included_indicators;
};

class CachingReachability{
public:
    CachingReachability(int,size_t,size_t,size_t,py::array_t<bool,py::array::c_style | py::array::forcecast> occu_grid);
    bool has_path_from_boundary(const Location&);
    bool no_collision(const Location&);
    bool can_transition_between(const Location& start, const Location& end);
    py::array_t<bool,py::array::c_style> occu_grid;
    int block_length;
    size_t x_dim;
    size_t y_dim;
    size_t z_dim;
    LocationSet reachable_locations;
    LocationSet unreachable_locations;
};

#endif