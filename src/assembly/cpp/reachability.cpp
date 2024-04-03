#include <iostream>
#include <stdexcept>
#include <queue>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "reachability.h"

LocationSet::LocationSet():elements(0),included_indicators(boost::extents[0][0][0][0]){}
LocationSet::LocationSet(size_t x_dim,size_t y_dim,size_t z_dim):elements(0),included_indicators(boost::extents[x_dim][y_dim][z_dim][2]){
    for(int i=0;i<x_dim;i++){
        for(int j=0;j<y_dim;j++){
            for(int k=0;k<z_dim;k++){
                included_indicators[i][j][k][0]=false;
                included_indicators[i][j][k][1]=false;
            }
        }
    }
}
LocationSet::LocationSet(size_t x_dim,size_t y_dim,size_t z_dim,const std::vector<Location>& initial_elements):elements(initial_elements),included_indicators(boost::extents[x_dim][y_dim][z_dim][2]){
    for(int i=0;i<x_dim;i++){
        for(int j=0;j<y_dim;j++){
            for(int k=0;k<z_dim;k++){
                included_indicators[i][j][k][0]=false;
                included_indicators[i][j][k][1]=false;
            }
        }
    }
    // for(auto it=included_indicators.begin();it!=included_indicators.end();it++){
    //     *it=false;
    // }
    for(auto element:elements){
        included_indicators(element)=true;
    }
}

bool LocationSet::count(const Location& query) const{
    // if (query[0]<0 || query[0]>=included_indicators.shape()[0] || query[1]<0 || query[1]>=included_indicators.shape()[1] || query[2]<0 || query[2]>=included_indicators.shape()[2]){
    //     std::cout<<"("<<query[0]<<","<<query[1]<<","<<query[2]<<") out of bounds"<<std::endl;
    //     return false;
    // }
    return included_indicators(query);
}
void LocationSet::insert(const Location& query){
    if(!count(query)){
        included_indicators(query)=true;
        elements.push_back(query);
    }
}

std::vector<Location>::const_iterator LocationSet::cbegin() const{
    return elements.cbegin();
}
std::vector<Location>::const_iterator LocationSet::cend() const{
    return elements.cend();
}

CachingReachability::CachingReachability(int block_length,size_t x_size,size_t y_size,size_t z_size,py::array_t<bool,py::array::c_style | py::array::forcecast> occupancy_array): 
occu_grid(occupancy_array),
unreachable_locations(x_size,y_size,z_size),
reachable_locations(x_size,y_size,z_size),
block_length(block_length)
 {  
    x_dim=x_size;
    y_dim=y_size;
    z_dim=z_size;
    if (occu_grid.ndim()!=3){
        int ndim=occu_grid.ndim();
        throw std::invalid_argument("occupancy_array was "+std::to_string(ndim)+",-D not 3-D");
    }
    if(x_dim!=occu_grid.shape(0)){
        throw std::invalid_argument("occupancy array x_dim="+std::to_string(occu_grid.shape(0))+", x_size="+std::to_string(x_size));
    }
    if(y_dim!=occu_grid.shape(1)){
        throw std::invalid_argument("occupancy array y_dim="+std::to_string(occu_grid.shape(1))+", y_size="+std::to_string(y_size));
    }
    if(z_dim!=occu_grid.shape(2)){
        throw std::invalid_argument("occupancy array z_dim="+std::to_string(occu_grid.shape(2))+", z_size="+std::to_string(z_size));
    }
    for(int y=0;y<y_dim;y++){
        Location cell1{0,y,0,0};
        reachable_locations.insert(cell1);
        Location cell2{x_dim-1,y,0};
        reachable_locations.insert(cell2);
        Location cell3{0,y,0,1};
        reachable_locations.insert(cell3);
        Location cell4{x_dim-1,y,1};
        reachable_locations.insert(cell4);
    }
    for(int x=1;x<x_dim-1;x++){
        Location cell5{x,0,0,0};
        reachable_locations.insert(cell5);
        Location cell6{x,y_dim-1,0,0};
        reachable_locations.insert(cell6);
        Location cell7{x,0,0,1};
        reachable_locations.insert(cell7);
        Location cell8{x,y_dim-1,0,1};
        reachable_locations.insert(cell8);
    }

 }

 bool CachingReachability::has_path_from_boundary(const Location& location){
    if(!can_stand_at(location[0],location[1],location[2],occu_grid)||!no_collision(location)){
        return false;
    }
    std::priority_queue<Location> open_list;
    open_list.push(location);
    LocationSet connected_to_location(x_dim,y_dim,z_dim,{location});
    
    Location current_location;
    while(!open_list.empty()){
        current_location=open_list.top();
        open_list.pop();

        if(reachable_locations.count(current_location)){
            //query cell and everything connected to it is reachable
            reachable_locations.insert(connected_to_location.cbegin(),connected_to_location.cend());
            return true;
        }
        if(unreachable_locations.count(current_location)){
            //query cell and everything connected to it is unreachable
            break;
        }
        for(auto next_location:all_next_locations(current_location,occu_grid)){
            if(can_transition_between(current_location,next_location) && no_collision(next_location) && !connected_to_location.count(next_location)){
                open_list.push(next_location);
                connected_to_location.insert(next_location);
            }
        }
    }
    //query cell and everything connected to it is unreachable
    unreachable_locations.insert(connected_to_location.cbegin(),connected_to_location.cend());
    return false;
 }

 bool CachingReachability::can_transition_between(const Location& start, const Location& end){
    //determine if the cells swept out by the transition are collision free
    //assumes that both start and end location are collision free!
    if(start[3]==end[3]){
        //if not rotating, no cells are swept out that are not occupied at either start or end so this is automatically true
        return true;
    }
    int xmin=start[0]-(int)block_length/2;
    int xmax=start[0]+(int)block_length/2;
    int ymin=start[1]-(int)block_length/2;
    int ymax=start[1]+(int)block_length/2;
    for(int x=xmin;x<=xmax;x++){
        for(int y=ymin;y<=ymax;y++){
            if(collides(x,y,start[2]+1,occu_grid)){
                return false;
            }
        }
    }
    return true;
 }

 bool CachingReachability::no_collision(const Location& location){
    std::vector<Cell> occupied=cells_occupied(block_length,location[3],location[0],location[1],location[2]+1);
    for(auto filled:occupied){
        if(collides(filled[0],filled[1],filled[2],occu_grid)){
            return false;
        }
    }
    return true;
 }

std::vector<Location> all_next_locations(const Location& location,const py::array_t<bool,py::array::c_style>& occu_grid){
    size_t x_dim=occu_grid.shape(0);
    size_t y_dim=occu_grid.shape(1);
    size_t z_dim=occu_grid.shape(2);
    std::array<int,3> move_dir={-1,0,1};
    std::vector<Location> next_locations;
    int x;
    int y;
    int z;
    for(auto dz:move_dir){
        if(location[3]==1){
            //move in x and possibly z
            for(auto dx:move_dir){
                if(dx!=0){
                    x=location[0]+dx;
                    y=location[1];
                    z=location[2]+dz;

                    if(can_stand_at(x,y,z,occu_grid)){
                        Location newcell{x,y,z,location[3]};
                        next_locations.push_back(newcell);
                    }
                }
            }
        }
        else{
            //move in y and possibly z
           for(auto dy:move_dir){
                if(dy!=0){
                    x=location[0];
                    y=location[1]+dy;
                    z=location[2]+dz;

                    if(can_stand_at(x,y,z,occu_grid)){
                        Location newcell{x,y,z,location[3]};
                        next_locations.push_back(newcell);
                    }
                }
           }
        }
    }
    //turn
    int other_rot;
    if(location[3]==0){
        other_rot=1;
    }else{
        other_rot=0;
    }
    Location newlocation{location[0],location[1],location[2],other_rot};
    next_locations.push_back(newlocation);
    return next_locations;
}

bool can_stand_at(int x,int y,int z,const py::array_t<bool,py::array::c_style>& occu_grid){
    size_t x_dim=occu_grid.shape(0);
    size_t y_dim=occu_grid.shape(1);
    size_t z_dim=occu_grid.shape(2);
    auto occupancy = occu_grid.unchecked<3>();
    if(x>=0 && x<x_dim && y>=0 && y<y_dim && z>=0 && z<z_dim){
        if(!occupancy(x,y,z) && (z==0 || occupancy(x,y,z-1))){
            return true;
        }
    }
    return false;
}

bool collides(int x,int y,int z,const py::array_t<bool,py::array::c_style>& occu_grid){
    size_t x_dim=occu_grid.shape(0);
    size_t y_dim=occu_grid.shape(1);
    size_t z_dim=occu_grid.shape(2);
    auto occupancy = occu_grid.unchecked<3>();
    if(x>=0 && x<x_dim && y>=0 && y<y_dim && z>=0 && z<z_dim){
        return occupancy(x,y,z);
    }
    return false;
}

std::vector<Cell> cells_occupied(int length, int rotation, int x,int y,int z){
    int xl,xu,yl,yu;
    if (rotation!=1){
        xl=-length/2;
        xu=length/2;
        yl=0;
        yu=0;
    }else{
        xl=0;
        xu=0;
        yl=-length/2;
        yu=length/2;
    }
    std::vector<Cell> occupied;
    occupied.reserve(length);
    for(int dx=xl;dx<=xu;dx++){
        for(int dy=yl;dy<=yu;dy++){
            occupied.emplace_back(Cell({x+dx,y+dy,z}));
        }
    }
    return occupied;
}
py::array_t<bool> occupancy_grid(const std::vector<std::vector<int>>& action_sequence,int x_dim,int y_dim,int z_dim){
    py::array_t<bool> result=py::array_t<bool>({x_dim,y_dim,z_dim});
    auto proxy=result.mutable_unchecked<3>();
    for(int i=0;i<x_dim;i++){
        for(int j=0;j<y_dim;j++){
            for(int k=0;k<z_dim;k++){
                proxy(i,j,k)=false;
            }
        }
    }
    for(auto action:action_sequence){
        for(Cell cell:cells_occupied(action[1],action[2],action[3],action[4],action[5])){
            if(in_bounds(x_dim,y_dim,z_dim,cell)){
                if(action[0]==-1){
                    //remove
                    proxy(cell[0],cell[1],cell[2])=false;
                }
                else if(action[0]==1){
                    //add
                    proxy(cell[0],cell[1],cell[2])=true;
                }
                else{
                    //unrecognized action type
                    throw std::invalid_argument(std::to_string(action[0])+" is not a recognized action type (-1 or 1)\n");
                }
            }
        }
    }
    return result;
}

PYBIND11_MODULE(reachability,m) {
    m.doc() = "pybind11 bindings for computing reachability of cells in 4-connected blockworld";
    m.def("can_stand_at",&can_stand_at,"test if a location is on the ground or has a block immediately below it");
    m.def("all_next_cells",&all_next_locations,"Return a list containing all the cells reachable from the input that can be stood upon");
    m.def("cells_occupied",&cells_occupied,"Return list of cells affected by the specified action");
    m.def("occupancy_grid",&occupancy_grid,"Return an (x_dim,y_dim,z_dim) bool numpy array that is true if a cell is occupied and false otherwise");

    py::class_<CachingReachability>(m,"CachingReachability")
        .def(py::init<int,size_t,size_t,size_t,py::array_t<bool,py::array::c_style | py::array::forcecast>>())
        .def("has_path_from_boundary",&CachingReachability::has_path_from_boundary)
        .def_readonly("occu_grid",&CachingReachability::occu_grid)
        .def_readonly("reachable_locations",&CachingReachability::reachable_locations)
        .def_readonly("unreachable_locations",&CachingReachability::unreachable_locations);

    py::class_<LocationSet>(m,"LocationSet")
        .def_readonly("elements",&LocationSet::elements);
}