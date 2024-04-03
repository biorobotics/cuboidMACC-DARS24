import os
import subprocess
import tempfile
import tarfile

import numpy as np
import trimesh
from plotly import graph_objects as go

import meshcat

from assembly import block_utils,problem_node
from path import cbs
from parallelization import interface

from visualize import visualize_2D_CBS
from typing import List,Dict

def RmatZ(radians):
    """
    rotation matrix to rotate about the Z axis by specified amount
    """
    c=np.cos(radians)
    s=np.sin(radians)
    return np.array([[c,s,0],
                     [-s,c,0],
                     [0,0,1]])
def get_transform(x,y,z,radians):
    rotation_matrix=RmatZ(radians)
    transformation_matrix=np.eye(4)
    transformation_matrix[:3,:3]=rotation_matrix
    transformation_matrix[:3,3]=(x,y,z)
    return transformation_matrix
def get_cuboid(x,y,z,radians,length,**kwargs):
    transformation_matrix=get_transform(x,y,z,radians)
    box=trimesh.primitives.Box((length,1,1),transformation_matrix)
    vertices=box.vertices
    faces=box.faces
    return go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],**kwargs)

def get_frames(t,solution,environment,world,interp_steps):
    frames=[]
    world_blocks=[]
    for block,location in world.blocks:
        world_blocks.append(get_cuboid(location.x,location.y,location.z,np.deg2rad(location.turn),block.length,color='red'))
    for i in range(interp_steps):
        robots=[]
        for agent in solution:
            state=environment.get_state(agent,solution,t)
            exit_state=environment.get_edge(agent,solution,t).exit_state()
            if isinstance(state.block,block_utils.NoBlock):
                length=1
            else:
                length=state.block.length
            x=state.location.x*(1-i/interp_steps)+exit_state.location.x*i/interp_steps
            y=state.location.y*(1-i/interp_steps)+exit_state.location.y*i/interp_steps
            z=state.location.z*(1-i/interp_steps)+exit_state.location.z*i/interp_steps
            theta=visualize_2D_CBS.angle_interp(i,0,state.location.turn,interp_steps,exit_state.location.turn)
            robot=get_cuboid(x,y,z,np.deg2rad(theta),length,color='cyan')
            robots.append(robot)
        frames.append(go.Frame(data=world_blocks+robots))
    return frames

def animate(environment:cbs.Environment,solution,interp_steps=4):
    xdim,ydim,zdim=environment.initial_world.shape
    world=environment.get_world_at(0)
    max_t = max([len(plan) for plan in solution.values()])
    frames=[]
    actions=dict()
    for t in range(max_t+1):
        frames.extend(get_frames(t,solution,environment,world,interp_steps))
        world,actions=environment.update_world(t,world,solution,actions)
    fig=go.Figure(data=frames[0].data,frames=frames[1:],
                  layout=go.Layout(updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None])])],
                                   scene=dict(aspectmode="data",xaxis=dict(nticks=xdim+1,range=(0,xdim)),yaxis=dict(nticks=ydim+1,range=(0,ydim)),zaxis=dict(nticks=zdim+1,range=(0,zdim)))))
    fig.show()

def state_block_path(agent:str,state:cbs.State,parent:str):
    agent_path=f"{parent}/{agent}"
    block=state.block
    if isinstance(block,block_utils.CuboidalBlock) and block.length>0:
        name=f"carried{block.length}"
        block_path=f"{agent_path}/{name}"
        return block_path
    return None

def get_edges(block_length):
    points=np.array([(-block_length,-1,-1),(-block_length,0,-1),(-block_length,0,0),(-block_length,0,-1),(-block_length,-1,-1),(-block_length,-1,0),(-block_length,-1,-1),
     (0,-1,-1),(0,0,-1),(0,0,0),(0,-1,0),(0,0,0),(0,0,-1),(0,-1,-1),
     (0,-1,0),(-block_length,-1,0),
     (-block_length,0,0),(0,0,0),
     (0,0,-1),(-block_length,0,-1)])
    return (points+np.array([block_length/2,.5,.5])).transpose()

def color_blocks(block_action_tuples,vis,parent,color=[255,0,0,255]):
    for block_action in block_action_tuples:
        add,length,turn,x,y,z=block_action
        block_path=f"{parent}/(L={length} X={x} Y={y} Z={z} Turn={turn})"
        vis[block_path].set_property("color",color)

def meshcat_add_robots(environment:cbs.Environment,solution:Dict[str,List[cbs.Edge]],vis:meshcat.Visualizer,parent:str):
    for agent in environment.agent_dict:
        start_state=solution[agent][0].state1
        start=start_state.location
        agent_path=f"{parent}/{agent}"
        robot_path=f"{agent_path}/robot"
        vis[agent_path].set_transform(get_transform(start.x,start.y,start.z,np.deg2rad(start.turn)))
        vis[robot_path].set_object(meshcat.geometry.Box((1,1,1)))
        vis[robot_path].set_transform(np.eye(4))
        vis[robot_path].set_property("color",[0,100,100,255])
        edges=solution[agent]
        blocks={edge.action.block for edge in edges if edge.action.action_type in cbs.Action.BLOCK_ACTIONS}
        for block in blocks:
            if block.length>0:
                name=f"carried{block.length}"
                block_path=f"{agent_path}/{name}"
                vis[block_path].set_object(meshcat.geometry.Box((block.length,1,1)))
                vis[block_path].set_transform(meshcat.transformations.translation_matrix([0,0,1]))
                vis[block_path].set_property("color",[100,100,0,255])
                vis[block_path].set_property("visible",False)
                vis[block_path+"/outline"].set_object(meshcat.geometry.Line(meshcat.geometry.PointsGeometry(get_edges(block.length))))
        #set initially carried block visible if any
        start_block_path= state_block_path(agent,start_state,parent)
        if start_block_path is not None:
            vis[start_block_path].set_property("visible",True)
        if not start.in_world:
            vis[agent_path].set_property("visible",False)
        else:
            vis[agent_path].set_property("visible",True)

def meshcat_reset_robot(environment:cbs.Environment,solution:Dict[str,List[cbs.Edge]],frame,parent:str):
    for agent in environment.agent_dict:
        agent_path=f"{parent}/{agent}"
        start_state=solution[agent][0].state1
        start=start_state.location
        edges=solution[agent]
        blocks={edge.action.block for edge in edges if edge.action.action_type in cbs.Action.BLOCK_ACTIONS}
        for block in blocks:
            if block.length>0:
                name=f"carried{block.length}"
                block_path=f"{agent_path}/{name}"
                frame[block_path].set_property("visible","bool",False)
        #set initially carried block visible if any
        start_block_path= state_block_path(agent,environment.agent_dict[agent].start,parent)
        if start_block_path is not None:
            frame[start_block_path].set_property("visible","bool",True)
        if not start.in_world:
            frame[agent_path].set_property("visible","bool",False)
        else:
            frame[agent_path].set_property("visible","bool",True)

def meshcat_add_world_blocks(world:cbs.WorldState,vis:meshcat.Visualizer,parent:str):
    block_status=dict()
    for block,location in world.blocks:
        length=block.length
        x=location.x
        y=location.y
        z=location.z
        turn=location.vertical()
        block_path=f"{parent}/(L={length} X={x} Y={y} Z={z} Turn={turn})"
        vis[block_path].set_object(meshcat.geometry.Box((length,1,1)))
        vis[block_path].set_transform(get_transform(x,y,z,np.pi/2*turn))
        vis[block_path].set_property("visible",True)
        vis[block_path].set_property("color",[100,100,0,255])
        vis[block_path+"/outline"].set_object(meshcat.geometry.Line(meshcat.geometry.PointsGeometry(get_edges(length))))
        block_status[block_path]=True
    return block_status

def meshcat_reset_blocks(block_status:Dict[str,bool],frame:meshcat.Visualizer):
    for block_path in block_status:
        frame[block_path].set_property("visible","bool",block_status[block_path])

def meshcat_add_goal_blocks(actions,vis,parent):
    block_status=dict()
    for action in actions:
        if action.action_type==cbs.Action.PLACEMENT:
            length=action.block.length
            x=action.location2.x
            y=action.location2.y
            z=action.location2.z
            turn=action.location2.vertical()
            block_path=f"{parent}/(L={length} X={x} Y={y} Z={z} Turn={turn})"
            vis[block_path].set_object(meshcat.geometry.Box((length,1,1)))
            vis[block_path].set_transform(get_transform(x,y,z,np.pi/2*turn))
            vis[block_path].set_property("visible",False)
            vis[block_path].set_property("color",[100,100,0,255])
            vis[block_path+"/outline"].set_object(meshcat.geometry.Line(meshcat.geometry.PointsGeometry(get_edges(length))))
            block_status[block_path]=False
    return block_status

def meshcat_interpolate_frames(t,solution,environment:cbs.Environment,interp_steps,start_frame_number,vis,anim:meshcat.animation.Animation,parent:str):
    for agent in solution:
        agent_path=f"{parent}/{agent}"
        edge=environment.get_edge(agent,solution,t)
        if edge.action.action_type==cbs.Action.MOVE:
            state=edge.state1
            exit_state=edge.exit_state()
            if not state.location.in_world and exit_state.location.in_world:
                #entering world, only be visible at end
                anim.at_frame(vis,start_frame_number+interp_steps)[agent_path].set_property("visible","bool",True)
            elif state.location.in_world and not exit_state.location.in_world:
                #leaving world, only be visible at first frame
                anim.at_frame(vis,start_frame_number+1)[agent_path].set_property("visible","bool",False)
            for i in range(interp_steps+1):
                x=state.location.x*(1-i/interp_steps)+exit_state.location.x*i/interp_steps
                y=state.location.y*(1-i/interp_steps)+exit_state.location.y*i/interp_steps
                z=state.location.z*(1-i/interp_steps)+exit_state.location.z*i/interp_steps
                theta=visualize_2D_CBS.angle_interp(i,0,state.location.turn,interp_steps,exit_state.location.turn)
                tf=get_transform(x,y,z,np.deg2rad(theta))
                anim.at_frame(vis,start_frame_number+i)[agent_path].set_transform(tf)
    return start_frame_number+interp_steps

def meshcat_block_actions(t,solution,environment,frame,parent):
    for agent in solution:
        edge=environment.get_edge(agent,solution,t)
        meshcat_block_action(edge.action,frame,parent)
        if edge.action.action_type==cbs.Action.PLACEMENT:
            block_path=state_block_path(agent,edge.state1,parent)
            frame[block_path].set_property("visible","bool",False)
        elif edge.action.action_type==cbs.Action.REMOVAL:
            block_path=state_block_path(agent,edge.exit_state(),parent)
            frame[block_path].set_property("visible","bool",True)

def meshcat_block_action(action,frame:meshcat.Visualizer,parent:str):
    if action.action_type in cbs.Action.BLOCK_ACTIONS:
        length=action.block.length
        x=action.location2.x
        y=action.location2.y
        z=action.location2.z
        turn=action.location2.vertical()
        name=f"(L={length} X={x} Y={y} Z={z} Turn={turn})"
        block_path=f"{parent}/(L={length} X={x} Y={y} Z={z} Turn={turn})"
        if action.action_type==cbs.Action.PLACEMENT:
            frame[block_path].set_property("visible","bool",True)
            # frame[name].set_object(meshcat.geometry.Box((length,1,1)))
            # frame[name].set_transform(get_transform(x,y,z,np.pi/2*turn))

            # frame[agent].set_object(meshcat.geometry.Box((1,1,1)))
        else:
            frame[block_path].set_property("visible","bool",False)
            # frame[name].delete()
            # frame[agent].set_object(meshcat.geometry.Box((length,1,1)))

def visualize_meshcat(environment:cbs.Environment,solution,interp_steps=4):
    vis=meshcat.Visualizer()
    vis.open()
    anim=meshcat.animation.Animation()
    xoff,yoff,_=environment.dimension
    parent="world"
    vis[parent].set_transform(meshcat.transformations.translation_matrix(np.array([-xoff/2,-yoff/2,0.5])))
    delta=max(xoff,yoff)*.75
    # vis["/Cameras/default/rotated"].set_transform(meshcat.transformations.translation_matrix([0,0,0]))
    # vis["/Cameras/default/rotated"].set_property("position",[delta,delta,delta])
    actions=[edge.action for path in solution.values() for edge in path if edge.action.action_type in cbs.Action.BLOCK_ACTIONS]
    meshcat_add_robots(environment,solution,vis,parent)
    goal_block_status=meshcat_add_goal_blocks(actions,vis,parent)
    world_block_status=meshcat_add_world_blocks(environment._world_state,vis,parent)
    frame0_block_status=goal_block_status
    frame0_block_status.update(world_block_status)
    max_t = max([plan[-1].state1.time for plan in solution.values()])
    next_frame_number=0
    with anim.at_frame(vis,0) as frame:
        meshcat_reset_blocks(frame0_block_status,frame)
        meshcat_reset_robot(environment,solution,frame,parent)
    for t in range(max_t+1):
        next_frame_number=meshcat_interpolate_frames(t,solution,environment,interp_steps,next_frame_number,vis,anim,parent)
        with anim.at_frame(vis,next_frame_number) as frame:
            meshcat_block_actions(t,solution,environment,frame,parent)
        next_frame_number+=interp_steps
    vis.set_animation(anim,play=False)
    return vis

def visualize_world_meshcat(world:cbs.WorldState,vis:meshcat.Visualizer=None):
    if vis is None:
        vis=meshcat.Visualizer()
        vis.open()
    xoff,yoff,_=world.occupancy.shape
    parent="world"
    vis[parent].set_transform(meshcat.transformations.translation_matrix(np.array([-xoff/2,-yoff/2,0.5])))
    delta=max(xoff,yoff)*.75
    # vis["/Cameras/default/rotated"].set_transform(meshcat.transformations.translation_matrix([0,0,0]))
    # vis["/Cameras/default/rotated"].set_property("position",[delta,delta,delta])
    meshcat_add_world_blocks(world,vis,parent)
    return vis

def animate_high_level_plan(start:problem_node.Assembly_Node,plan:List[problem_node.Assembly_Node]):
    initial_world=interface.world_state_from_assembly_node(start)
    actions=[interface.action_from_assembly_node(node) for node in plan]
    vis=meshcat.Visualizer()
    vis.open()
    anim=meshcat.animation.Animation()
    xoff,yoff,_=initial_world.occupancy.shape
    parent="world"
    vis[parent].set_transform(meshcat.transformations.translation_matrix(np.array([-xoff/2,-yoff/2,0.5])))
    delta=max(xoff,yoff)*.75
    # vis["/Cameras/default/rotated"].set_transform(meshcat.transformations.translation_matrix([0,0,0]))
    # vis["/Cameras/default/rotated"].set_property("position",[delta,delta,delta])
    world_block_status=meshcat_add_world_blocks(initial_world,vis,parent)
    goal_block_status=meshcat_add_goal_blocks(actions,vis,parent)
    frame0_block_status=goal_block_status
    frame0_block_status.update(world_block_status)
    max_t = len(plan)
    with anim.at_frame(vis,0) as frame:
        meshcat_reset_blocks(frame0_block_status,frame)
    for t in range(max_t):
        with anim.at_frame(vis,t+1) as frame:
            meshcat_block_action(actions[t],frame,parent)
    vis.set_animation(anim,play=False)
    return vis

def meshcat_frames_to_video(tar_file_path,output_path="output.mp4",framerate=60,overwrite=False):
    """
    Try to convert a tar file containing a sequence of frames saved by the
    meshcat viewer into a single video file.

    This relies on having `ffmpeg` installed on your system.

    derived from meshcat-python, but using a slightly different ffmpeg command since the provided one produces a black video for some reason
    """
    output_path = os.path.abspath(output_path)
    if os.path.isfile(output_path) and not overwrite:
        raise ValueError("The output path {:s} already exists. To overwrite that file, you can pass overwrite=True to this function.".format(output_path))
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(tar_file_path) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, tmp_dir)
        args = ["ffmpeg",
                "-r", str(framerate),
                "-i", r"%07d.png",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p"]
        if overwrite:
            args.append("-y")
        args.append(output_path)
        try:
            subprocess.check_call(args, cwd=tmp_dir)
        except subprocess.CalledProcessError as e:
            print("""
Could not call `ffmpeg` to convert your frames into a video.
If you want to convert the frames manually, you can extract the
.tar archive into a directory, cd to that directory, and run:
ffmpeg -r $FRAMERATE -i %07d.png -vcodec libx264 -pix_fmt yuv420p $SAVE_PATH
                """)
            raise
    print("Saved output as {:s}".format(output_path))
    return output_path