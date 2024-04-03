import os
import time
import dill as pickle
import multiprocessing
import git
import numpy as np

from assembly import problem_node
from parallelization import interface

def plan_for_structure(file_path,time_limit=float('inf'),world_size=None):
    goal=problem_node.Assembly_Node.from_npy(file_path,world_size)
    start=problem_node.Assembly_Node([],x_dim=goal.x_dim,y_dim=goal.y_dim,z_dim=goal.z_dim)
    return interface.solve_assembly_problem_using_rounds(start,goal,time_limit)

def plan_and_save(in_file_path,out_file_path,time_limit=float('inf'),world_size=None):
    result=plan_for_structure(in_file_path,time_limit,world_size)
    result["time_limit"]=time_limit
    result["world_size"]=world_size
    result["in_file_path"]=in_file_path
    with open(out_file_path,"wb") as fh:
        pickle.dump(result,fh)
    return result

def process_folder_of_structures(in_folder,out_folder,structure_prefix:str,x_dim,y_dim,z_dim,num_structs,time_limit=float('inf')):
    results=dict()
    for i in range(num_structs):
        file_name=structure_prefix+str(i)
        input_file=os.path.join(in_folder,file_name+".npy")
        output_file=os.path.join(out_folder,file_name+".out.pkl")
        try:
            out=plan_and_save(input_file,output_file,time_limit,(x_dim,y_dim,z_dim))
        except Exception as e:
            out=f"{e}"
        results[file_name]=out
    return results

def subprocess_structures(in_folder,structure_filenames,out_folder,x_dim,y_dim,z_dim,time_limit=float('inf')):
    for structure_name in structure_filenames:
        input_file=os.path.join(in_folder,structure_name+".npy")
        output_file=os.path.join(out_folder,structure_name+".out.pkl")
        process=multiprocessing.Process(target=plan_and_save,args=(input_file,output_file,time_limit,(x_dim,y_dim,z_dim)))
        process.start()
        process.join()
        with open(os.path.join(out_folder,"log.txt"),"at") as fh:
            fh.write(f"{structure_name}: {process.exitcode}\n")

def subprocess_folder_of_structures(in_folder,out_folder,structure_prefix:str,x_dim,y_dim,z_dim,structure_numbers,time_limit=float('inf')):
        for i in structure_numbers:
            file_name=structure_prefix+str(i)
            input_file=os.path.join(in_folder,file_name+".npy")
            output_file=os.path.join(out_folder,file_name+".out.pkl")
            process=multiprocessing.Process(target=plan_and_save,args=(input_file,output_file,time_limit,(x_dim,y_dim,z_dim)))
            process.start()
            process.join()
            with open(os.path.join(out_folder,"log.txt"),"at") as fh:
                fh.write(f"{i}: {process.exitcode}\n")

def run_and_save(in_folder,out_prefix,structure_prefix,x_dim,y_dim,z_dim,num_structs,time_limit=float('inf')):
    """
    create a folder at out_prefix/structure_prefix_YYYYMMDD_HHMMSS and save pickled planning info there

    Parameters: in_folder : path-like
                    folder containing npy files specifying structures, with paths like f"{in_folder}/{structure_prefix}{integer}.npy"
                out_prefix : path-like
                    folder to create a timestamped output folder in (if it doesn't exist, out_prefix will be created)
                structure_prefix : str
                    string specifying the naming scheme of the structure files
                x_dim : int
                    x_dim to set for the world to do the construction in
                y_dim : int
                    y_dim to set for the world to do the construction in
                z_dim : int
                    z_dim to set for the world to do the construction in
                num_structs : int
                    the number of structures in in_folder matching the structure_prefix.
                time_limit : float (default infinity)
                    maximum time allowed for planning for a single structure
    """
    run_specific_structs_and_save(in_folder,out_prefix,structure_prefix,x_dim,y_dim,z_dim,list(range(num_structs)),time_limit)

def run_specific_structs_and_save(in_folder,out_prefix,structure_prefix,x_dim,y_dim,z_dim,structure_numbers,time_limit=float('inf')):
    """
    create a folder at out_prefix/structure_prefix_YYYYMMDD_HHMMSS and save pickled planning info there

    Parameters: in_folder : path-like
                    folder containing npy files specifying structures, with paths like f"{in_folder}/{structure_prefix}{integer}.npy"
                out_prefix : path-like
                    folder to create a timestamped output folder in (if it doesn't exist, out_prefix will be created)
                structure_prefix : str
                    string specifying the naming scheme of the structure files
                x_dim : int
                    x_dim to set for the world to do the construction in
                y_dim : int
                    y_dim to set for the world to do the construction in
                z_dim : int
                    z_dim to set for the world to do the construction in
                structure_numbers : List[int]
                    the numbers of the structures in in_folder matching the structure_prefix.
                time_limit : float (default infinity)
                    maximum time allowed for planning for a single structure
    """
    timestr=time.strftime("%Y%m%d_%H%M%S")
    outfolder=os.path.join(out_prefix,structure_prefix+"_"+timestr)
    os.makedirs(outfolder,exist_ok=True)
    write_README(outfolder,structure_prefix,x_dim,y_dim,z_dim,structure_numbers,time_limit,"interface.solve_assembly_problem_using_rounds")
    subprocess_folder_of_structures(in_folder,outfolder,structure_prefix,x_dim,y_dim,z_dim,structure_numbers,time_limit)

def run_named_structs_and_save(in_folder,out_prefix,structure_prefix,x_dim,y_dim,z_dim,structure_names,time_limit=float('inf')):
    """
    create a folder at out_prefix/structure_prefix_YYYYMMDD_HHMMSS and save pickled planning info there

    Parameters: in_folder : path-like
                    folder containing npy files specifying structures, with paths like f"{in_folder}/{structure_prefix}{integer}.npy"
                out_prefix : path-like
                    folder to create a timestamped output folder in (if it doesn't exist, out_prefix will be created)
                structure_prefix : str
                    string specifying the naming scheme of the structure files
                x_dim : int
                    x_dim to set for the world to do the construction in
                y_dim : int
                    y_dim to set for the world to do the construction in
                z_dim : int
                    z_dim to set for the world to do the construction in
                structure_names : List[str]
                    the names of the structures in in_folder
                time_limit : float (default infinity)
                    maximum time allowed for planning for a single structure
    """
    timestr=time.strftime("%Y%m%d_%H%M%S")
    outfolder=os.path.join(out_prefix,structure_prefix+"_"+timestr)
    os.makedirs(outfolder,exist_ok=True)
    write_README(outfolder,structure_prefix,x_dim,y_dim,z_dim,structure_names,time_limit,"interface.solve_assembly_problem_using_rounds")
    subprocess_structures(in_folder,structure_names,outfolder,x_dim,y_dim,z_dim,time_limit)

def write_README(outfolder,structure_prefix,x_dim,y_dim,z_dim,structure_numbers,time_limit,function_name):
    repo=git.Repo(os.path.abspath(os.path.join(os.path.split(os.path.abspath(interface.__file__))[0],"..","..")))
    commit_name=repo.head.commit.name_rev
    with open(os.path.join(outfolder,"README"),"w") as fh:
        fh.writelines([f"structure_prefix: {structure_prefix}\n",
                       f"world_size: ({x_dim},{y_dim},{z_dim})\n",
                       f"time_limit: {time_limit}\n",
                       f"function: {function_name}\n",
                       f"commit: {commit_name}\n",
                       f"structures: {structure_numbers}\n"])
    
def load(folder,structure_prefix,structure_numbers):
    data=dict()
    for i in structure_numbers:
        path=os.path.join(folder,structure_prefix+str(i)+".out.pkl")
        try:
            with open(path,"rb") as fh:
                data[i]=pickle.load(fh)
        except Exception:
            data[i]=None
    return data

def classify_cases(data_dict_from_load):
    success=[]
    failed=[]
    infeasible=[]
    lltimeout=[]
    hltimeout=[]
    hlinfeasible=[]
    other_exception=[]
    for key in data_dict_from_load:
        dat=data_dict_from_load[key]
        if dat is not None:
            hls=dat["high_level_plan_status"]
            if hls=="FOUND":
                status=dat["low_level_plan_status"]
                if status=="FOUND":
                    success.append(key)
                if status=="INFEASIBLE":
                    infeasible.append(key)
                if status=="TIMEOUT":
                    lltimeout.append(key)
                if status=="OtherException":
                    other_exception.append(key)
            elif hls=="INFEASIBLE":
                hlinfeasible.append(key)
            elif hls=="TIMEOUT":
                hltimeout.append(key)
        else:
            failed.append(key)
    return {"success":success,"failed":failed,"llinfeasible":infeasible,"timeout":lltimeout,"hlinfeasible":hlinfeasible,"hltimeout":hltimeout,"other_exception":other_exception}

def success_durations(data_dict_from_load):
    outcomes=classify_cases(data_dict_from_load)
    return [data_dict_from_load[i]["phase_durations"] for i in outcomes["success"]]

def success_statistics(success_durs):
    phases=list(success_durs[0].keys())
    times_by_phase={phase:[d[phase] for d in success_durs] for phase in phases}
    minimum={phase:np.min(times_by_phase[phase]) for phase in phases}
    maximum={phase:np.max(times_by_phase[phase]) for phase in phases}
    median={phase:np.median(times_by_phase[phase]) for phase in phases}
    avg={phase:np.mean(times_by_phase[phase]) for phase in phases}
    longest={phase:0 for phase in phases}
    for d in success_durs:
        longest[phases[np.argmax([d[i] for i in phases])]]+=1
    return {"min":minimum,"max":maximum,"median":median,"mean":avg,"longest":longest}