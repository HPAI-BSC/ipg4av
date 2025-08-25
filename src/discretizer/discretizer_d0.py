from database.utils import vector_angle, create_rectangle
from discretizer.predicates import get_action_id, get_action_from_id, IsTrafficLightNearby,IsZebraNearby, FrontObjects, Action, LanePosition, StopAreaNearby, BlockProgress, NextIntersection, Velocity, Rotation
from policy_graph.environment import AVEnvironment
import numpy as np
from typing import Tuple, Union, Any, List
from pgeon.discretizer import Discretizer
from policy_graph.discretizer import AVPredicate
import pandas as pd

class AVDiscretizer(Discretizer):
    def __init__(self, environment: AVEnvironment, id:str):     
        
        super().__init__()
        self.environment =environment
        self.id = id

        self.eps_rot = 0.3
        self.eps_vel = 0.2  
        self.eps_acc = 0.3

        self.vel_values = [Velocity.STOPPED, Velocity.MOVING] if 'a' in self.id else [Velocity.STOPPED, Velocity.LOW, Velocity.MEDIUM, Velocity.HIGH]
        
        #
        self.agent_size =(1.730, 4.084) #width, length in meters #TODO: move

        self.vehicle_state_columns = ['x', 'y', 'velocity', 'steering_angle', 'yaw']   
        self.state_columns_for_detection = ['category', 'attribute', 'bbox_center']
        self.state_columns_for_action = ['velocity', 'acceleration', 'steering_angle']     


    
    ##################################
    ### DISCRETIZERS
    ##################################

    def discretize(self,
                   state: np.ndarray, detections:pd.DataFrame=None
                   ) -> Tuple[AVPredicate, ...]:
        x, y, velocity, steer_angle, yaw = state 
        block_progress, lane_pos_pred  = self.discretize_position(x,y,yaw)
        mov_predicate = self.discretize_speed(velocity)
        rot_predicate = self.discretize_steering_angle(steer_angle)
        sign_predicate, zebra_predicate, traffic_light_predicate = self.discretize_stop_line(x,y,yaw)
        detection_predicate = self.discretize_detections(detections)
        
        return (AVPredicate(BlockProgress, [block_progress]),
                AVPredicate(LanePosition, [lane_pos_pred]),
                AVPredicate(NextIntersection, [NextIntersection.NONE]),
                AVPredicate(Velocity, [mov_predicate]),
                AVPredicate(Rotation, [rot_predicate]),
                AVPredicate(StopAreaNearby, [sign_predicate]),
                AVPredicate(IsZebraNearby,[zebra_predicate]),
                AVPredicate(IsTrafficLightNearby, [traffic_light_predicate]),
                AVPredicate(FrontObjects,[detection_predicate])
        )
        

    def discretize_detections(self, detections:pd.DataFrame):
        tot_count = 0
        for _, row in detections.iterrows():

            category = row['category']
            attribute = str(row['attribute'])
            x_center, y_center = eval(row['bbox_center'])

            # Exclude humans, bicycles and motorcycles, objects not affecting the driver (e.g, barriers, bycicle racks)
            #if 'human'  in category or 'without_rider' in attribute:
            if any(keyword in category for keyword in {'human', 'cycle'}):
                continue

            map_position = self.environment.nusc_map.layers_on_point(x_center, y_center, layer_names=['drivable_area', 'carpark_area']) 
            if map_position['drivable_area']: # Check if (x, y) are in the drivable area else exclude object
                # Exclude parked vehicles and objects in parking area.
                if map_position['carpark_area'] and ('parked' in attribute or 'object' in category):
                    continue
                tot_count+=1
                
        return FrontObjects.YES  if tot_count > 0 else FrontObjects.NO



    def discretize_steering_angle(self, steering_angle: float)->Rotation:
        if steering_angle <= -self.eps_rot:  
            return Rotation.RIGHT
        elif steering_angle <= self.eps_rot:  
            return Rotation.FORWARD
        else:
            return Rotation.LEFT


    def discretize_position(self, x:float,y:float,yaw:float)-> LanePosition:
        block_progress, lane_position = self.environment.get_position_predicates(x,y, yaw, eps=0.3, agent_size=self.agent_size)
        return block_progress, lane_position
    


    def discretize_speed(self, speed:float) -> Velocity:
        if speed <= self.eps_vel: 
            return Velocity.STOPPED
        else:
            if 'a' in self.id:
                return Velocity.MOVING
            else: 
                if speed <= 4.1:
                    return Velocity.LOW
                elif speed <=8.3:
                    return Velocity.MEDIUM
                else:
                    return Velocity.HIGH 
    
    def discretize_stop_line(self, x:float,y:float,yaw:float) -> Tuple[StopAreaNearby, IsZebraNearby, IsTrafficLightNearby]:
        
        # Create a rotated rectangle around the vehicle's current pose
        yaw_in_deg = np.degrees(-(np.pi / 2) + yaw)
        area = create_rectangle((x,y), yaw_in_deg, size=(10,12), shift_distance=6)
        
        stop_area = self.environment.is_near_stop_area(x,y,area)
        if stop_area is None:
            is_stop_nearby = StopAreaNearby.NO
        elif 'STOP_SIGN' == stop_area:
            is_stop_nearby = StopAreaNearby.STOP
        elif 'YIELD' == stop_area:
            is_stop_nearby = StopAreaNearby.YIELD
        elif 'TURN_STOP' == stop_area:
            is_stop_nearby = StopAreaNearby.TURN_STOP
        else:
            is_stop_nearby = StopAreaNearby.NO


        is_zebra_nearby = IsZebraNearby.YES  if self.environment.is_near_ped_crossing(area) else IsZebraNearby.NO
        
        is_traffic_light_nearby = IsTrafficLightNearby.YES  if self.environment.is_near_traffic_light(yaw, area) else IsTrafficLightNearby.NO

        return is_stop_nearby, is_zebra_nearby, is_traffic_light_nearby


    def assign_intersection_actions(self, trajectory:List[Any], intersection_info:List[Tuple[int, Action]], verbose:bool = False) -> List[Any]:
        #TODO: improve typing
        """
        Assigns actions based on intersection information.

        Args:
            trajectory: List containing the discretized trajectory.
            intersection_info: List storing information about intersections as.

        Returns:
            Updated trajectory with assigned actions for intersections.
        """
        for i in range(0, len(trajectory), 2):  # Access states

            for idx, action in intersection_info:
                if 2 * idx > i and 2 * idx < len(trajectory) - 1: # Check if the intersection state (2*idx) comes next the current state (i)
                    state = list(trajectory[i])
                    next_intersect_idx = next((i for i, predicate in enumerate(state) if predicate.predicate.__name__ == 'NextIntersection'), None)
                    state[next_intersect_idx] = action
                    trajectory[i] = tuple(state)
                    break
            if verbose:
                    print(f'frame {int(i/2)} --> {list(trajectory[i])}')
                    if i<len(trajectory) - 1:
                        print(f'action: {get_action_from_id(trajectory[i+1])}')
                    else:
                        print('END')

        return trajectory
    

    @staticmethod
    def determine_intersection_action(start_position:Tuple[float,float,float,float], end_position:Tuple[float,float,float,float]) -> NextIntersection:
        """
        Determine the action at the intersection based on positional changes.
        
        Args:
            start_position (x,y,x1,y1): vector containing starting position just before the intersection and at the beginning of the intersection.
            end_position (x,y,x1,y1): vector containin position at the intersection and just after the intersection.
        Returns:
            Action: NextIntersection.RIGHT, NextIntersection.LEFT, or NextIntersection.STRAIGHT.
        """


        x_1,y_1, x_2, y_2 = start_position
        x_n,y_n, x_n1, y_n1 = end_position

        pre_vector = np.array([x_2 - x_1, y_2 - y_1]) 
        post_vector = np.array([x_n1 - x_n, y_n1 - y_n]) 
        angle = vector_angle(pre_vector, post_vector)
        
        if abs(angle) < np.radians(20):
            return AVPredicate(NextIntersection,[NextIntersection.STRAIGHT])
        elif np.cross(pre_vector, post_vector) > 0:
            return AVPredicate(NextIntersection,[NextIntersection.LEFT])
        else:
            return AVPredicate(NextIntersection,[NextIntersection.RIGHT])

    ##################################
    



    def determine_action(self, next_state:Tuple[float,float,float]) -> Action:
        vel_t1,  acc_t1, steer_t1 = next_state
        if vel_t1 <= self.eps_vel and abs(acc_t1)<self.eps_acc:#is_close(acc_t1,0,self.eps_acc):
            return Action.IDLE
    
        # determine acceleration
        if acc_t1 >= self.eps_acc and vel_t1>self.eps_vel:
            acc_action = Action.GAS
        elif acc_t1 <= -self.eps_acc and vel_t1>self.eps_vel:
            acc_action = Action.BRAKE
        else:
            acc_action = None

        # determine direction
        if steer_t1 <= -self.eps_rot:
            dir_action = Action.TURN_RIGHT
        elif steer_t1 > self.eps_rot:
            dir_action = Action.TURN_LEFT
        else:
            dir_action = Action.STRAIGHT

       # Combine acceleration and direction actions
        if acc_action == Action.GAS:
            if dir_action == Action.TURN_RIGHT:
                return Action.GAS_TURN_RIGHT
            elif dir_action == Action.TURN_LEFT:
                return Action.GAS_TURN_LEFT
            else:
                return Action.GAS
        elif acc_action == Action.BRAKE:
            if dir_action == Action.TURN_RIGHT:
                return Action.BRAKE_TURN_RIGHT
            elif dir_action == Action.TURN_LEFT:
                return Action.BRAKE_TURN_LEFT
            else:
                return Action.BRAKE
        elif acc_action is None:
            # Fallback to direction if no acceleration action was determined
            return dir_action

        # if no other conditions met
        return Action.STRAIGHT



    def _discretize_state_and_action(self, state_0, state_1, observation_0):
        
        """ Given a scene from the dataset, it discretizes the current state (i) and determines the following action. """
        current_state_to_discretize = state_0[self.vehicle_state_columns] 
        discretized_current_state = self.discretize(current_state_to_discretize, observation_0[self.state_columns_for_detection])

        next_state_for_action = state_1[self.state_columns_for_action] 
        action = self.determine_action(next_state_for_action)
        action_id = get_action_id(action)
        
        return discretized_current_state, action_id



    def state_to_str(self,
                     state: Tuple[Union[AVPredicate, ]]
                     ) -> str:
        return ' '.join(str(pred) for pred in state)

    
    def str_to_state(self, state_str: str) -> Tuple[Union[AVPredicate, ]]:

        lane_pos_str, next_inter_str, vel_str, rot_str, stop_str, zebra_str, traffic_light_str, detection_str  = state_str.split(' ')
        position_predicate = LanePosition[lane_pos_str[:-1].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-1].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-1].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-1].split('(')[1]]
        stop_predicate = StopAreaNearby[stop_str[:-1].split('(')[1]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-1].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-1].split('(')[1]] 
        detection_predicate = FrontObjects[detection_str[:-1].split('(')[1]]

        predicates = [
            
            AVPredicate(LanePosition, [position_predicate]),
            AVPredicate(NextIntersection, [intersection_predicate]),
            AVPredicate(Velocity, [mov_predicate]),
            AVPredicate(Rotation, [rot_predicate]),
            AVPredicate(StopAreaNearby, [stop_predicate]),
            AVPredicate(IsZebraNearby, [zebra_predicate]),
            AVPredicate(IsTrafficLightNearby, [traffic_light_predicate]),
            AVPredicate(FrontObjects, [detection_predicate])

        ]
        
        return tuple(predicates)

    

    def all_actions(self):
        return list(Action) 
        
    def get_predicate_space(self):
        all_tuples = []
        for l in LanePosition:
            for n in NextIntersection:
                for v in self.vel_values:
                    for r in Rotation:
                        for s in StopAreaNearby:
                            for z in IsZebraNearby:
                                for t in IsTrafficLightNearby:
                                    for o in FrontObjects: #discretization[self.obj_discretization]:
                                        all_tuples.append((l,n,v,r,s,z,t,o))
        return all_tuples

    def nearest_state(self, state:Tuple[Union[AVPredicate, ]]) -> Tuple[Union[AVPredicate, ]]:
        pass
    
    
    def distance(self, state0:Tuple[Union[AVPredicate, ]], state1: Tuple[Union[AVPredicate, ]]) -> int:
       pass
    
    
