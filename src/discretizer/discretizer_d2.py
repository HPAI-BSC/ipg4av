from discretizer.predicates import IdleTime, Velocity, Rotation, IsTrafficLightNearby,IsZebraNearby,StopAreaNearby, PedestrianNearby, IsTwoWheelNearby,LanePosition, FrontObjects, BlockProgress, NextIntersection
from discretizer.discretizer_d1 import AVDiscretizerD1
from policy_graph.environment import AVEnvironment
from policy_graph.discretizer import AVPredicate
import numpy as np
from typing import Tuple, Union


class AVDiscretizerD2(AVDiscretizerD1):
    def __init__(self, environment: AVEnvironment, id:str):
        super().__init__(environment, id)
               
    
    ##################################
    ### Overridden Methods
    ##################################

    
    def discretize(self, state: np.ndarray, detections=None) -> Tuple[AVPredicate, ...]:
        predicates = super().discretize(state, detections)
        return (AVPredicate(IdleTime, [IdleTime(0)]), ) + predicates

    
    def str_to_state(self, state_str: str) -> Tuple[Union[AVPredicate, ]]:
        idle_time_str,pedestrian_str, bike_str, block_str, lane_pos_str, next_inter_str, vel_str, rot_str, stop_str, zebra_str, traffic_light_str, detection_str = state_str.split(' ')
        idle_time_predicate = idle_time_str[:-1].split('(')[1]
        pedestrian_predicate = PedestrianNearby[pedestrian_str[:-1].split('(')[1]] #pedestrian_str[:-1].split('(')[1] 
        bike_predicate = IsTwoWheelNearby[bike_str[:-1].split('(')[1]] 
        progress_predicate = BlockProgress[block_str[:-1].split('(')[1]] 
        position_predicate = LanePosition[lane_pos_str[:-1].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-1].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-1].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-1].split('(')[1]]
        stop_predicate = StopAreaNearby[stop_str[:-1].split('(')[1]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-1].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-1].split('(')[1]]         
        detection_predicate = FrontObjects[detection_str[:-1].split('(')[1]] #detection_str[:-1].split('(')[1]

        predicates = [
            AVPredicate(IdleTime, [IdleTime(idle_time_predicate)]),
            AVPredicate(PedestrianNearby, [pedestrian_predicate]),#PedestrianNearby(n_pedestrians)]),
            AVPredicate(IsTwoWheelNearby, [bike_predicate]),
            AVPredicate(BlockProgress, [progress_predicate]),
            AVPredicate(LanePosition, [position_predicate]),
            AVPredicate(NextIntersection, [intersection_predicate]),
            AVPredicate(Velocity, [mov_predicate]),
            AVPredicate(Rotation, [rot_predicate]),
            AVPredicate(StopAreaNearby, [stop_predicate]),
            AVPredicate(IsZebraNearby, [zebra_predicate]),
            AVPredicate(IsTrafficLightNearby, [traffic_light_predicate]),
            AVPredicate(FrontObjects, [detection_predicate])#FrontObjects(detection_value)])
        ]
        return tuple(predicates)

             
    

    def get_predicate_space(self):
        all_tuples = []
        for i in IdleTime.chunks:
            for p in PedestrianNearby:
                for tw in IsTwoWheelNearby:
                    for b in BlockProgress:
                        for l in LanePosition:
                            for n in NextIntersection:
                                for v in self.vel_values: 
                                    for r in Rotation:
                                        for s in StopAreaNearby:
                                            for z in IsZebraNearby:
                                                for t in IsTrafficLightNearby:
                                                    for o in FrontObjects:
                                                            all_tuples.append((i, p, tw, b, l,n,v,r,s,z,t,o))
        return all_tuples
    

    
