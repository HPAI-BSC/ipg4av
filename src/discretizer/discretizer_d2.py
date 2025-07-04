from discretizer.utils import calculate_object_distance, IdleTime, Detection, Velocity, Rotation, IsTrafficLightNearby,IsZebraNearby,StopAreaNearby, PedestrianNearby, IsTwoWheelNearby,LanePosition, FrontObjects, BlockProgress, NextIntersection
from discretizer.discretizer_d1 import AVDiscretizerD1
from policy_graph.environment import AVEnvironment
from policy_graph.discretizer import AVPredicate
import numpy as np
from typing import Tuple, Union


class AVDiscretizerD2(AVDiscretizerD1):
    def __init__(self, environment: AVEnvironment, vel_discretization = 'binary', obj_discretization = "binary", id='2a'):
        super().__init__(environment)
        
        self.environment = environment
        self.obj_discretization = obj_discretization
        self.vel_discretization = vel_discretization
        self.id = id
        id_to_eps = { #TODO:
            '2a': 8,
            '2b': 10            
        }
        self.eps = id_to_eps.get(self.id)
        
    ##################################
    ### Overridden Methods
    ##################################

    
    def discretize(self, state: np.ndarray, detections=None) -> Tuple[AVPredicate, ...]:
        predicates = super().discretize(state, detections)
        return (AVPredicate(IdleTime, [IdleTime(0)]), ) + predicates

    
    def str_to_state(self, state_str: str) -> Tuple[Union[AVPredicate, ]]:
        idle_time_str,pedestrian_str, bike_str, block_str, lane_pos_str, next_inter_str, vel_str, rot_str, stop_str, zebra_str, traffic_light_str, detection_str = state_str.split(' ')
        idle_time_predicate = idle_time_str[:-1].split('(')[1]
        n_pedestrians = pedestrian_str[:-1].split('(')[1] 
        bike_predicate = IsTwoWheelNearby[bike_str[:-1].split('(')[1]] 
        progress_predicate = BlockProgress[block_str[:-1].split('(')[1]] 
        position_predicate = LanePosition[lane_pos_str[:-1].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-1].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-1].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-1].split('(')[1]]
        stop_predicate = StopAreaNearby[stop_str[:-1].split('(')[1]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-1].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-1].split('(')[1]]         
        detection_value = detection_str[:-1].split('(')[1]

        predicates = [
            AVPredicate(IdleTime, [IdleTime(idle_time_predicate)]),
            AVPredicate(PedestrianNearby, [PedestrianNearby(n_pedestrians)]),
            AVPredicate(IsTwoWheelNearby, [bike_predicate]),
            AVPredicate(BlockProgress, [progress_predicate]),
            AVPredicate(LanePosition, [position_predicate]),
            AVPredicate(NextIntersection, [intersection_predicate]),
            AVPredicate(Velocity, [mov_predicate]),
            AVPredicate(Rotation, [rot_predicate]),
            AVPredicate(StopAreaNearby, [stop_predicate]),
            AVPredicate(IsZebraNearby, [zebra_predicate]),
            AVPredicate(IsTrafficLightNearby, [traffic_light_predicate]),
            AVPredicate(FrontObjects, [FrontObjects(detection_value)])
        ]
        return tuple(predicates)

        

   

    def nearest_state(self, state):
        idle_time, pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, detections = state

        #NOTE: the order of the following conditions affects the yielded AVPredicates, thus introducing bias. Prioritize more influent predicates.
        # Generate nearby positions considering discretization
        

        for v in self.vel_values:
            if [v] != velocity.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, AVPredicate(Velocity, [v]), rotation, stop_sign, zebra_crossing, traffic_light, detections
        
        for r in Rotation:
            if [r]!= rotation.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity, AVPredicate(Rotation, [r]), stop_sign, zebra_crossing, traffic_light, detections


        for s in StopAreaNearby:
            if [s]!= stop_sign.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity,rotation, AVPredicate(StopAreaNearby, [s]), zebra_crossing, traffic_light, detections
        
        for z in IsZebraNearby:
            if [z]!= zebra_crossing.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity,rotation, stop_sign, AVPredicate(IsZebraNearby, [z]), traffic_light, detections
   
        for t in IsTrafficLightNearby:
            if [t]!= traffic_light.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity,rotation, stop_sign, zebra_crossing, AVPredicate(IsTrafficLightNearby,[ t]), detections

        
        for l in LanePosition:
            if [l] != lane_position.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, AVPredicate(LanePosition, [l]), next_intersection, velocity,rotation, stop_sign, zebra_crossing,traffic_light, detections

        for n in NextIntersection:
            if [n] != next_intersection.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, AVPredicate(NextIntersection,[n]), velocity,rotation, stop_sign, zebra_crossing, traffic_light, detections
        
        for p in PedestrianNearby.discretizations[self.obj_discretization]:
            if [p] != pedestrian_cross.value:
                yield idle_time,AVPredicate(PedestrianNearby, [PedestrianNearby(p, self.obj_discretization)]), two_wheeler, block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, detections
        
        for t in IsTwoWheelNearby:
            if [t] != two_wheeler.value:
                yield idle_time,pedestrian_cross, AVPredicate(IsTwoWheelNearby, [t]), block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, detections
        
        for b in BlockProgress:
            if [b] != block_progress.value:
                yield idle_time,pedestrian_cross, two_wheeler, AVPredicate(BlockProgress, [b]), lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, detections
        
            
        for value in Detection.discretizations[self.obj_discretization]:
            if [value] != detections[0].value:
                yield idle_time, pedestrian_cross, two_wheeler, block_progress, lane_position, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, AVPredicate(FrontObjects, [FrontObjects(value, self.obj_discretization)])  

        for i in IdleTime.chunks:
            if [i] != idle_time.value:
                yield AVPredicate(IdleTime, [IdleTime(i)]),pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, detections
        

        for i in IdleTime.chunks:
            for p in PedestrianNearby.discretizations[self.obj_discretization]:
                for tw in IsTwoWheelNearby:
                    for b in BlockProgress:
                        for l in LanePosition:
                            for n in NextIntersection:
                                for v in self.vel_values:
                                    for r in Rotation:
                                        for s in StopAreaNearby:
                                            for z in IsZebraNearby:
                                                for t in IsTrafficLightNearby:
                                                    for cam in Detection.discretizations[self.obj_discretization]:
                                                            nearby_state =  AVPredicate(IdleTime, [IdleTime(i)]), AVPredicate(PedestrianNearby, [PedestrianNearby(p, self.obj_discretization)]), AVPredicate(IsTwoWheelNearby, [tw]), AVPredicate(BlockProgress, [b]), \
                                                                    AVPredicate(LanePosition, [l]), AVPredicate(NextIntersection, [n]), AVPredicate(Velocity, [v]), \
                                                                    AVPredicate(Rotation, [r]), AVPredicate(StopAreaNearby, [s]), AVPredicate(IsZebraNearby, [z]), \
                                                                    AVPredicate(IsTrafficLightNearby, [t]), AVPredicate(FrontObjects, [FrontObjects(cam, self.obj_discretization)])
                                                            
                                                            if 1 < self.distance(state, nearby_state) < self.eps:
                                                                yield nearby_state



    def distance(self, original_pred, nearby_pred):
        '''
        Function that returns the distance between 2 states.
        '''

        o_idle_time, o_pedestrian_cross, o_two_wheeler, o_block_progress, _, _, _, _, _, _, _, _ = original_pred
        n_idle_time, n_pedestrian_cross, n_two_wheeler, n_block_progress, _, _, _, _, _, _, _, _ = nearby_pred

        
        distance = super().distance(original_pred[4:], nearby_pred[4:]) + int(o_idle_time.value != n_idle_time.value)

        if self.obj_discretization == 'binary':
            distance += int(o_pedestrian_cross.value != n_pedestrian_cross.value)
        else:
            distance += calculate_object_distance(o_pedestrian_cross.value[0].count, n_pedestrian_cross.value[0].count)

        return distance + int(o_two_wheeler.value !=n_two_wheeler.value) + int(o_block_progress.value !=n_block_progress.value)         
    

    def get_predicate_space(self):
        all_tuples = []
        for i in IdleTime.chunks:
            for p in PedestrianNearby.discretizations[self.obj_discretization]:
                for tw in IsTwoWheelNearby:
                    for b in BlockProgress:
                        for l in LanePosition:
                            for n in NextIntersection:
                                for v in self.vel_values: 
                                    for r in Rotation:
                                        for s in StopAreaNearby:
                                            for z in IsZebraNearby:
                                                for t in IsTrafficLightNearby:
                                                    for cam in Detection.discretizations[self.obj_discretization]:
                                                            all_tuples.append((i, p, tw, b, l,n,v,r,s,z,t,cam))
        return all_tuples
    

    
