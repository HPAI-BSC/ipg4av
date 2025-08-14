from policy_graph.desire import AVDesire, DesireCategory
from discretizer.predicates import IsTrafficLightNearby, StopAreaNearby, IsTwoWheelNearby, Rotation, Velocity, IsZebraNearby, LanePosition, PedestrianNearby, BlockProgress,NextIntersection


# ======================
# Any Desire
# ======================
ANY = AVDesire("any", None, set(), DesireCategory.ANY)


# ======================
# Cruising Desires
# ======================

LANE_KEEPING = AVDesire("lane_keeping", [4, 5, 6], {
    LanePosition: [LanePosition.ALIGNED], Rotation: [Rotation.FORWARD], 
    NextIntersection:[NextIntersection.NONE, NextIntersection.STRAIGHT], 
    Velocity: [Velocity.HIGH, Velocity.LOW, Velocity.MEDIUM, Velocity.MOVING]}, 
    DesireCategory.CRUISING)

TURN_LEFT = AVDesire("turn_left", [2, 8, 10], {
    BlockProgress: [BlockProgress.END], NextIntersection: [NextIntersection.LEFT],
    Rotation: [Rotation.LEFT], Velocity: [Velocity.HIGH, Velocity.LOW, Velocity.MEDIUM, Velocity.MOVING]},
    DesireCategory.CRUISING
)

TURN_RIGHT = AVDesire("turn_right", [3, 7, 9], {
    BlockProgress: [BlockProgress.END], NextIntersection: [NextIntersection.RIGHT],
    Rotation: [Rotation.RIGHT], Velocity: [Velocity.HIGH, Velocity.LOW, Velocity.MEDIUM, Velocity.MOVING]},
    DesireCategory.CRUISING
)


LANE_CHANGE_LEFT = AVDesire("lane_change_left", [2,4,5,6,8,10], {
    Rotation: [Rotation.LEFT], LanePosition: [LanePosition.CENTER], 
    BlockProgress: [BlockProgress.START, BlockProgress.MIDDLE, BlockProgress.END], 
    Velocity: [Velocity.HIGH, Velocity.LOW, Velocity.MEDIUM, Velocity.MOVING]},
    DesireCategory.CRUISING) 

LANE_CHANGE_RIGHT = AVDesire("lane_change_right", [3,4,5,6,7,9], {
    Rotation: [Rotation.RIGHT], LanePosition: [LanePosition.CENTER], 
    BlockProgress: [BlockProgress.START, BlockProgress.MIDDLE, BlockProgress.END], 
    Velocity: [Velocity.HIGH, Velocity.LOW, Velocity.MEDIUM, Velocity.MOVING]},
    DesireCategory.CRUISING) 

#TODO: handle missing lane dividers at intersections (not possible with nuScenes)



# ======================
# Traffic Sign Desires
# ======================

APPROACH_TRAFFIC_LIGHT = AVDesire("approach_traffic_light", [1,5,9,10], {
    IsTrafficLightNearby:  [IsTrafficLightNearby.YES], Velocity: [Velocity.LOW,Velocity.MEDIUM]},
    DesireCategory.TRAFFIC_SIGN) 

APPROACH_STOP_SIGN = AVDesire("approach_stop_sign", [1, 5, 9, 10], {
    StopAreaNearby: [StopAreaNearby.STOP],Velocity: [Velocity.MEDIUM, Velocity.LOW], 
    IsTrafficLightNearby: [IsTrafficLightNearby.NO]},
    DesireCategory.TRAFFIC_SIGN)




# ======================
# Vulnerable Road Users
# ======================

PEDS_AT_CROSSWALK= AVDesire("peds_at_crosswalk", [1,5, 9, 10], {
    IsZebraNearby: [IsZebraNearby.YES],Velocity:[Velocity.LOW, Velocity.MEDIUM,Velocity.HIGH, Velocity.MOVING], 
    PedestrianNearby: [PedestrianNearby.YES]},
    DesireCategory.VULNERABLE_USERS)

JAYWALKING_PEDS= AVDesire("jaywalking_peds", [1,5, 9, 10], {
    IsZebraNearby: [IsZebraNearby.NO],Velocity:[Velocity.LOW, Velocity.MEDIUM,Velocity.HIGH, Velocity.MOVING], 
    PedestrianNearby: [PedestrianNearby.YES]},
    DesireCategory.VULNERABLE_USERS) 

#NOTE: The vehicle does not always desires to stop as some time the pedestrians may have already started crossing.



# ======================
# Undesirable Behaviors
# ======================

IGNORE_TWO_WHEEL = AVDesire('ignore_two_wheel', [4], {
    LanePosition: [LanePosition.ALIGNED], Rotation: [Rotation.FORWARD], IsTwoWheelNearby: [IsTwoWheelNearby.YES], 
    Velocity: [Velocity.HIGH]},
    DesireCategory.UNSAFE)


IGNORE_PEDS_HIGH = AVDesire("ignore_peds_high", [2,3,4,6,7,8], {
    Velocity:[ Velocity.MEDIUM,Velocity.HIGH], PedestrianNearby: [PedestrianNearby.YES]},
    DesireCategory.UNSAFE) 
IGNORE_PEDS_LOW = AVDesire("ignore_peds_low", [4,7,8], {
    Velocity:[ Velocity.LOW], PedestrianNearby: [PedestrianNearby.YES]},
    DesireCategory.UNSAFE) 
#NOTE: The vehicle does not always desires to stop as some time the pedestrians may have already started crossing.


IGNORE_STOP_SIGN = AVDesire("ignore_stop_sign", [2,3,4,6,7,8], {
    IsTrafficLightNearby: [IsTrafficLightNearby.NO], StopAreaNearby: [StopAreaNearby.STOP], 
    Velocity: [Velocity.LOW, Velocity.HIGH, Velocity.MEDIUM, Velocity.MOVING]},
    DesireCategory.UNSAFE)

OUT_OF_DRIVING_AREA = AVDesire("out_of_driving_area", [1,2,3,4,5,6,7,8,9,10], 
                               {BlockProgress: [BlockProgress.NONE]},
                               DesireCategory.UNSAFE)




# For easy access: 

DESIRE_MAPPING = {
    #ANY.name: ANY,
    LANE_KEEPING.name: LANE_KEEPING,
    TURN_LEFT.name: TURN_LEFT,
    TURN_RIGHT.name: TURN_RIGHT,
    LANE_CHANGE_LEFT.name: LANE_CHANGE_LEFT,
    LANE_CHANGE_RIGHT.name: LANE_CHANGE_RIGHT,
    APPROACH_TRAFFIC_LIGHT.name: APPROACH_TRAFFIC_LIGHT,
    APPROACH_STOP_SIGN.name: APPROACH_STOP_SIGN,
    PEDS_AT_CROSSWALK.name: PEDS_AT_CROSSWALK,
    JAYWALKING_PEDS.name: JAYWALKING_PEDS,
    IGNORE_TWO_WHEEL.name: IGNORE_TWO_WHEEL,
    IGNORE_PEDS_HIGH.name: IGNORE_PEDS_HIGH,
    IGNORE_PEDS_LOW.name: IGNORE_PEDS_LOW,
    IGNORE_STOP_SIGN.name: IGNORE_STOP_SIGN,
    OUT_OF_DRIVING_AREA.name: OUT_OF_DRIVING_AREA
}