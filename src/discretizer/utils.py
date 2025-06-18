from enum import Enum, auto

class Velocity(Enum):
  STOPPED = auto()
  LOW = auto()
  MEDIUM = auto()
  HIGH = auto()
  #VERY_HIGH = auto()
  MOVING = auto()
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Rotation(Enum):
  RIGHT = auto()
  #SLIGHT_RIGHT = auto()
  FORWARD = auto()
  #SLIGHT_LEFT = auto()
  LEFT = auto()

  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class LanePosition(Enum):
    CENTER = auto()
    ALIGNED = auto()
    OPPOSITE = auto()
    NONE = auto() #for all the cases not includend in the previous categories (e.g car headed perpendicular to the road, parkins, etc..)
    #TODO: handle intersections    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'

class BlockProgress(Enum):
    START = auto()
    MIDDLE = auto()
    END = auto()
    INTERSECTION = auto()
    NONE = auto() #for all the cases not includend in the previous categories (e.g. car parkings, walkway)

    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'


class NextIntersection(Enum):
    #Stores where the driver plans to go at the next intersection
    RIGHT = auto()
    LEFT = auto()
    STRAIGHT = auto()
    NONE = auto()

    def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Detection:
    discretizations = {
        "multiple": ["0", "1-3", "4+"],
        "binary": ["NO", "YES"]
    }

    def __init__(self, count=0, discretization="binary"):
        self.chunks = self.discretizations[discretization]

        if isinstance(count, str):
            self.count = count
        elif discretization == "multiple":
            
            if count == 0:
                self.count = self.chunks[0]
            elif count < 4:
                self.count = self.chunks[1]
            else:
                self.count = self.chunks[2]
        elif discretization == "binary":
            self.count = self.chunks[1] if count > 0 else self.chunks[0]
    
    def __str__(self) -> str:
        return f'{self.count}'

    def __eq__(self, other):
        return self.count == other.count
    
    def __hash__(self):
        return hash(self.count)

class FrontObjects(Detection):
    pass

class PedestrianNearby(Detection):
    pass


class IsTwoWheelNearby(Enum): 
    # Includes bycicles and scooters
    YES = auto()
    NO = auto()
    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'



class IsTrafficLightNearby(Enum):
  YES = auto()
  NO = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class IsZebraNearby(Enum):
  # Includes pedestrian crossings and turn stops
  YES = auto()
  NO = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'
    

class StopAreaNearby(Enum): 
  # Includes stop signs and yield signs
  STOP = auto()
  YIELD = auto()
  TURN_STOP = auto()
  NO = auto()
  #PED_CROSS = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'



class IdleTime:
    def __init__(self, count=0):
        self.chunks = ["0", "4", "5+"]

        if isinstance(count, str):
            self.count = count
        else:
            if count ==0:
                self.count = self.chunks[0]
            elif count <=4:
                self.count = self.chunks[1]
            else:
                self.count = self.chunks[2]
    
    def __str__(self) -> str:
        return f'{self.count}'

    def __eq__(self, other):
        return self.count == other.count
    
    def __hash__(self):
        return hash(self.count)



class Action(Enum):
  IDLE = auto()  
  TURN_LEFT = auto() 
  TURN_RIGHT = auto() 
  GAS = auto() 
  BRAKE = auto() 
  STRAIGHT = auto() 
  GAS_TURN_RIGHT= auto() 
  GAS_TURN_LEFT= auto() 
  BRAKE_TURN_RIGHT = auto()  
  BRAKE_TURN_LEFT = auto() 


def is_close(a, b, eps=0.1):
    return abs(a - b) < eps

def get_action_id(action):
    return action.value
    
def get_action_from_id(action_id):
    for action in Action:
        if action.value == action_id:
            return action
    raise ValueError("Invalid action ID")


def calculate_object_distance(value1, value2):
        """
        Calculate the object distance based on the given rule:
        dist(0, 1-3)= 2, dist(0, 4+)= 3, dist(1-3, 4+) = 1
        """
        if value1 == '0':
            if value2 == '1-3':
                return 2
            elif value2 == '4+':
                return 3
        elif value1 == '1-3':
            if value2 == '0':
                return 2
            elif value2 == '4+':
                return 1
        elif value1 == '4+':
            if value2 == '0':
                return 3
            elif value2 == '1-3':
                return 1
        return 0
    


def calculate_velocity_distance(value1, value2):
        """
        Calculate the object distance based on the given rule:
        dist(not moving, slow)= 2, dist(not moving, medium)= 3, dist(not moving, high) = 4
        dist(slow, medium) = 1, dist(slow,high) = 2,
        dist(medium,high) = 1
        """

        if value1 == Velocity.STOPPED:
            if value2 == Velocity.LOW:
                return 2
            elif value2 == Velocity.MEDIUM:
                return 3
            elif value2 == Velocity.HIGH:
                return 4
        elif value1 == Velocity.LOW:
            if value2 == Velocity.STOPPED:
                return 2
            elif value2 == Velocity.MEDIUM:
                return 1
            elif value2 == Velocity.HIGH:
                return 2
        elif value1 == Velocity.MEDIUM:
            if value2 == Velocity.STOPPED:
                return 3
            elif value2 == Velocity.LOW:
                return 1
            elif value2 == Velocity.HIGH:
                return 1
        elif value1 == Velocity.HIGH:
            if value2 == Velocity.STOPPED:
                return 4
            elif value2 == Velocity.LOW:
                return 2
            elif value2 == Velocity.MEDIUM:
                return 1
        return 0  


