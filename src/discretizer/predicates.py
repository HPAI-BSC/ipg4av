from enum import Enum, auto

class Velocity(Enum):
  STOPPED = auto()
  LOW = auto()
  MEDIUM = auto()
  HIGH = auto()
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
    NONE = auto() #for all the cases not includend in the previous categories (e.g. car parkings, walkways)

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



class FrontObjects(Enum):
    YES = auto()
    NO = auto()
    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'


class PedestrianNearby(Enum):
    YES = auto()
    NO = auto()
    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'


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


#def is_close(a, b, eps=0.1):
#     return abs(a - b) < eps

def get_action_id(action):
    return action.value
    
def get_action_from_id(action_id):
    for action in Action:
        if action.value == action_id:
            return action
    raise ValueError("Invalid action ID")

