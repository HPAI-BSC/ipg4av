from typing import Optional, List, Dict
from pgeon.discretizer import Predicate
from pgeon.desire import Desire
from enum import Enum



class DesireCategory(Enum):
    CRUISING = "Cruising"
    TRAFFIC_SIGN = "Traffic Sign"
    VULNERABLE_USERS = "Vulnerable Users"
    UNSAFE = "Unsafe Behaviour" 
    ANY= "Any"
    

class AVDesire(Desire):
    def __init__(self, name: str, actions: Optional[List[int]], clause: Dict[Predicate, List[str]], category:DesireCategory= None):
        self.name = name
        self.actions = actions
        self.clause = clause # dictionary where keys are predicates that should be in Sd, and values the list of possible values they can have.
        self.category = category 


    def __repr__(self):
        return f"Desire[{self.name}]=<{self.clause}, {self.actions}>"

    def __str__(self):
        return f"Desire[{self.name}]=<{self.clause}, {self.actions}>"


