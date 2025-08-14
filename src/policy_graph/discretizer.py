from typing import Union, Sequence, List
from enum import Enum
from pgeon.discretizer import Predicate

class AVPredicate(Predicate): #TODO: match pgeon Predicate definition
    def __init__(self, predicate: Union[Enum, type], value: Union[Sequence[Union[Enum, int]], Enum, int]):
        self.predicate = predicate
        # Ensure value is always stored as a list for consistency
        if isinstance(value, (Enum, int)):
            self.value: List[Union[Enum, int]] = [value]
        else:
            self.value: List[Union[Enum, int]] = list(value)
            

    def __str__(self):
        # Handle both Enums and ints in value
        values_str = ",".join(self.format_value(val) for val in self.value)
        return f'{self.predicate.__name__}({values_str})'


    def __lt__(self, other):
        if not isinstance(other, Predicate):
            raise ValueError("Cannot compare Predicate with non-Predicate type.")
        else:
            return hash(self.predicate) < hash(other.predicate)

    @staticmethod
    def format_value(val):
        if isinstance(val, Enum):
            return val.name  # For Enum members, use the name
        else:
            return str(val)  # For integers or other types, convert directly to string

