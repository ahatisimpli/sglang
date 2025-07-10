from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    SIMPLE_SPEC = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_simple_spec(self):
        return self == SpeculativeAlgorithm.SIMPLE_SPEC
    
    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "SIMPLE_SPEC": SpeculativeAlgorithm.SIMPLE_SPEC,       
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
