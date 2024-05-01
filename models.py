from typing import List


class Result:
    def __init__(self, inputs: List[int], outputs: List[int], targets: List[int] = None):
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets

    def __hash__(self):
        return hash((self.inputs, self.outputs))

    # def __eq__(self, other):
    #     return self.inputs == other.inputs and self.outputs == other.outputs
