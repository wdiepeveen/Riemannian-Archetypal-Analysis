from src.radials.multimodal import MultiModalRadial

class IntersectedRadial(MultiModalRadial):
    def __init__(self, radials, aggregation='min'):
        super().__init__(d=radials[0].d, radials=radials, aggregation=aggregation)