from src.radials import Radial

class MultiModalRadial(Radial):
    def __init__(self, d):
        super().__init__(d)

    def extract_mode(self, idx):
        """
        Extract the idx-th mode of the multimodal radial.
        :param idx: int
        :return: Radial
        """
        raise NotImplementedError