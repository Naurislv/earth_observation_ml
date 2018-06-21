"""Provide unique genome IDs."""

class IDgen():
    """Generate unique IDs.
    """

    def __init__(self):
        """Keep track of IDs.
        """
        self.current_id = 0
        self.current_gen = 1

    def get_next_id(self):
        """Inreace id by one"""

        self.current_id += 1

        return self.current_id

    def increase_gen(self):
        """Increase gen by one"""

        self.current_gen += 1

    def get_gen(self):
        """get current gen number"""

        return self.current_gen
