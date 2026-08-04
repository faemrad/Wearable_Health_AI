
from program.processors.file_processor import FileProcessor

class Program:
    """
    Main class for the program. Responsible for setting up and checking the data
    """
    def __init__(self, file_path: str, frequency: int, column_labels: list[str]):
        self.file_path = file_path
        self.frequency = frequency
        self.column_labels = column_labels
        self.setup()

    def setup(self):
        self.data = FileProcessor.process_file(self.file_path, self.column_labels, self.frequency)
        pass

    def run(self):
        # TODO
        pass