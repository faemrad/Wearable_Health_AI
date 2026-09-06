
from program.tools.file_processor import FileProcessor
from processors.hr_processor import HRProcessor
from processors.activity_predictor import ActivityPredictor

class Program:
    """
    Main class for the program. Responsible for setting up and checking the data
    """
    def __init__(self, file_path: str, frequency: int, column_labels: list[str], use_ecg=False):
        self.file_path = file_path
        self.frequency = frequency
        self.column_labels = column_labels
        self.use_ecg = use_ecg
        self.setup()

    def setup(self):
        pass

    def run(self) -> dict:
        """
        Runs the program to estimate the activity, and extract the heart rate.
        
        Outputs:
            dict: A dictionary containing the activity, confidence score, and heart rate
            
        """
        hr_data = HRProcessor().get_hr_data(self.data, self.frequency, self.use_ecg)
        activity_data = ActivityPredictor.predict(self.data)

        return {"activity": activity_data[0], "confidence": activity_data[1], "hr": hr_data[0], "hr_variability": hr_data[1]}

        # TODO
        pass