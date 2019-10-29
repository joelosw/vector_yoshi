import configparser
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

class img_prediction(object):

    def __init__(self, config_file_path = r'azure_config.txt'):       
        config_parser = configparser.RawConfigParser()
        config_parser.read(config_file_path)

        self.ENDPOINT = config_parser.get('CustomVision', 'endpoint')



#        self.training_key = config_parser.get('CustomVision', 'training_key')
#        self.prediction_key = config_parser.get('CustomVision', 'prediction_key')
#        self.prediction_resource_id = config_parser.get('CustomVision', 'ressource_id')
#        self.publish_iteration_name = config_parser.get('CustomVision', 'publish_iteration_name')
#        self.project_id = config_parser.get('CustomVision', 'project_id')

        self.training_key = config_parser.get('CustomVision', 'training_key')
        self.prediction_key = config_parser.get('CustomVision', 'prediction_key')
        self.prediction_resource_id = config_parser.get('CustomVision', 'ressource_id')
        self.publish_iteration_name = config_parser.get('CustomVision', 'publish_iteration_name')
        self.project_id = config_parser.get('CustomVision', 'project_id')

        print('Endpoint: ', self.ENDPOINT)


        
        
        self.predictor = CustomVisionPredictionClient('627746c9a7084f74a448c0040ad15a32', 'https://joelcustomvision.cognitiveservices.azure.com/')
    
        
        



    def predict(self, img_path):
        # Open the image and get back the prediction results as a dict with tuple (left, top, width, height)
        with open(img_path, mode="rb") as image_to_predict:
            results = self.predictor.detect_image('002e7a08-8696-4ca8-8769-fe0cbc2bd9b0', "Iteration0.1", image_to_predict)

        # Display the results, and return them as a dict (Tuple of four for ecery Tag) 
        tag_dict = dict()   
        for prediction in results.predictions:
            print("\t" + prediction.tag_name + 
            ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, 
            prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
            if prediction.probability > 0.5:
                tag_dict[prediction.tag_name] = (prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
        
        return tag_dict


if __name__ == '__main__':
    prediction = img_prediction()
    results = prediction.predict('./balloon_pic.jpg')
    print(results)
