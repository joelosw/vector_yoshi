import configparser
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import cv2
##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
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


        
        self.predictor = CustomVisionPredictionClient(self.prediction_key, self.ENDPOINT)
    
        
        



    def predict(self, img_path):
        # Open the image and get back the prediction results as a dict with tuple (left, top, width, height)
        with open(img_path, mode="rb") as image_to_predict:
            results = self.predictor.detect_image('002e7a08-8696-4ca8-8769-fe0cbc2bd9b0', self.publish_iteration_name, image_to_predict)

        # Display the results, and return them as a dict (Tuple of four for ecery Tag) 
        tag_dict = dict()   
        for prediction in results.predictions:
            print("\t" + prediction.tag_name + 
            ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, 
            prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
            probability = 0.5
            if prediction.probability > probability:
                probability = prediction.probability
                tag_dict[prediction.tag_name] = (prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
        
        return tag_dict


##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
def draw_bounding_boxes(im_path, result_dict):
    img = cv2.imread(im_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    for result in result_dict.values():
        left = result[0]* width
        top = result[1]* height
        w=result[2]* width
        h = result[3]* height
        cv2.rectangle(img,(int(left), int(top)),(int(left + w),int(top + h)), (0, 0, 255), 5)
  
    window = cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
if __name__ == '__main__':
    print('WARNING!!!!: CLOSE IMAGES WITH ENTER TO CONITNUE PROGRAMM!!!!')
    prediction = img_prediction()
    results = prediction.predict('./balloon_pic.jpg')
    print(results)
    draw_bounding_boxes('./balloon_pic.jpg', results)
    

    results2 = prediction.predict('./balloon_and_robot.jpg')
    print(results2)
    draw_bounding_boxes('./balloon_and_robot.jpg', results2)

