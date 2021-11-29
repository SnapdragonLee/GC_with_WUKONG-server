import mini.mini_sdk as MiniSdk
from mini.apis.api_sence import TakePicture, TakePictureType
import client


def take_picture():
    (resultType, response) = TakePicture(take_picture_type=TakePictureType.IMMEDIATELY).execute()

