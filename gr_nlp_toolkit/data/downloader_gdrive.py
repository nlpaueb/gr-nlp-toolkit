from gr_nlp_toolkit.data.downloader import Downloader
import gdown


class GDriveDownloader(Downloader):
    def __init__(self):
        self.urls = {
            'pos': 'https://drive.google.com/uc?id=1Or5HDk1kVnxI3_w0fwgR8-dzO0jvcc_L',  # pos link
            'ner': 'https://drive.google.com/uc?id=1fx0pHtcN7F2Vj9L8y5TUpbjSqKTUaT3i',  # ner link
            'dp': 'https://drive.google.com/uc?id=1NhEqmLBf67Ydw-LdI7eB-f0afMPgNSmG'  # dp link
        }

    def download_processor(self, processor_name: str, target_path: str):
        gdown.download(self.urls[processor_name], output=target_path, quiet=False)
