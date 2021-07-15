from gr_nlp_toolkit.data.downloader import Downloader
import os


class DownloaderStub(Downloader):
    def download_processor(self, processor_name: str, target_path: str):
        with open(target_path , 'wb') as f:
            pass