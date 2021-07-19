from abc import ABC, abstractmethod


class Downloader(ABC):

    @abstractmethod
    def download_processor(self, processor_name: str, target_path: str):
        pass
