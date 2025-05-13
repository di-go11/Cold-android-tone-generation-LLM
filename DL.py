from huggingface_hub import snapshot_download
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os


class DL(ABC):
    @abstractmethod
    def download(self):
        pass


class HummingFaceDL(DL):
    def __init__(self):
        load_dotenv()
        # Hagging FaceのリポジトリのID
        self.repo_id = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
        # ローカルに保存するディレクトリ
        self.local_dir = "./LLM/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
        # トークン
        self.token = os.getenv("MY_TOKEN")

        self.download()

    def download(self):
        try:
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.local_dir,
                token=self.token,
            )
            print("✅ Downloaded successfully.")
        except Exception as e:
            print("❌ Download failed:", e)

if __name__ == "__main__":
    dl = HummingFaceDL()