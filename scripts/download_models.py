# importing libraries

from utils.model_manager import ModelManager


def main() -> None:

    manager = ModelManager()

    print()
    print("=" * 70)
    print(" MeterSerial-VLM Model Downloader ")
    print("=" * 70)
    print()

    results = manager.download_all()

    print()
    print("=" * 70)
    print(" Download Summary ")
    print("=" * 70)

    for model_name in manager.list_models():

        status = "READY" if results[model_name] else "FAILED"

        print(f"{model_name:<20} {status}")

    print()


if __name__ == "__main__":
    main()