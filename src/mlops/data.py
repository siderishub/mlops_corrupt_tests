import os
import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(
    raw_dir: str = "../../data/raw", 
    processed_dir: str = "../../data/processed"
) -> None:
    """Process raw data and save it to the processed directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    raw_dir = os.path.join(script_dir, raw_dir)
    processed_dir = os.path.join(script_dir, processed_dir)

    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(os.path.join(raw_dir, f"train_images_{i}.pt")))
        train_target.append(torch.load(os.path.join(raw_dir, f"train_target_{i}.pt")))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(os.path.join(raw_dir, "test_images.pt"))
    test_target: torch.Tensor = torch.load(os.path.join(raw_dir, "test_target.pt"))

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    os.makedirs(processed_dir, exist_ok=True)  # Create the processed directory if it doesn't exist
    torch.save(train_images, os.path.join(processed_dir, "train_images.pt"))
    torch.save(train_target, os.path.join(processed_dir, "train_target.pt"))
    torch.save(test_images, os.path.join(processed_dir, "test_images.pt"))
    torch.save(test_target, os.path.join(processed_dir, "test_target.pt"))


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    processed_dir = os.path.join(script_dir, "data/processed")

    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
