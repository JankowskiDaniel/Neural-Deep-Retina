from torch.utils.data import DataLoader
from typing import NamedTuple, Tuple
from data_handlers import CurriculumBaselineRGBDataset
from data_models.config_models import CurriculumSchedule


class CurriculumDatasets(NamedTuple):
    train_dataset: CurriculumBaselineRGBDataset
    val_dataset: CurriculumBaselineRGBDataset


class CurriculumDataloaders(NamedTuple):
    train_dataloader: DataLoader
    val_dataloader: DataLoader


class CurriculumHandler:
    """
    Handles curriculum learning by managing dataloaders and datasets according
    to a specified schedule.
    Attributes:
        dataloaders (CurriculumDataloaders): The default dataloaders for
        training and validation.
        is_curriculum (bool): Flag indicating whether curriculum learning
        is enabled.
        curriculum_datasets (CurriculumDatasets | None): The datasets used for
        curriculum learning.
        curriculum_schedule (CurriculumSchedule | None): The schedule defining
        the stages of curriculum learning.
        upcoming_stage (int): The index of the next stage in the curriculum
        schedule.
        logger (Logger): Logger for logging information.
    Methods:
        get_dataloaders(epoch: int) -> Tuple[DataLoader, DataLoader]:
            Returns the appropriate dataloaders for the given epoch,
            either default or curriculum-based.
        get_curriculum_dataloaders(epoch: int) -> Tuple[
            DataLoader,
            DataLoader
        ]:
            Returns the curriculum dataloaders for the given epoch,
            updating the stage if necessary.
    """

    def __init__(
        self,
        curriculum_dataloaders: CurriculumDataloaders,
        batch_size: int,
        curriculum_datasets: CurriculumDatasets | None = None,
        is_curriculum: bool = False,
        curriculum_schedule: CurriculumSchedule | None = None,
        # params for data loaders
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> None:
        self.dataloaders = curriculum_dataloaders

        # Check requirements for curriculum learning
        if is_curriculum:
            assert curriculum_datasets is not None
            assert isinstance(
                curriculum_datasets.train_dataset, CurriculumBaselineRGBDataset
            )
            assert isinstance(
                curriculum_datasets.val_dataset, CurriculumBaselineRGBDataset
            )
            assert curriculum_schedule is not None
        self.is_curriculum = is_curriculum
        self.curriculum_datasets = curriculum_datasets
        self.curriculum_schedule = curriculum_schedule

        self.upcoming_stage: int = 0

        # params for data loaders
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Workaround for circular imports
        from utils import get_logger

        self.logger = get_logger(__name__)

    def update_datasets(self, sigma: float) -> None:
        self.curriculum_datasets.train_dataset.update_data(sigma=sigma)  # type: ignore
        self.curriculum_datasets.val_dataset.update_data(sigma=sigma)  # type: ignore

    def update_dataloaders(self) -> None:
        updated_loaders = CurriculumDataloaders(
            train_dataloader=DataLoader(
                self.curriculum_datasets.train_dataset,  # type: ignore
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            ),
            val_dataloader=DataLoader(
                self.curriculum_datasets.val_dataset,  # type: ignore
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            ),
        )
        self.dataloaders = updated_loaders

    def get_dataloaders(self, epoch: int) -> Tuple[DataLoader, DataLoader]:
        train_dataloader, val_dataloader = None, None
        if self.is_curriculum:
            # Get the curriculum dataloaders
            train_dataloader, val_dataloader = self.get_curriculum_dataloaders(
                epoch
            )
        else:
            # Return the default dataloaders
            train_dataloader, val_dataloader = (
                self.dataloaders.train_dataloader,
                self.dataloaders.val_dataloader,
            )
        return train_dataloader, val_dataloader

    def get_curriculum_dataloaders(self, epoch: int) -> Tuple[DataLoader, DataLoader]:
        # Ensure we are within the available stages
        num_stages = len(self.curriculum_schedule.stages)
        while (
            self.upcoming_stage < num_stages
            and epoch >= self.curriculum_schedule.stages[self.upcoming_stage].start_epoch  # noqa: E501
        ):
            stage_schedule = self.curriculum_schedule.stages[self.upcoming_stage]
            self.logger.info(f"Moving to curriculum stage {self.upcoming_stage} with sigma {stage_schedule.sigma}")  # noqa: E501
            # Update datasets and dataloaders based on the new stage
            self.update_datasets(stage_schedule.sigma)
            self.update_dataloaders()
            # Move to the next stage
            self.upcoming_stage += 1
        return (
            self.dataloaders.train_dataloader,
            self.dataloaders.val_dataloader,
        )
