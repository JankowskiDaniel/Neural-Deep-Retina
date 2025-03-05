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
        # Store the default dataloaders
        self.dataloaders = curriculum_dataloaders

        if is_curriculum:
            # et the default dataloaders as initial updated dataloaders
            self.updated_dataloaders = curriculum_dataloaders
            # Check requirements for curriculum learning
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

    def update_dataloaders(self) -> CurriculumDataloaders:
        train_dataloader = DataLoader(
            self.curriculum_datasets.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        val_dataloader = DataLoader(
            self.curriculum_datasets.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        return CurriculumDataloaders(train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader)

    def get_dataloaders(self, epoch: int) -> Tuple[DataLoader, DataLoader]:
        """
        Retrieves the appropriate dataloaders for training and validation based on the
        current epoch and whether curriculum learning is enabled.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing
            the training and validation dataloaders.
        """
        train_dataloader, val_dataloader = None, None
        if self.is_curriculum:
            # Get the curriculum dataloaders
            cds = self.get_curriculum_dataloaders(epoch)
            train_dataloader, val_dataloader = (cds.train_dataloader,
                                                cds.val_dataloader)
        else:
            # Return the default dataloaders
            train_dataloader, val_dataloader = (
                self.dataloaders.train_dataloader,
                self.dataloaders.val_dataloader,
            )
        return train_dataloader, val_dataloader

    def get_curriculum_dataloaders(self, epoch: int) -> CurriculumDataloaders:
        # Get the schedule for the upcoming stage
        stage_schedule = self.curriculum_schedule.stages[self.upcoming_stage]

        # Ensure we are within the available stages
        num_stages = len(self.curriculum_schedule.stages)

        if not stage_schedule.is_smoothened:
            # If the stage is not smoothened, return the default dataloaders
            self.logger.info(f"Smoothening disabled at stage {self.upcoming_stage}")  # noqa: E501
            return self.dataloaders
        
        elif (
            self.upcoming_stage < num_stages
            and epoch >= self.curriculum_schedule.stages[self.upcoming_stage].start_epoch  # type: ignore # noqa: E501
        ):
            # If we enter the next stage, update the datasets and dataloaders
            self.logger.info(f"Moving to curriculum stage {self.upcoming_stage} with sigma {stage_schedule.sigma}")  # noqa: E501
            self.update_datasets(stage_schedule.sigma)  # type: ignore  # noqa: E501
            self.updated_dataloaders = self.update_dataloaders()
            # Move to the next stage
            self.upcoming_stage += 1
        
        return self.updated_dataloaders
