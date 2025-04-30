import pytest
import h5py
from unittest.mock import MagicMock
from data.neural_code_data_manipulator import NeuralCodeData


@pytest.fixture
def temp_hdf5_file(tmp_path):
    # Create a temporary HDF5 file for testing
    file_path = tmp_path / "test_data.h5"
    with h5py.File(file_path, "w") as f:
        # Create mock data
        f.create_dataset("train/response/key1", data=[[1, 2, 3, 4, 5]])
        f.create_dataset("train/stimulus", data=[1, 2, 3, 4, 5])
        f.create_dataset("test/response/key1", data=[[6, 7, 8, 9, 10]])
        f.create_dataset("test/stimulus", data=[6, 7, 8, 9, 10])
    yield file_path


@pytest.fixture
def mock_logger():
    # Create a mock logger
    logger = MagicMock()
    return logger


def test_read_data(temp_hdf5_file, mock_logger):
    data = NeuralCodeData(temp_hdf5_file, mock_logger)
    data.read_data()
    assert "train/response/key1" in data.data
    assert "test/response/key1" in data.data


def test_read_data_error(temp_hdf5_file, mock_logger, monkeypatch):
    # Simulate an error during file reading
    monkeypatch.setattr(h5py, "File", lambda *args, **kwargs: 1 / 0)
    data = NeuralCodeData(temp_hdf5_file, mock_logger)
    with pytest.raises(SystemExit):
        data.read_data()
    mock_logger.error.assert_called()


def test_make_validation_split(temp_hdf5_file, mock_logger):
    data = NeuralCodeData(temp_hdf5_file, mock_logger)
    data.read_data()
    data.make_validation_split(0.6)

    new_path = temp_hdf5_file.parent / f"{temp_hdf5_file.stem}_with_val.h5"
    with h5py.File(new_path, "r") as f:
        assert "train/response/key1" in f
        assert "val/response/key1" in f
        assert "test/response/key1" in f
        assert "train/stimulus" in f
        assert "val/stimulus" in f
        assert "test/stimulus" in f

        train_data = f["train/response/key1"][:]
        val_data = f["val/response/key1"][:]
        test_data = f["test/response/key1"][:]
        train_stimulus = f["train/stimulus"][:]
        val_stimulus = f["val/stimulus"][:]
        test_stimulus = f["test/stimulus"][:]

        assert train_data.shape[1] == 3  # 60% of 5
        assert val_data.shape[1] == 2  # 40% of 5
        assert (train_data == [[1, 2, 3]]).all()
        assert (val_data == [[4, 5]]).all()
        assert (test_data == [[6, 7, 8, 9, 10]]).all()
        assert (train_stimulus == [1, 2, 3]).all()
        assert (val_stimulus == [4, 5]).all()
        assert (test_stimulus == [6, 7, 8, 9, 10]).all()
