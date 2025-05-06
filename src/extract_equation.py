from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from utils.logger import get_logger
from data_models.config_models import Config
from utils import (
    load_model,
    load_data_handler,
)
import pysindy as ps
import hydra
from scipy.integrate import solve_ivp
import sys


def rhs(t, h, model: ps.SINDy):
    """
    Right-hand side of the ODE system.
    """
    # Reshape h to match the input shape of the model
    h = h.reshape(1, -1)
    # Compute the derivative using the SINDy model
    h_dot = model.predict(h)
    return h_dot.flatten()


def simulate_sindy_from_input(
    model, sindy, input_sequence, dt=0.01, T=1.0, num_steps=100
):
    """
    Simulate hidden dynamics from the initial latent using a SINDy model.

    Args:
        model: your full PyTorch model with an encoder
        sindy: trained PySINDy model
        input_sequence: a single input (B=1, T, C, H, W) or batched sequence
        dt: time step (s)
        T: total time to simulate (s)
        num_steps: number of time points (affects resolution)

    Returns:
        t: time points
        h_sim: hidden states over time (num_steps x hidden_dim)
    """
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.to(next(model.parameters()).device)
        latents = model.encoder(input_sequence)  # shape: (B, T, latent_dim)

        # Use the first latent in the sequence as initial input to CFC
        latent_seq = latents[0].unsqueeze(0)  # (1, T, latent_dim)
        h_seq, _ = model.predictor.cfc(latent_seq)
        h0 = h_seq[0, 0].cpu().numpy()  # first hidden state: shape (H,)

    # Simulate ODE using SINDy
    t_eval = np.linspace(0, T, num_steps)
    sol = solve_ivp(
        rhs,
        t_span=(0, T),
        y0=h0,
        t_eval=t_eval,
        args=(sindy,),
        method="Radau",
        rtol=1e-5,
        atol=1e-8,
    )
    print(t_eval.shape)
    print(f"Simulated {len(sol.t)} time points.")
    print(f"Final time: {sol.t[-1]} s")
    print(f"Final hidden state: {sol.y[:, -1]}")

    return sol.t, sol.y.T  # shape: (num_steps, hidden_dim)


def evaluate_sindy_on_dataset(
    model,
    sindy,
    data_loader,
    results_dir: Path,
    logger,
    dt=0.01,
    T=1.0,
    num_steps=1000,
    max_batches=1,
):
    """
    Evaluate SINDy model by predicting final hidden state from the initial one,
    and comparing it to the true final hidden state from the model.

    Args:
        model: your full PyTorch model (should contain encoder and predictor.cfc)
        sindy: trained PySINDy model
        data_loader: DataLoader for dataset
        results_dir: directory to save results
        logger: logger for logging information
        dt: time step for ODE solver
        T: total simulation time
        num_steps: resolution of ODE solver
        max_batches: number of batches to evaluate (for speed control)
    """
    model.eval()
    device = next(model.parameters()).device

    for batch_idx, (images, targets) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)  # shape: (B, T, C, H, W)

        with torch.no_grad():
            # Get encoder latents
            latents = model.encoder(images)  # shape: (B, T, latent_dim)

            # Get true hidden state sequence
            h_seq, _ = model.predictor.cfc(latents)  # shape: (B, T, H)
            h_seq = h_seq.cpu().numpy()

        cfc_preds = h_seq[:, -1]  # final hidden state from CFC
        sindy_preds = []  # predicted hidden states from SINDy
        targets = targets.cpu().numpy()  # shape: (B, T, C)

        for b in tqdm(range(h_seq.shape[0]), desc="Processing samples"):
            h0 = h_seq[b, 0]  # initial hidden state

            # Simulate ODE with SINDy
            t_eval = np.linspace(0, T, num_steps)
            sol = solve_ivp(
                rhs,
                t_span=(0, T),
                y0=h0,
                t_eval=t_eval,
                args=(sindy,),
                method="RK45",
            )

            h_sindy = sol.y[:, -1]  # final hidden state from simulation

            sindy_preds.append(h_sindy)

    sindy_preds_array = np.array(sindy_preds)

    # Calculate RELU on predictions
    sindy_preds_array = np.maximum(sindy_preds_array, 0)
    cfc_preds = np.maximum(cfc_preds, 0)
    targets = targets.cpu().numpy()

    # Calculate metrics
    calculate_metrics(sindy_preds_array, cfc_preds, targets, logger)

    # Save predictions
    pd.DataFrame(sindy_preds).to_csv(
        results_dir / "sindy_preds.csv", index=False
    )
    pd.DataFrame(cfc_preds).to_csv(results_dir / "cfc_preds.csv", index=False)
    pd.DataFrame(targets).to_csv(results_dir / "targets.csv", index=False)


def calculate_metrics(sindy_preds, cfc_preds, targets, logger):
    """
    Calculate metrics for SINDy predictions.

    Args:
        sindy_preds: predicted hidden states from SINDy
        cfc_preds: predicted hidden states from CFC
        targets: true hidden states
        logger: logger for logging information
    """
    # Calculate mse
    mse_sindy = np.mean((sindy_preds - targets) ** 2)
    mse_cfc = np.mean((cfc_preds - targets) ** 2)
    logger.info(f"MSE (SINDy): {mse_sindy:.4f}")
    logger.info(f"MSE (CFC): {mse_cfc:.4f}")

    # Calculate Pearson correlation
    corr_sindy = np.corrcoef(sindy_preds.flatten(), targets.flatten())[0, 1]
    corr_cfc = np.corrcoef(cfc_preds.flatten(), targets.flatten())[0, 1]

    logger.info(f"Pearson correlation (SINDy): {corr_sindy:.4f}")
    logger.info(f"Pearson correlation (CFC): {corr_cfc:.4f}")


def plot_sindy_results(t, h_sim, results_dir: Path) -> None:
    """
    Plot the results of the SINDy simulation.

    Args:
        t: time points
        h_sim: simulated hidden states
        sindy: trained SINDy model
    """
    plt.figure(figsize=(10, 6))
    for i in range(h_sim.shape[1]):
        plt.plot(t, h_sim[:, i], label=f"h{i}")
    plt.title("Simulated Hidden State Dynamics (SINDy)")
    plt.xlabel("Time")
    plt.ylabel("Hidden States")
    plt.legend()
    plt.savefig(results_dir / "sindy_simulation.png")
    plt.close()


@hydra.main(
    config_path="../configs", config_name="config_sindy", version_base=None
)
def train(config: Config) -> None:

    results_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        log_to_file=config.training.save_logs,
        log_file=results_dir / "train_logs.log",
    )

    logger.info(f"Results directory: {results_dir}")

    # load the model
    logger.info("Loading model...")

    if config.training.encoder.weights is not None:
        logger.info(
            f"Loading encoder weights from {config.training.encoder.weights}"
        )
    if config.training.predictor.weights is not None:
        logger.info(
            f"Loading predictor weights from {config.training.predictor.weights}"
        )
    model = load_model(config)

    y_scaler = MinMaxScaler()

    # load the data handler
    logger.info("Loading data handler...")
    train_dataset = load_data_handler(
        config.data,
        results_dir=results_dir,
        is_train=True,
        y_scaler=y_scaler,
    )
    val_dataset = load_data_handler(
        config.data,
        results_dir=results_dir,
        is_train=False,
        subset_type="val",
        use_saved_scaler=True,
        y_scaler=y_scaler,
    )

    # Get sample data to check dimensions
    X, y = train_dataset[0]
    logger.info(f"Input data point shape: {X.shape}")
    logger.info(f"Target data point shape: {y.shape}")

    logger.info(
        f"Predictions will be made for channels: {train_dataset.pred_channels}"
    )
    TORCH_SEED = 12

    logger.info(f"Manually set PyTorch seed: {TORCH_SEED}")
    torch.manual_seed(TORCH_SEED)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Define training parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = False
    NUM_WORKERS = 0
    BATCH_SIZE = config.training.batch_size

    # Define data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    model.predictor.cfc.return_sequences = True

    model.to(DEVICE)
    model.eval()

    all_h_seq = []
    all_u_seq = []  # only if you want to include inputs u in your SINDy model

    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(DEVICE)  # shape: (B, T, C, H, W)
            # assume encoder is part of `model` or apply it yourself:
            latents = model.encoder(images)  # -> (B, T, input_size)

            # cfc returns (sequence, final_state)
            # seq: (B, T, hidden_size)
            h_seq, _ = model.predictor.cfc(latents)
            h_seq = h_seq.cpu().numpy()  # -> (B, T, H)

            # optionally collect inputs u(t):
            u_seq = latents.cpu().numpy()  # -> (B, T, input_size)

            # flatten batch‐and‐time into a long list of states
            B, T, H = h_seq.shape
            all_h_seq.append(h_seq.reshape(B * T, H))
            all_u_seq.append(u_seq.reshape(B * T, -1))

    all_h_seq = np.concatenate(all_h_seq, axis=0)  # shape: (N_total, H)
    all_u_seq = np.concatenate(all_u_seq, axis=0)

    # SINDY model
    dt = 0.01  # 10 ms time step
    all_h_dot = np.gradient(all_h_seq, dt, axis=0)

    poly_lib = ps.PolynomialLibrary(degree=3, include_interaction=True)
    fourier_lib = ps.FourierLibrary(n_frequencies=5)
    library = poly_lib + fourier_lib

    optimizer = ps.STLSQ(threshold=0.1, alpha=0.0)

    sindy = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        differentiation_method=ps.FiniteDifference(),
        discrete_time=False,
        feature_names=[f"h{i}" for i in range(all_h_seq.shape[1])],  # type: ignore
    )

    sindy.fit(all_h_seq, t=dt, x_dot=all_h_dot)
    with open(results_dir / "sindy_equations.txt", "w") as f:
        sys.stdout = f  # Redirect standard output to the file
        sindy.print(lhs=None, precision=3)
        sys.stdout = sys.__stdout__  # Restore standard output
    score = sindy.score(all_h_seq, t=dt, x_dot=all_h_dot)
    print(f"R² on training data: {score:.3f}")

    images, _ = next(iter(train_loader))  # shape: (B, T, C, H, W)
    images = images[:1]  # take just one sample for simulation

    # Simulate
    logger.info("Simulating hidden dynamics from input...")
    t, h_sim = simulate_sindy_from_input(
        model, sindy, images, dt=0.01, T=100, num_steps=10000
    )

    plot_sindy_results(t, h_sim, results_dir)

    # Evaluate SINDy on dataset
    logger.info("Evaluating SINDy on dataset...")
    evaluate_sindy_on_dataset(
        model,
        sindy,
        val_loader,
        results_dir=results_dir,
        logger=logger,
        dt=0.01,
        T=100,
        num_steps=10000,
        max_batches=1,
    )


if __name__ == "__main__":

    train()
