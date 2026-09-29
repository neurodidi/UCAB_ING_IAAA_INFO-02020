"""
Export the two perceptron training animations from W3_D2_Perceptron.ipynb
as both .gif and .mp4 files.

Usage:
    python export_perceptron_animations.py

Outputs (saved in the same directory as this script):
    - logistic_curve_animation.gif
    - logistic_curve_animation.mp4
    - gaussian_curve_animation.gif
    - gaussian_curve_animation.mp4
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ─────────────────────────────────────────────────────────
# Perceptron class (copied from notebook cell 1)
# ─────────────────────────────────────────────────────────
class PerceptronScratch:
    def __init__(self, learning_rate=0.1, n_epochs=10, activation='step', historial=False):
        """
        activation: 'step' (Clasificación), 'linear' (Regresión), 'sigmoid' (Regresión Logística/No lineal)
        historial: True para guardar los pesos de cada época (útil para animaciones)
        """
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.historial = historial
        self.weights = None
        self.bias = None
        self.historial_weights = []

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.historial_weights = []

        y = np.ravel(y)

        for epoch in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias

                if self.activation == 'step':
                    y_predicted = np.where(linear_output >= 0, 1, 0)
                    update = self.lr * (y[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update

                elif self.activation == 'linear':
                    y_predicted = linear_output
                    error = y_predicted - y[idx]
                    self.weights -= self.lr * 2 * error * x_i
                    self.bias -= self.lr * 2 * error

                elif self.activation == 'sigmoid':
                    y_predicted = self._sigmoid(linear_output)
                    error = y_predicted - y[idx]
                    d_pred = error * self._sigmoid_derivative(y_predicted)
                    self.weights -= self.lr * 2 * d_pred * x_i
                    self.bias -= self.lr * 2 * d_pred

            if self.historial:
                self.historial_weights.append((self.weights.copy(), self.bias))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        if self.activation == 'step':
            return np.where(linear_output >= 0, 1, 0)
        elif self.activation == 'linear':
            return linear_output
        elif self.activation == 'sigmoid':
            return self._sigmoid(linear_output)


# ─────────────────────────────────────────────────────────
# Helper: save animation to both .gif and .mp4
# ─────────────────────────────────────────────────────────
def save_animation(anim, base_name, output_dir, fps=25, dpi=150):
    """Save a FuncAnimation as .gif and .mp4"""

    gif_path = os.path.join(output_dir, f"{base_name}.gif")
    mp4_path = os.path.join(output_dir, f"{base_name}.mp4")

    # Save as GIF (uses Pillow)
    print(f"  Saving {gif_path} ...")
    anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
    print(f"  ✓ GIF saved ({os.path.getsize(gif_path) / 1024 / 1024:.1f} MB)")

    # Save as MP4 (uses ffmpeg)
    # Locate ffmpeg from the conda env (may not be on system PATH)
    import sys
    env_bin = os.path.dirname(sys.executable)
    ffmpeg_path = os.path.join(env_bin, 'ffmpeg')
    if not os.path.isfile(ffmpeg_path):
        # Fallback: try system ffmpeg
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'])
    print(f"  Saving {mp4_path} ...")
    anim.save(mp4_path, writer=writer, dpi=dpi)
    print(f"  ✓ MP4 saved ({os.path.getsize(mp4_path) / 1024 / 1024:.1f} MB)")


# ─────────────────────────────────────────────────────────
# Animation 1: Logistic Curve (notebook cell 3)
# ─────────────────────────────────────────────────────────
def build_logistic_animation():
    print("\n[1/2] Building Logistic Curve animation...")

    np.random.seed(42)
    X = np.sort(np.random.uniform(-10, 10, 300)).reshape(-1, 1)

    z_true = 0.8 * X - 1.5
    y_true = 1 / (1 + np.exp(-z_true))
    y = y_true + np.random.normal(0, 0.07, X.shape)

    epochs = 500
    linear_model = PerceptronScratch(learning_rate=0.002, n_epochs=epochs,
                                     activation="linear", historial=True)
    sigmoid_model = PerceptronScratch(learning_rate=0.002, n_epochs=epochs,
                                      activation="sigmoid", historial=True)

    linear_model.fit(X, y)
    sigmoid_model.fit(X, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    X_plot = np.linspace(-10, 10, 300).reshape(-1, 1)

    ax.scatter(X, y, color='gray', alpha=0.3,
               label=r"Adjusted curve $\rho + \mathcal{N}(0, \sigma^2$)")
    ax.plot(X_plot, y_true, color='green', linestyle=':', linewidth=2,
            label=r'True function ($\rho$)')

    lin_line, = ax.plot([], [], color='red', linewidth=3,
                        label='Linear function perceptron')
    sig_line, = ax.plot([], [], color='blue', linewidth=3,
                        label='Sigmoid perceptron')

    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-0.3, 1.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.axis('off')
    title = ax.set_title("Epoch: 0", fontsize=14)

    def init():
        lin_line.set_data([], [])
        sig_line.set_data([], [])
        title.set_text("Epoch: 0")
        return lin_line, sig_line, title

    def update(frame):
        w_lin, b_lin = linear_model.historial_weights[frame]
        y_pred_lin = np.dot(X_plot, w_lin) + b_lin
        lin_line.set_data(X_plot, y_pred_lin)

        w_sig, b_sig = sigmoid_model.historial_weights[frame]
        z_sig = np.dot(X_plot, w_sig) + b_sig
        y_pred_sig = 1 / (1 + np.exp(-np.clip(z_sig, -250, 250)))
        sig_line.set_data(X_plot, y_pred_sig)

        title.set_text(f"Training evolution - Epoch: {frame}")
        return lin_line, sig_line, title

    anim = FuncAnimation(fig, update,
                         frames=len(linear_model.historial_weights),
                         init_func=init, blit=True, interval=40)
    plt.close(fig)
    return anim


# ─────────────────────────────────────────────────────────
# Animation 2: Gaussian Curve (notebook cell 4)
# ─────────────────────────────────────────────────────────
def build_gaussian_animation():
    print("\n[2/2] Building Gaussian Curve animation...")

    np.random.seed(42)
    X = np.sort(np.random.uniform(-10, 10, 300)).reshape(-1, 1)

    true_y = np.exp(-0.5 * (X / 2.5)**2)
    y = true_y + np.random.normal(0, 0.07, X.shape)

    epochs = 500
    linear_model = PerceptronScratch(learning_rate=0.002, n_epochs=epochs,
                                     activation="linear", historial=True)
    sigmoid_model = PerceptronScratch(learning_rate=0.05, n_epochs=epochs,
                                      activation="sigmoid", historial=True)

    linear_model.fit(X, y)
    sigmoid_model.fit(X, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    X_plot = np.linspace(-10, 10, 300).reshape(-1, 1)

    ax.scatter(X, y, color='gray', alpha=0.3,
               label=r"Adjusted curve $\mathcal{N}(0, \sigma^2)  + \eta$")
    ax.plot(X_plot, true_y, color='green', linestyle=':', linewidth=2,
            label=r'True function ($\mathcal{N}(0, \sigma^2)$')

    lin_line, = ax.plot([], [], color='red', linewidth=3,
                        label='Linear function perceptron')
    sig_line, = ax.plot([], [], color='blue', linewidth=3,
                        label='Sigmoid perceptron')

    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-0.3, 1.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')
    title = ax.set_title("Epoch: 0", fontsize=14)

    def init():
        lin_line.set_data([], [])
        sig_line.set_data([], [])
        title.set_text("Epoch: 0")
        return lin_line, sig_line, title

    def update(frame):
        w_lin, b_lin = linear_model.historial_weights[frame]
        y_pred_lin = np.dot(X_plot, w_lin) + b_lin
        lin_line.set_data(X_plot, y_pred_lin)

        w_sig, b_sig = sigmoid_model.historial_weights[frame]
        z_sig = np.dot(X_plot, w_sig) + b_sig
        y_pred_sig = 1 / (1 + np.exp(-np.clip(z_sig, -250, 250)))
        sig_line.set_data(X_plot, y_pred_sig)

        title.set_text(f"Training evolution - Epoch: {frame}")
        return lin_line, sig_line, title

    anim = FuncAnimation(fig, update,
                         frames=len(linear_model.historial_weights),
                         init_func=init, blit=True, interval=40)
    plt.close(fig)
    return anim


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Output directory: {output_dir}")

    anim1 = build_logistic_animation()
    save_animation(anim1, "logistic_curve_animation", output_dir)

    anim2 = build_gaussian_animation()
    save_animation(anim2, "gaussian_curve_animation", output_dir)

    print("\n✅ All animations exported successfully!")
