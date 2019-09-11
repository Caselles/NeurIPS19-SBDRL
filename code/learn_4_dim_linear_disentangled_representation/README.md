How to use:

1) create repo named 'images/' and run: python flatland/flat_game/generate_data.py (with OpenAI venv)
2) run: python 01_train_vae.py (with gym-gpu venv)
3) run: python interactive_plot_4dim.py (play around with zqsd) and python interactive_plot_circles.py


The VAE has learned a linear disentangled representation of the environment. The architecture is specific (A-VAE).
