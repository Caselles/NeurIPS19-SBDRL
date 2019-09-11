mkdir images

python flatland/flat_game/generate_data.py

python 01_train_vae.py

python flatland/flat_game/generate_data_ff.py

python 02_train_forward.py

python interactive_plot.py