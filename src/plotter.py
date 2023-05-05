import pandas as pd
import matplotlib.pyplot as plt

def plot(plot_files, title, sparsity=10, directory='./cache/pt'):
    csv_dir = f'{directory}/logs/'
    out_dir = f'{directory}/outputs'

    for csv_file_name, plot_label in plot_files.items():
        df = pd.read_csv(f'{csv_dir}/{csv_file_name}.csv').iloc[::sparsity]

        x_col = 'epochs'
        y_col = 'rewards'
        plt.plot(df[x_col], df[y_col], label=plot_label)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title(title)

    plt.savefig(f'{out_dir}/{title}.png')
    plt.clf()
