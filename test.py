import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tutorial import Runner, DataLoader

def create_dataset(dataframe, label_name):
    col_name_list = [col_name for col_name in dataframe.columns[1:]]
    col_name_list.remove(label_name)
    input_data = np.array([dataframe[col_name] for col_name in col_name_list]).transpose()
    target_data = np.array(dataframe[label_name])
    return (input_data, target_data)

def plot_and_save(runner, save_name='test_plot'):
    sns.set_palette("GnBu_d")
    sns.set_style('whitegrid')
    plt.scatter(runner.y_test, runner.model.predict(runner.X_test))
    for h, metric_name in zip([0.77, 0.73, 0.69], runner.metric.keys()):
        plt.figtext(.18, h, f"{metric_name}: {np.round(runner.metric[metric_name], 4)}")
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.savefig(f"{save_name}.png")
    print(f"Plot is saved at {save_name}.png")

if __name__ == '__main__':
    # open example files and create data_loader
    file_name = "./data/Advertising.csv"
    df = pd.read_csv(file_name)
    dataset = create_dataset(df, 'Sales')

    # create data_dict
    data_dict = {'name': 'Sales', 'loader': DataLoader(dataset), 'task': 'regression'}
    
    runner = Runner(data_dict)
    runner.fit_model()
    runner.eval_model()
    plot_and_save(runner, 'actual_predict')


